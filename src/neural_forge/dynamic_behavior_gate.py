"""Cached-parent whole-output promotion gates for folded children.

Exact token or exact text equality is diagnostic only. The hard contract is
that a child must not introduce critical behavioral failures such as a new
empty response, a newly fragmentary response, an EOS-margin collapse, missing
cases, or a deterministic task-correctness regression.

:class:`WholeOutputCase` and :class:`CandidateWholeOutput` describe text-only
review.  :class:`MultimodalWholeOutputCase`,
:class:`MultimodalCandidateOutput`, and :class:`MediaReference` describe the
same review contract when a case carries decoded media on either side.  The
media types are strictly additive: they reuse the identity, coverage,
threshold, and manifest machinery of the text gate so that a new modality lane
does not have to clone a gate to describe its cases.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dynamic_behavior_execution import (
    BehaviorInferenceExecutionContract,
    DynamicBehaviorExecutionError,
    execution_contract_satisfied,
    inference_execution_contract_from_manifest,
    observe_model_execution,
    resolve_behavior_inference_execution_contract,
)
from .dynamic_circuit_ir import (
    MODALITY_AUDIO,
    MODALITY_IMAGE,
    MODALITY_TEXT,
    DynamicCircuitIRError,
    JsonScalar,
    canonical_modality,
    canonical_sha256,
)
from .dynamic_fold_executor import (
    FoldDecision,
    FoldEvaluationContext,
)


SCHEMA = "neural-forge-cached-whole-output-gate-v2"
MANIFEST_SCHEMA = "neural-forge-behavior-gate-manifest-v2"
MEDIA_REFERENCE_SCHEMA = "neural-forge-media-reference-v1"
MULTIMODAL_CASE_SCHEMA = "neural-forge-multimodal-whole-output-case-v1"
MULTIMODAL_CANDIDATE_SCHEMA = "neural-forge-multimodal-candidate-output-v1"
MULTIMODAL_BASELINE_SCHEMA = "neural-forge-multimodal-whole-output-gate-v1"
MULTIMODAL_COVERAGE_SCHEMA = "neural-forge-multimodal-case-coverage-v1"

# Decoded-content hash bases.  A media reference binds what a model actually
# consumes, never the container: two PNG or WAV encodings of the same decoded
# payload share one content digest, and a re-encoded container that changes a
# single decoded sample fails closed.  ``IMAGE_HASH_BASIS`` is byte-identical
# to the basis string already sealed into NEO native-vision probe manifests.
IMAGE_HASH_BASIS = "sha256('RGB:<width>x<height>\\n' + decoded RGB bytes)"
AUDIO_HASH_BASIS = (
    "sha256('PCM16:<rate>:<channels>:<frames>\\n' + little-endian int16 samples)"
)
TEXT_HASH_BASIS = "sha256('TEXT:utf-8\\n' + UTF-8 bytes)"
CANONICAL_MEDIA_HASH_BASES: Mapping[str, str] = MappingProxyType(
    {
        MODALITY_TEXT: TEXT_HASH_BASIS,
        MODALITY_IMAGE: IMAGE_HASH_BASIS,
        MODALITY_AUDIO: AUDIO_HASH_BASIS,
    }
)


DynamicBehaviorGateError = DynamicBehaviorExecutionError


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicBehaviorGateError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicBehaviorGateError(f"{name} must be a positive integer.")
    return value


def _canonical_copy(value: Any, name: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise DynamicBehaviorGateError(f"{name} must be canonical JSON.") from exc


def _type_name(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _stable_resource_value(value: Any, *, depth: int = 0) -> Any:
    """Describe callable resources without object addresses or mutable reprs."""

    if depth > 5:
        return {"type": _type_name(value), "truncated": True}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"type": "float", "value": str(value)}
        return value
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, Path):
        return {"type": _type_name(value), "path": str(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _stable_resource_value(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable_resource_value(item, depth=depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_stable_resource_value(item, depth=depth + 1) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": _type_name(value),
            "fields": {
                descriptor.name: _stable_resource_value(
                    getattr(value, descriptor.name),
                    depth=depth + 1,
                )
                for descriptor in fields(value)
            },
        }
    for name in (
        "behavior_resource_fingerprint",
        "resource_fingerprint",
        "baseline_fingerprint",
        "fingerprint",
        "cache_key",
        "source_sha256",
    ):
        candidate = getattr(value, name, None)
        if isinstance(candidate, str) and len(candidate) == 64:
            return {
                "type": _type_name(value),
                "identity_field": name,
                "identity": candidate,
            }
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        public = {
            str(name): _stable_resource_value(item, depth=depth + 1)
            for name, item in sorted(attributes.items())
            if not str(name).startswith("_")
        }
        if public:
            return {"type": _type_name(value), "attributes": public}
    return {"type": _type_name(value)}


def _callable_function(value: Callable[..., Any]) -> Callable[..., Any]:
    target = getattr(value, "__func__", None)
    if callable(target):
        return target
    if inspect.isfunction(value) or inspect.isbuiltin(value):
        return value
    call = getattr(type(value), "__call__", None)
    return call if callable(call) else value


def _callable_code_fingerprint(value: Callable[..., Any]) -> str:
    target = _callable_function(value)
    module = str(getattr(target, "__module__", type(value).__module__))
    qualname = str(
        getattr(target, "__qualname__", type(value).__qualname__ + ".__call__")
    )
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        source = ""
    code = getattr(target, "__code__", None)
    code_payload: dict[str, Any] | None = None
    if code is not None:
        code_payload = {
            "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
            "constants": _stable_resource_value(code.co_consts),
            "names": list(code.co_names),
            "variable_names": list(code.co_varnames),
            "positional_argument_count": code.co_argcount,
            "keyword_only_argument_count": code.co_kwonlyargcount,
        }
    return canonical_sha256(
        {
            "schema": "neural-forge-behavior-callable-code-v1",
            "module": module,
            "qualname": qualname,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "code": code_payload,
        }
    )


def _callable_resource_fingerprint(value: Callable[..., Any]) -> str:
    target = _callable_function(value)
    closure = getattr(target, "__closure__", None)
    closure_values = []
    for cell in closure or ():
        try:
            closure_values.append(_stable_resource_value(cell.cell_contents))
        except ValueError:
            closure_values.append({"empty_cell": True})
    bound_owner = getattr(value, "__self__", None)
    if bound_owner is None and not inspect.isfunction(value):
        bound_owner = value
    return canonical_sha256(
        {
            "schema": "neural-forge-behavior-callable-resources-v1",
            "defaults": _stable_resource_value(getattr(target, "__defaults__", None)),
            "keyword_defaults": _stable_resource_value(
                getattr(target, "__kwdefaults__", None)
            ),
            "closure": closure_values,
            "bound_owner": _stable_resource_value(bound_owner),
        }
    )


@dataclass(frozen=True, slots=True)
class BehaviorCallableIdentity:
    """Registered implementation and immutable resource identity."""

    implementation_id: str
    version: int
    code_fingerprint: str
    resource_fingerprint: str
    registered: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "implementation_id",
            _text(self.implementation_id, "implementation_id"),
        )
        object.__setattr__(self, "version", _positive_int(self.version, "version"))
        object.__setattr__(
            self,
            "code_fingerprint",
            _digest(self.code_fingerprint, "code_fingerprint"),
        )
        object.__setattr__(
            self,
            "resource_fingerprint",
            _digest(self.resource_fingerprint, "resource_fingerprint"),
        )
        if not isinstance(self.registered, bool):
            raise DynamicBehaviorGateError("registered must be boolean.")

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "implementation_id": self.implementation_id,
            "version": self.version,
            "code_fingerprint": self.code_fingerprint,
            "resource_fingerprint": self.resource_fingerprint,
            "registered": self.registered,
        }


def identify_behavior_callable(
    value: Callable[..., Any],
    *,
    implementation_id: str | None = None,
    version: int = 1,
    resource_fingerprint: str | None = None,
    code_fingerprint: str | None = None,
) -> BehaviorCallableIdentity:
    """Create a stable callable identity; explicit IDs mark it registered."""

    if not callable(value):
        raise DynamicBehaviorGateError("Behavior implementation must be callable.")
    target = _callable_function(value)
    default_id = (
        f"{getattr(target, '__module__', type(value).__module__)}."
        f"{getattr(target, '__qualname__', type(value).__qualname__ + '.__call__')}"
    )
    registered = implementation_id is not None
    observed_code_fingerprint = _callable_code_fingerprint(value)
    observed_resource_fingerprint = _callable_resource_fingerprint(value)
    if (
        code_fingerprint is not None
        and code_fingerprint != observed_code_fingerprint
    ):
        raise DynamicBehaviorGateError(
            "Declared behavior callable code fingerprint differs from the "
            "compiler-observed implementation."
        )
    if (
        resource_fingerprint is not None
        and resource_fingerprint != observed_resource_fingerprint
    ):
        raise DynamicBehaviorGateError(
            "Declared behavior callable resource fingerprint differs from the "
            "compiler-observed bindings."
        )
    return BehaviorCallableIdentity(
        implementation_id=default_id
        if implementation_id is None
        else implementation_id,
        version=version,
        code_fingerprint=(
            observed_code_fingerprint
            if code_fingerprint is None
            else code_fingerprint
        ),
        resource_fingerprint=(
            observed_resource_fingerprint
            if resource_fingerprint is None
            else resource_fingerprint
        ),
        registered=registered,
    )


def _verify_behavior_callable_identity(
    value: Callable[..., Any],
    identity: BehaviorCallableIdentity,
    *,
    name: str,
) -> None:
    observed_code = _callable_code_fingerprint(value)
    observed_resources = _callable_resource_fingerprint(value)
    if (
        identity.code_fingerprint != observed_code
        or identity.resource_fingerprint != observed_resources
    ):
        raise DynamicBehaviorGateError(
            f"{name} identity differs from the compiler-observed callable "
            "implementation or resources."
        )


@dataclass(frozen=True, slots=True)
class BehaviorGateBinding:
    """Canonical manifest shared by in-memory and fresh-reload evaluators."""

    gate_id: str
    gate_version: int
    baseline_fingerprint: str
    manifest_fingerprint: str
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_id", _text(self.gate_id, "gate_id"))
        object.__setattr__(
            self,
            "gate_version",
            _positive_int(self.gate_version, "gate_version"),
        )
        object.__setattr__(
            self,
            "baseline_fingerprint",
            _digest(self.baseline_fingerprint, "baseline_fingerprint"),
        )
        object.__setattr__(
            self,
            "manifest_fingerprint",
            _digest(self.manifest_fingerprint, "manifest_fingerprint"),
        )
        normalized = _canonical_copy(dict(self.manifest), "behavior gate manifest")
        seal = normalized.pop("manifest_fingerprint", None)
        try:
            inference_execution_contract_from_manifest(normalized)
        except DynamicBehaviorExecutionError as exc:
            raise DynamicBehaviorGateError(
                "Behavior gate inference execution contract is invalid."
            ) from exc
        if (
            seal != self.manifest_fingerprint
            or canonical_sha256(normalized) != self.manifest_fingerprint
            or normalized.get("gate_id") != self.gate_id
            or normalized.get("gate_version") != self.gate_version
            or normalized.get("baseline_fingerprint") != self.baseline_fingerprint
        ):
            raise DynamicBehaviorGateError("Behavior gate manifest seal is invalid.")
        normalized["manifest_fingerprint"] = seal
        object.__setattr__(self, "manifest", MappingProxyType(normalized))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.manifest)

    def fields(self) -> dict[str, Any]:
        return {
            "behavior_gate_id": self.gate_id,
            "behavior_gate_version": self.gate_version,
            "behavior_baseline_fingerprint": self.baseline_fingerprint,
            "behavior_gate_manifest_fingerprint": self.manifest_fingerprint,
            "behavior_gate_manifest": self.as_dict(),
        }


def build_behavior_gate_binding(
    *,
    gate_id: str,
    gate_version: int,
    baseline_fingerprint: str,
    threshold_config: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    generator_identity: BehaviorCallableIdentity,
    scorer_identity: BehaviorCallableIdentity | None,
    coverage_policy: Mapping[str, Any],
    components: Sequence[Mapping[str, Any]] = (),
    inference_execution_contract: (
        BehaviorInferenceExecutionContract | None
    ) = None,
) -> BehaviorGateBinding:
    if not isinstance(generator_identity, BehaviorCallableIdentity):
        raise DynamicBehaviorGateError(
            "generator_identity must be BehaviorCallableIdentity."
        )
    if scorer_identity is not None and not isinstance(
        scorer_identity,
        BehaviorCallableIdentity,
    ):
        raise DynamicBehaviorGateError(
            "scorer_identity must be BehaviorCallableIdentity or None."
        )
    selected_execution_contract = (
        BehaviorInferenceExecutionContract.top_level_forward()
        if inference_execution_contract is None
        else inference_execution_contract
    )
    if not isinstance(
        selected_execution_contract,
        BehaviorInferenceExecutionContract,
    ):
        raise DynamicBehaviorGateError(
            "inference_execution_contract must be "
            "BehaviorInferenceExecutionContract or None."
        )
    body = _canonical_copy(
        {
            "schema": MANIFEST_SCHEMA,
            "gate_id": _text(gate_id, "gate_id"),
            "gate_version": _positive_int(gate_version, "gate_version"),
            "baseline_fingerprint": _digest(
                baseline_fingerprint,
                "baseline_fingerprint",
            ),
            "threshold_config": dict(threshold_config),
            "coverage_policy": dict(coverage_policy),
            "cases": [dict(value) for value in cases],
            "baseline": dict(baseline),
            "generator": generator_identity.as_dict(),
            "scorer": (None if scorer_identity is None else scorer_identity.as_dict()),
            "components": [dict(value) for value in components],
            "inference_execution": selected_execution_contract.as_dict(),
        },
        "behavior gate manifest",
    )
    fingerprint = canonical_sha256(body)
    manifest = {**body, "manifest_fingerprint": fingerprint}
    return BehaviorGateBinding(
        gate_id=body["gate_id"],
        gate_version=body["gate_version"],
        baseline_fingerprint=body["baseline_fingerprint"],
        manifest_fingerprint=fingerprint,
        manifest=manifest,
    )


def expose_behavior_gate_binding(owner: Any, binding: BehaviorGateBinding) -> None:
    if not isinstance(binding, BehaviorGateBinding):
        raise DynamicBehaviorGateError("binding must be BehaviorGateBinding.")
    owner._behavior_gate_binding = binding
    owner.behavior_gate_id = binding.gate_id
    owner.behavior_gate_version = binding.gate_version
    owner.baseline_fingerprint = binding.baseline_fingerprint
    owner.behavior_baseline_fingerprint = binding.baseline_fingerprint
    owner.behavior_gate_manifest_fingerprint = binding.manifest_fingerprint
    owner.behavior_gate_manifest = binding.manifest
    owner.behavior_inference_execution_contract = (
        inference_execution_contract_from_manifest(binding.manifest)
    )


def resolve_behavior_gate_binding(value: Any) -> BehaviorGateBinding:
    """Resolve the same manifest from an evaluator object or its bound method."""

    candidates = [value]
    bound_owner = getattr(value, "__self__", None)
    if bound_owner is not None:
        candidates.append(bound_owner)
    explicit_owner = getattr(value, "behavior_gate_owner", None)
    if explicit_owner is not None:
        candidates.append(explicit_owner)
    for candidate in candidates:
        binding = getattr(candidate, "_behavior_gate_binding", None)
        if isinstance(binding, BehaviorGateBinding):
            return binding
    raise DynamicBehaviorGateError(
        "Evaluator does not expose a canonical behavior gate manifest."
    )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicBehaviorGateError(f"{name} must be a nonempty string.")
    return value.strip()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicBehaviorGateError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicBehaviorGateError(f"{name} must be finite.")
    return result


def _task_score(value: bool | float, name: str) -> float:
    result = float(int(value)) if isinstance(value, bool) else _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise DynamicBehaviorGateError(
            f"{name} must be a boolean or a score between zero and one."
        )
    return result


@dataclass(frozen=True, slots=True)
class WholeOutputMetadata:
    """Mechanical rollout metadata used for parent-relative regression checks."""

    generated_token_count: int | None = None
    generation_limit: int | None = None
    maximum_repeated_ngram_share: float | None = None

    def __post_init__(self) -> None:
        if self.generated_token_count is not None:
            value = self.generated_token_count
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DynamicBehaviorGateError(
                    "generated_token_count must be a non-negative integer."
                )
        if self.generation_limit is not None:
            value = self.generation_limit
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DynamicBehaviorGateError(
                    "generation_limit must be a positive integer."
                )
        if (
            self.generated_token_count is not None
            and self.generation_limit is not None
            and self.generated_token_count > self.generation_limit
        ):
            raise DynamicBehaviorGateError(
                "generated_token_count cannot exceed generation_limit."
            )
        if self.maximum_repeated_ngram_share is not None:
            value = _finite(
                self.maximum_repeated_ngram_share,
                "maximum_repeated_ngram_share",
            )
            if not 0.0 <= value <= 1.0:
                raise DynamicBehaviorGateError(
                    "maximum_repeated_ngram_share must be between zero and one."
                )
            object.__setattr__(self, "maximum_repeated_ngram_share", value)

    @property
    def generation_limit_saturated(self) -> bool | None:
        if self.generated_token_count is None or self.generation_limit is None:
            return None
        return self.generated_token_count == self.generation_limit

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_token_count": self.generated_token_count,
            "generation_limit": self.generation_limit,
            "generation_limit_saturated": self.generation_limit_saturated,
            "maximum_repeated_ngram_share": self.maximum_repeated_ngram_share,
        }


@dataclass(frozen=True, slots=True)
class WholeOutputCase:
    case_id: str
    prompt: str
    parent_output: str
    category: str = "unspecified"
    parent_fragment_like: bool = False
    parent_eos_margin: float | None = None
    critical_nonempty: bool = False
    parent_metadata: WholeOutputMetadata | None = None
    split: str = "unspecified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(self, "prompt", _text(self.prompt, "prompt"))
        if not isinstance(self.parent_output, str):
            raise DynamicBehaviorGateError("parent_output must be a string.")
        object.__setattr__(
            self,
            "category",
            _text(self.category, "category"),
        )
        if not isinstance(self.parent_fragment_like, bool):
            raise DynamicBehaviorGateError("parent_fragment_like must be boolean.")
        if self.parent_eos_margin is not None:
            object.__setattr__(
                self,
                "parent_eos_margin",
                _finite(self.parent_eos_margin, "parent_eos_margin"),
            )
        if not isinstance(self.critical_nonempty, bool):
            raise DynamicBehaviorGateError("critical_nonempty must be boolean.")
        if self.parent_metadata is not None and not isinstance(
            self.parent_metadata,
            WholeOutputMetadata,
        ):
            raise DynamicBehaviorGateError(
                "parent_metadata must be WholeOutputMetadata when supplied."
            )
        object.__setattr__(self, "split", _text(self.split, "split"))

    @property
    def parent_empty(self) -> bool:
        return not self.parent_output.strip()

    def as_dict(self) -> dict[str, Any]:
        result = {
            "case_id": self.case_id,
            "prompt": self.prompt,
            "parent_output": self.parent_output,
            "category": self.category,
            "parent_fragment_like": self.parent_fragment_like,
            "parent_eos_margin": self.parent_eos_margin,
            "critical_nonempty": self.critical_nonempty,
            "split": self.split,
        }
        if self.parent_metadata is not None:
            result["parent_metadata"] = self.parent_metadata.as_dict()
        return result


@dataclass(frozen=True, slots=True)
class CandidateWholeOutput:
    output: str
    fragment_like: bool = False
    eos_margin: float | None = None
    metadata: WholeOutputMetadata | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output, str):
            raise DynamicBehaviorGateError("Candidate output must be a string.")
        if not isinstance(self.fragment_like, bool):
            raise DynamicBehaviorGateError("fragment_like must be boolean.")
        if self.eos_margin is not None:
            object.__setattr__(
                self,
                "eos_margin",
                _finite(self.eos_margin, "candidate eos_margin"),
            )
        if self.metadata is not None and not isinstance(
            self.metadata,
            WholeOutputMetadata,
        ):
            raise DynamicBehaviorGateError(
                "Candidate metadata must be WholeOutputMetadata when supplied."
            )

    @property
    def empty(self) -> bool:
        return not self.output.strip()


def media_content_sha256(header: str, payload: bytes) -> str:
    """Hash decoded media as ``sha256(header + '\\n' + decoded payload)``."""

    label = _text(header, "media content header")
    if not label.isascii() or "\n" in label:
        raise DynamicBehaviorGateError(
            "media content header must be one ASCII line without a newline."
        )
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise DynamicBehaviorGateError("media content payload must be bytes-like.")
    digest = hashlib.sha256()
    digest.update(f"{label}\n".encode("ascii"))
    digest.update(bytes(payload))
    return digest.hexdigest()


def image_content_sha256(width: int, height: int, rgb_bytes: bytes) -> str:
    """Hash decoded RGB pixels exactly as the NEO native-vision lane does."""

    return media_content_sha256(
        f"RGB:{_positive_int(width, 'image width')}x"
        f"{_positive_int(height, 'image height')}",
        rgb_bytes,
    )


def audio_content_sha256(
    sample_rate: int,
    channels: int,
    frames: int,
    samples_little_endian: bytes,
) -> str:
    """Hash decoded PCM16 audio under :data:`AUDIO_HASH_BASIS`."""

    rate = _positive_int(sample_rate, "audio sample_rate")
    channel_count = _positive_int(channels, "audio channels")
    frame_count = _nonnegative_int(frames, "audio frames")
    if not isinstance(samples_little_endian, (bytes, bytearray, memoryview)):
        raise DynamicBehaviorGateError("audio samples must be bytes-like.")
    expected = frame_count * channel_count * 2
    if len(bytes(samples_little_endian)) != expected:
        raise DynamicBehaviorGateError(
            "audio sample bytes must equal frames * channels * 2 for PCM16."
        )
    return media_content_sha256(
        f"PCM16:{rate}:{channel_count}:{frame_count}",
        samples_little_endian,
    )


def text_content_sha256(value: str) -> str:
    """Hash text output under :data:`TEXT_HASH_BASIS`."""

    if not isinstance(value, str):
        raise DynamicBehaviorGateError("text content must be a string.")
    return media_content_sha256("TEXT:utf-8", value.encode("utf-8"))


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicBehaviorGateError(f"{name} must be a non-negative integer.")
    return value


def _media_attributes(
    value: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] | None,
    name: str,
) -> tuple[tuple[str, JsonScalar], ...]:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    result: list[tuple[str, JsonScalar]] = []
    for entry in items:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise DynamicBehaviorGateError(f"{name} entries must be key/value pairs.")
        key, item = entry
        key_text = _text(key, f"{name} key")
        if isinstance(item, float) and not math.isfinite(item):
            raise DynamicBehaviorGateError(f"{name}.{key_text} must be finite.")
        if not isinstance(item, (str, int, float, bool, type(None))):
            raise DynamicBehaviorGateError(f"{name}.{key_text} must be a JSON scalar.")
        result.append((key_text, item))
    result.sort(key=lambda pair: pair[0])
    if len({key for key, _ in result}) != len(result):
        raise DynamicBehaviorGateError(f"{name} keys must be unique.")
    return tuple(result)


def _canonical_modality(value: Any, name: str) -> str:
    try:
        return canonical_modality(value, name)
    except DynamicCircuitIRError as exc:
        raise DynamicBehaviorGateError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class MediaReference:
    """One decoded media payload bound by content, not by container bytes."""

    modality: str
    content_sha256: str
    uri: str
    byte_count: int
    codec: str
    hash_basis: str | None = None
    container_sha256: str | None = None
    attributes: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "modality",
            _canonical_modality(self.modality, "media modality"),
        )
        object.__setattr__(
            self,
            "content_sha256",
            _digest(self.content_sha256, "content_sha256"),
        )
        object.__setattr__(self, "uri", _text(self.uri, "media uri"))
        if "\x00" in self.uri:
            raise DynamicBehaviorGateError("media uri must not contain NUL.")
        object.__setattr__(
            self,
            "byte_count",
            _nonnegative_int(self.byte_count, "media byte_count"),
        )
        object.__setattr__(self, "codec", _text(self.codec, "media codec"))
        basis = self.hash_basis
        if basis is None:
            basis = CANONICAL_MEDIA_HASH_BASES.get(self.modality)
            if basis is None:
                raise DynamicBehaviorGateError(
                    f"Modality {self.modality!r} has no canonical decoded-content "
                    "hash basis; supply hash_basis explicitly."
                )
        object.__setattr__(self, "hash_basis", _text(basis, "media hash_basis"))
        if self.container_sha256 is not None:
            object.__setattr__(
                self,
                "container_sha256",
                _digest(self.container_sha256, "container_sha256"),
            )
        object.__setattr__(
            self,
            "attributes",
            _media_attributes(self.attributes, "media attributes"),
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def local_path(self, root: str | Path | None = None) -> Path:
        """Resolve the reference as a local file path under an optional root."""

        if "://" in self.uri:
            raise DynamicBehaviorGateError(
                f"Media reference {self.uri!r} is not a local filesystem path."
            )
        candidate = Path(self.uri)
        if root is None:
            return candidate.expanduser().resolve()
        base = Path(root).expanduser().resolve()
        resolved = (base / candidate).expanduser().resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise DynamicBehaviorGateError(
                "Media reference escapes its declared root directory."
            ) from exc
        return resolved

    def verify_container(self, root: str | Path | None = None) -> Path:
        """Fail closed unless the stored container still matches its seal."""

        if self.container_sha256 is None:
            raise DynamicBehaviorGateError(
                "Media reference declares no container digest to verify."
            )
        path = self.local_path(root)
        if not path.is_file():
            raise DynamicBehaviorGateError(f"Media container is missing: {path}.")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024**2):
                digest.update(block)
        observed = digest.hexdigest()
        size = path.stat().st_size
        if observed != self.container_sha256 or size != self.byte_count:
            raise DynamicBehaviorGateError(
                f"Media container changed after sealing: {path}."
            )
        return path

    def verify_content(self, observed_sha256: str) -> None:
        """Fail closed unless a freshly decoded payload rehashes identically."""

        if _digest(observed_sha256, "observed content digest") != self.content_sha256:
            raise DynamicBehaviorGateError(
                f"Decoded {self.modality} content for {self.uri!r} does not match "
                "its sealed content digest."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MEDIA_REFERENCE_SCHEMA,
            "modality": self.modality,
            "content_sha256": self.content_sha256,
            "uri": self.uri,
            "byte_count": self.byte_count,
            "codec": self.codec,
            "hash_basis": self.hash_basis,
            "container_sha256": self.container_sha256,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MediaReference:
        payload = _mapping(value, "media reference")
        _expect_schema(payload, MEDIA_REFERENCE_SCHEMA, "media reference")
        _expect_keys(
            payload,
            {
                "schema",
                "modality",
                "content_sha256",
                "uri",
                "byte_count",
                "codec",
                "hash_basis",
                "container_sha256",
                "attributes",
            },
            "media reference",
        )
        return cls(
            modality=payload["modality"],
            content_sha256=payload["content_sha256"],
            uri=payload["uri"],
            byte_count=payload["byte_count"],
            codec=payload["codec"],
            hash_basis=payload["hash_basis"],
            container_sha256=payload["container_sha256"],
            attributes=_media_attributes(
                payload["attributes"],
                "media attributes",
            ),
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicBehaviorGateError(f"{name} must be one JSON object.")
    return value


def _expect_schema(payload: Mapping[str, Any], schema: str, name: str) -> None:
    if payload.get("schema") != schema:
        raise DynamicBehaviorGateError(f"{name} schema is unsupported.")


def _expect_keys(
    payload: Mapping[str, Any],
    required: set[str],
    name: str,
) -> None:
    observed = set(payload)
    if observed != required:
        raise DynamicBehaviorGateError(
            f"{name} fields are incomplete or unknown: "
            f"missing={sorted(required - observed)}, "
            f"unknown={sorted(observed - required)}."
        )


def _review_value(value: Any, name: str) -> str | MediaReference:
    if isinstance(value, MediaReference):
        return value
    if isinstance(value, str):
        return value
    raise DynamicBehaviorGateError(f"{name} must be text or a MediaReference.")


def _review_payload(value: str | MediaReference) -> dict[str, Any]:
    if isinstance(value, MediaReference):
        return {"kind": "media", "media": value.as_dict()}
    return {"kind": "text", "text": value}


def _review_from_payload(value: Any, name: str) -> str | MediaReference:
    payload = _mapping(value, name)
    kind = payload.get("kind")
    if kind == "text":
        _expect_keys(payload, {"kind", "text"}, name)
        if not isinstance(payload["text"], str):
            raise DynamicBehaviorGateError(f"{name} text must be a string.")
        return payload["text"]
    if kind == "media":
        _expect_keys(payload, {"kind", "media"}, name)
        return MediaReference.from_dict(payload["media"])
    raise DynamicBehaviorGateError(f"{name} kind must be 'text' or 'media'.")


def _review_empty(value: str | MediaReference) -> bool:
    if isinstance(value, MediaReference):
        return value.byte_count == 0
    return not value.strip()


def _review_modality(value: str | MediaReference) -> str:
    return value.modality if isinstance(value, MediaReference) else MODALITY_TEXT


@dataclass(frozen=True, slots=True)
class MultimodalWholeOutputCase:
    """One reviewed case whose inputs or outputs may be decoded media."""

    case_id: str
    prompt: str
    parent_output: str | MediaReference
    inputs: tuple[MediaReference, ...] = ()
    expected: str | MediaReference | None = None
    category: str = "unspecified"
    parent_fragment_like: bool = False
    parent_eos_margin: float | None = None
    critical_nonempty: bool = False
    parent_metadata: WholeOutputMetadata | None = None
    split: str = "unspecified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        if not isinstance(self.prompt, str):
            raise DynamicBehaviorGateError("prompt must be a string.")
        object.__setattr__(
            self,
            "parent_output",
            _review_value(self.parent_output, "parent_output"),
        )
        inputs = tuple(self.inputs)
        if any(not isinstance(value, MediaReference) for value in inputs):
            raise DynamicBehaviorGateError("inputs must be MediaReference values.")
        if not inputs and not isinstance(self.parent_output, MediaReference):
            raise DynamicBehaviorGateError(
                "A multimodal case must carry at least one media input or a "
                "media parent output; use WholeOutputCase for text-only review."
            )
        object.__setattr__(self, "inputs", inputs)
        if self.expected is not None:
            object.__setattr__(
                self,
                "expected",
                _review_value(self.expected, "expected"),
            )
        object.__setattr__(self, "category", _text(self.category, "category"))
        if not isinstance(self.parent_fragment_like, bool):
            raise DynamicBehaviorGateError("parent_fragment_like must be boolean.")
        if self.parent_eos_margin is not None:
            object.__setattr__(
                self,
                "parent_eos_margin",
                _finite(self.parent_eos_margin, "parent_eos_margin"),
            )
        if not isinstance(self.critical_nonempty, bool):
            raise DynamicBehaviorGateError("critical_nonempty must be boolean.")
        if self.parent_metadata is not None and not isinstance(
            self.parent_metadata,
            WholeOutputMetadata,
        ):
            raise DynamicBehaviorGateError(
                "parent_metadata must be WholeOutputMetadata when supplied."
            )
        object.__setattr__(self, "split", _text(self.split, "split"))

    @property
    def parent_empty(self) -> bool:
        return _review_empty(self.parent_output)

    @property
    def parent_output_is_media(self) -> bool:
        return isinstance(self.parent_output, MediaReference)

    @property
    def input_modalities(self) -> tuple[str, ...]:
        return tuple(value.modality for value in self.inputs)

    @property
    def modalities(self) -> tuple[str, ...]:
        observed = {*self.input_modalities, _review_modality(self.parent_output)}
        if self.expected is not None:
            observed.add(_review_modality(self.expected))
        if self.prompt.strip():
            observed.add(MODALITY_TEXT)
        return tuple(sorted(observed))

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def verify_inputs(self, root: str | Path | None = None) -> None:
        """Verify every input container that declares a container digest."""

        for value in self.inputs:
            if value.container_sha256 is not None:
                value.verify_container(root)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MULTIMODAL_CASE_SCHEMA,
            "case_id": self.case_id,
            "prompt": self.prompt,
            "parent_output": _review_payload(self.parent_output),
            "inputs": [value.as_dict() for value in self.inputs],
            "expected": (
                None if self.expected is None else _review_payload(self.expected)
            ),
            "category": self.category,
            "parent_fragment_like": self.parent_fragment_like,
            "parent_eos_margin": self.parent_eos_margin,
            "critical_nonempty": self.critical_nonempty,
            "parent_metadata": (
                None if self.parent_metadata is None else self.parent_metadata.as_dict()
            ),
            "split": self.split,
            "modalities": list(self.modalities),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MultimodalWholeOutputCase:
        payload = _mapping(value, "multimodal case")
        _expect_schema(payload, MULTIMODAL_CASE_SCHEMA, "multimodal case")
        _expect_keys(
            payload,
            {
                "schema",
                "case_id",
                "prompt",
                "parent_output",
                "inputs",
                "expected",
                "category",
                "parent_fragment_like",
                "parent_eos_margin",
                "critical_nonempty",
                "parent_metadata",
                "split",
                "modalities",
            },
            "multimodal case",
        )
        inputs = payload["inputs"]
        if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
            raise DynamicBehaviorGateError("multimodal case inputs must be a list.")
        result = cls(
            case_id=payload["case_id"],
            prompt=payload["prompt"],
            parent_output=_review_from_payload(
                payload["parent_output"],
                "parent_output",
            ),
            inputs=tuple(MediaReference.from_dict(item) for item in inputs),
            expected=(
                None
                if payload["expected"] is None
                else _review_from_payload(payload["expected"], "expected")
            ),
            category=payload["category"],
            parent_fragment_like=payload["parent_fragment_like"],
            parent_eos_margin=payload["parent_eos_margin"],
            critical_nonempty=payload["critical_nonempty"],
            parent_metadata=_metadata_from_dict(payload["parent_metadata"]),
            split=payload["split"],
        )
        if list(result.modalities) != list(payload["modalities"]):
            raise DynamicBehaviorGateError(
                "multimodal case modality summary is inconsistent."
            )
        return result


def _metadata_from_dict(value: Any) -> WholeOutputMetadata | None:
    if value is None:
        return None
    payload = _mapping(value, "whole-output metadata")
    _expect_keys(
        payload,
        {
            "generated_token_count",
            "generation_limit",
            "generation_limit_saturated",
            "maximum_repeated_ngram_share",
        },
        "whole-output metadata",
    )
    result = WholeOutputMetadata(
        generated_token_count=payload["generated_token_count"],
        generation_limit=payload["generation_limit"],
        maximum_repeated_ngram_share=payload["maximum_repeated_ngram_share"],
    )
    if result.generation_limit_saturated != payload["generation_limit_saturated"]:
        raise DynamicBehaviorGateError(
            "whole-output metadata saturation flag is inconsistent."
        )
    return result


@dataclass(frozen=True, slots=True)
class MultimodalCandidateOutput:
    """One child rollout whose output may itself be decoded media."""

    output: str | MediaReference
    fragment_like: bool = False
    eos_margin: float | None = None
    metadata: WholeOutputMetadata | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", _review_value(self.output, "output"))
        if not isinstance(self.fragment_like, bool):
            raise DynamicBehaviorGateError("fragment_like must be boolean.")
        if self.eos_margin is not None:
            object.__setattr__(
                self,
                "eos_margin",
                _finite(self.eos_margin, "candidate eos_margin"),
            )
        if self.metadata is not None and not isinstance(
            self.metadata,
            WholeOutputMetadata,
        ):
            raise DynamicBehaviorGateError(
                "Candidate metadata must be WholeOutputMetadata when supplied."
            )

    @property
    def empty(self) -> bool:
        return _review_empty(self.output)

    @property
    def is_media(self) -> bool:
        return isinstance(self.output, MediaReference)

    @property
    def modality(self) -> str:
        return _review_modality(self.output)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MULTIMODAL_CANDIDATE_SCHEMA,
            "output": _review_payload(self.output),
            "fragment_like": self.fragment_like,
            "eos_margin": self.eos_margin,
            "metadata": (None if self.metadata is None else self.metadata.as_dict()),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MultimodalCandidateOutput:
        payload = _mapping(value, "multimodal candidate")
        _expect_schema(payload, MULTIMODAL_CANDIDATE_SCHEMA, "multimodal candidate")
        _expect_keys(
            payload,
            {"schema", "output", "fragment_like", "eos_margin", "metadata"},
            "multimodal candidate",
        )
        return cls(
            output=_review_from_payload(payload["output"], "output"),
            fragment_like=payload["fragment_like"],
            eos_margin=payload["eos_margin"],
            metadata=_metadata_from_dict(payload["metadata"]),
        )


def _multimodal_cases(
    cases: Sequence[MultimodalWholeOutputCase],
) -> tuple[MultimodalWholeOutputCase, ...]:
    values = tuple(cases)
    if not values or any(
        not isinstance(value, MultimodalWholeOutputCase) for value in values
    ):
        raise DynamicBehaviorGateError(
            "cases must contain MultimodalWholeOutputCase values."
        )
    if len({value.case_id for value in values}) != len(values):
        raise DynamicBehaviorGateError("Multimodal case IDs repeat.")
    return values


def multimodal_case_baseline_fingerprint(
    cases: Sequence[MultimodalWholeOutputCase],
) -> str:
    """Seal an ordered multimodal case list exactly as the text gate does."""

    values = _multimodal_cases(cases)
    return canonical_sha256(
        {
            "schema": MULTIMODAL_BASELINE_SCHEMA,
            "cases": [value.as_dict() for value in values],
        }
    )


def multimodal_case_coverage(
    cases: Sequence[MultimodalWholeOutputCase],
) -> dict[str, Any]:
    """Summarize which modalities a sealed multimodal case list exercises."""

    values = _multimodal_cases(cases)
    modalities = sorted({name for value in values for name in value.modalities})
    input_modalities = sorted(
        {name for value in values for name in value.input_modalities}
    )
    output_modalities = sorted(
        {_review_modality(value.parent_output) for value in values}
    )
    return {
        "schema": MULTIMODAL_COVERAGE_SCHEMA,
        "case_count": len(values),
        "modalities": modalities,
        "input_modalities": input_modalities,
        "output_modalities": output_modalities,
        "media_input_count": sum(len(value.inputs) for value in values),
        "media_output_count": sum(value.parent_output_is_media for value in values),
        "splits": sorted({value.split for value in values}),
        "categories": sorted({value.category for value in values}),
    }


def build_multimodal_behavior_gate_binding(
    *,
    gate_id: str,
    gate_version: int,
    cases: Sequence[MultimodalWholeOutputCase],
    threshold_config: Mapping[str, Any],
    generator_identity: BehaviorCallableIdentity,
    scorer_identity: BehaviorCallableIdentity | None,
    coverage_policy: Mapping[str, Any],
    baseline_fingerprint: str | None = None,
    baseline: Mapping[str, Any] | None = None,
    components: Sequence[Mapping[str, Any]] = (),
    inference_execution_contract: (BehaviorInferenceExecutionContract | None) = None,
) -> BehaviorGateBinding:
    """Seal a media-capable case list into the canonical gate manifest.

    The manifest body, seal, and reload contract are exactly the text-gate
    ones from :func:`build_behavior_gate_binding`; only the case payloads and
    one extra modality-coverage component differ.
    """

    values = _multimodal_cases(cases)
    coverage = multimodal_case_coverage(values)
    sealed_fingerprint = (
        multimodal_case_baseline_fingerprint(values)
        if baseline_fingerprint is None
        else _digest(baseline_fingerprint, "baseline_fingerprint")
    )
    sealed_baseline = (
        {
            "multimodal_cases_fingerprint": sealed_fingerprint,
            "case_count": coverage["case_count"],
            "modalities": coverage["modalities"],
        }
        if baseline is None
        else dict(baseline)
    )
    return build_behavior_gate_binding(
        gate_id=gate_id,
        gate_version=gate_version,
        baseline_fingerprint=sealed_fingerprint,
        threshold_config=threshold_config,
        cases=[value.as_dict() for value in values],
        baseline=sealed_baseline,
        generator_identity=generator_identity,
        scorer_identity=scorer_identity,
        coverage_policy=coverage_policy,
        components=[*[dict(value) for value in components], coverage],
        inference_execution_contract=inference_execution_contract,
    )


def multimodal_cases_from_manifest(
    manifest: Mapping[str, Any],
) -> tuple[MultimodalWholeOutputCase, ...]:
    """Rebuild sealed multimodal cases from a reloaded gate manifest."""

    payload = _mapping(manifest, "behavior gate manifest")
    cases = payload.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases:
        raise DynamicBehaviorGateError("Behavior gate manifest carries no cases.")
    return tuple(MultimodalWholeOutputCase.from_dict(value) for value in cases)


@dataclass(frozen=True, slots=True)
class TaskScoreCoveragePolicy:
    """Required parent-correct coverage for a task-scoring promotion gate."""

    require_any_scored_case: bool = False
    require_any_parent_correct_case: bool = False
    required_splits: tuple[str, ...] = ()
    required_categories: tuple[str, ...] = ()
    require_split_category_cross_product: bool = False
    parent_correct_minimum_score: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "require_any_scored_case",
            "require_any_parent_correct_case",
            "require_split_category_cross_product",
        ):
            if not isinstance(getattr(self, name), bool):
                raise DynamicBehaviorGateError(f"{name} must be boolean.")
        for name in ("required_splits", "required_categories"):
            values = tuple(_text(value, name) for value in getattr(self, name))
            if len(set(values)) != len(values):
                raise DynamicBehaviorGateError(f"{name} must be unique.")
            object.__setattr__(self, name, values)
        threshold = _finite(
            self.parent_correct_minimum_score,
            "parent_correct_minimum_score",
        )
        if not 0.0 < threshold <= 1.0:
            raise DynamicBehaviorGateError(
                "parent_correct_minimum_score must be in (0, 1]."
            )
        object.__setattr__(self, "parent_correct_minimum_score", threshold)
        if self.require_split_category_cross_product and (
            not self.required_splits or not self.required_categories
        ):
            raise DynamicBehaviorGateError(
                "Cross-product coverage requires splits and categories."
            )

    @classmethod
    def production(
        cls,
        *,
        required_splits: Sequence[str] = ("heldout",),
        required_categories: Sequence[str] = (),
        require_split_category_cross_product: bool = False,
        parent_correct_minimum_score: float = 1.0,
    ) -> TaskScoreCoveragePolicy:
        return cls(
            require_any_scored_case=True,
            require_any_parent_correct_case=True,
            required_splits=tuple(required_splits),
            required_categories=tuple(required_categories),
            require_split_category_cross_product=(require_split_category_cross_product),
            parent_correct_minimum_score=parent_correct_minimum_score,
        )

    @property
    def requires_task_scorer(self) -> bool:
        return bool(
            self.require_any_scored_case
            or self.require_any_parent_correct_case
            or self.required_splits
            or self.required_categories
            or self.require_split_category_cross_product
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "require_any_scored_case": self.require_any_scored_case,
            "require_any_parent_correct_case": (self.require_any_parent_correct_case),
            "required_splits": list(self.required_splits),
            "required_categories": list(self.required_categories),
            "require_split_category_cross_product": (
                self.require_split_category_cross_product
            ),
            "parent_correct_minimum_score": self.parent_correct_minimum_score,
        }


@dataclass(frozen=True, slots=True)
class WholeOutputGateConfig:
    maximum_new_empty_responses: int = 0
    maximum_new_fragment_responses: int = 0
    minimum_parent_nonempty_retention: float = 1.0
    minimum_task_score_retention: float = 1.0
    maximum_parent_task_score_regressions: int = 0
    minimum_eos_margin_delta: float | None = None
    maximum_new_generation_limit_saturations: int | None = None
    maximum_repeated_ngram_share_regressions: int | None = None
    maximum_repeated_ngram_share_delta: float | None = None
    task_score_coverage: TaskScoreCoveragePolicy = field(
        default_factory=TaskScoreCoveragePolicy
    )

    def __post_init__(self) -> None:
        for name in (
            "maximum_new_empty_responses",
            "maximum_new_fragment_responses",
            "maximum_parent_task_score_regressions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DynamicBehaviorGateError(
                    f"{name} must be a non-negative integer."
                )
        for name in (
            "maximum_new_generation_limit_saturations",
            "maximum_repeated_ngram_share_regressions",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DynamicBehaviorGateError(
                    f"{name} must be a non-negative integer or None."
                )
        for name in (
            "minimum_parent_nonempty_retention",
            "minimum_task_score_retention",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise DynamicBehaviorGateError(f"{name} must be between zero and one.")
            object.__setattr__(self, name, value)
        if self.minimum_eos_margin_delta is not None:
            object.__setattr__(
                self,
                "minimum_eos_margin_delta",
                _finite(
                    self.minimum_eos_margin_delta,
                    "minimum_eos_margin_delta",
                ),
            )
        repetition_count_configured = (
            self.maximum_repeated_ngram_share_regressions is not None
        )
        repetition_delta_configured = (
            self.maximum_repeated_ngram_share_delta is not None
        )
        if repetition_count_configured != repetition_delta_configured:
            raise DynamicBehaviorGateError(
                "maximum_repeated_ngram_share_regressions and "
                "maximum_repeated_ngram_share_delta must be configured together."
            )
        if self.maximum_repeated_ngram_share_delta is not None:
            repeated_delta = _finite(
                self.maximum_repeated_ngram_share_delta,
                "maximum_repeated_ngram_share_delta",
            )
            if not 0.0 <= repeated_delta <= 1.0:
                raise DynamicBehaviorGateError(
                    "maximum_repeated_ngram_share_delta must be between zero and one."
                )
            object.__setattr__(
                self,
                "maximum_repeated_ngram_share_delta",
                repeated_delta,
            )
        if not isinstance(self.task_score_coverage, TaskScoreCoveragePolicy):
            raise DynamicBehaviorGateError(
                "task_score_coverage must be TaskScoreCoveragePolicy."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "maximum_new_empty_responses": self.maximum_new_empty_responses,
            "maximum_new_fragment_responses": (self.maximum_new_fragment_responses),
            "minimum_parent_nonempty_retention": (
                self.minimum_parent_nonempty_retention
            ),
            "minimum_task_score_retention": self.minimum_task_score_retention,
            "maximum_parent_task_score_regressions": (
                self.maximum_parent_task_score_regressions
            ),
            "minimum_eos_margin_delta": self.minimum_eos_margin_delta,
            "maximum_new_generation_limit_saturations": (
                self.maximum_new_generation_limit_saturations
            ),
            "maximum_repeated_ngram_share_regressions": (
                self.maximum_repeated_ngram_share_regressions
            ),
            "maximum_repeated_ngram_share_delta": (
                self.maximum_repeated_ngram_share_delta
            ),
            "task_score_coverage": self.task_score_coverage.as_dict(),
        }


Generator = Callable[
    [Any, tuple[WholeOutputCase, ...]],
    Mapping[str, str | CandidateWholeOutput],
]
TaskScorer = Callable[[WholeOutputCase, str], bool | float | None]


class CachedWholeOutputBehaviorGate:
    """Use cached parent responses and current child rollouts as a hard gate."""

    def __init__(
        self,
        cases: Sequence[WholeOutputCase],
        *,
        generator: Generator,
        config: WholeOutputGateConfig | None = None,
        task_scorer: TaskScorer | None = None,
        gate_id: str = "neural_forge.cached_whole_output_behavior",
        gate_version: int = 2,
        generator_identity: BehaviorCallableIdentity | None = None,
        task_scorer_identity: BehaviorCallableIdentity | None = None,
        baseline_binding: Mapping[str, Any] | None = None,
        inference_execution_contract: (
            BehaviorInferenceExecutionContract | None
        ) = None,
    ) -> None:
        values = tuple(cases)
        if not values or any(
            not isinstance(value, WholeOutputCase) for value in values
        ):
            raise DynamicBehaviorGateError("cases must contain WholeOutputCase values.")
        if len({value.case_id for value in values}) != len(values):
            raise DynamicBehaviorGateError("Whole-output case IDs repeat.")
        if not callable(generator):
            raise DynamicBehaviorGateError("generator must be callable.")
        if task_scorer is not None and not callable(task_scorer):
            raise DynamicBehaviorGateError(
                "task_scorer must be callable when supplied."
            )
        self.cases = values
        self.generator = generator
        self.config = config or WholeOutputGateConfig()
        if not isinstance(self.config, WholeOutputGateConfig):
            raise DynamicBehaviorGateError("config must be WholeOutputGateConfig.")
        self.task_scorer = task_scorer
        if task_scorer is None and self.config.task_score_coverage.requires_task_scorer:
            raise DynamicBehaviorGateError(
                "Configured task-score coverage requires a task_scorer."
            )
        if generator_identity is None:
            generator_identity = identify_behavior_callable(generator)
        if not isinstance(generator_identity, BehaviorCallableIdentity):
            raise DynamicBehaviorGateError(
                "generator_identity must be BehaviorCallableIdentity."
            )
        _verify_behavior_callable_identity(
            generator,
            generator_identity,
            name="Generator",
        )
        if task_scorer is None:
            if task_scorer_identity is not None:
                raise DynamicBehaviorGateError(
                    "task_scorer_identity requires a task_scorer."
                )
        elif task_scorer_identity is None:
            task_scorer_identity = identify_behavior_callable(task_scorer)
        if task_scorer_identity is not None and not isinstance(
            task_scorer_identity,
            BehaviorCallableIdentity,
        ):
            raise DynamicBehaviorGateError(
                "task_scorer_identity must be BehaviorCallableIdentity."
            )
        if task_scorer is not None and task_scorer_identity is not None:
            _verify_behavior_callable_identity(
                task_scorer,
                task_scorer_identity,
                name="Task scorer",
            )
        missing_parent_generation_metadata = tuple(
            value.case_id
            for value in values
            if (
                self.config.maximum_new_generation_limit_saturations is not None
                and (
                    value.parent_metadata is None
                    or value.parent_metadata.generation_limit_saturated is None
                )
            )
        )
        if missing_parent_generation_metadata:
            raise DynamicBehaviorGateError(
                "Generation-limit gating requires complete parent metadata for "
                + ",".join(missing_parent_generation_metadata)
                + "."
            )
        missing_parent_repetition_metadata = tuple(
            value.case_id
            for value in values
            if (
                self.config.maximum_repeated_ngram_share_regressions is not None
                and (
                    value.parent_metadata is None
                    or value.parent_metadata.maximum_repeated_ngram_share is None
                )
            )
        )
        if missing_parent_repetition_metadata:
            raise DynamicBehaviorGateError(
                "Repetition gating requires complete parent metadata for "
                + ",".join(missing_parent_repetition_metadata)
                + "."
            )
        self.baseline_fingerprint = canonical_sha256(
            {
                "schema": SCHEMA,
                "cases": [value.as_dict() for value in values],
            }
        )
        normalized_baseline_binding = (
            {}
            if baseline_binding is None
            else _canonical_copy(dict(baseline_binding), "baseline_binding")
        )
        binding = build_behavior_gate_binding(
            gate_id=gate_id,
            gate_version=gate_version,
            baseline_fingerprint=self.baseline_fingerprint,
            threshold_config=self.config.as_dict(),
            cases=[value.as_dict() for value in values],
            baseline={
                "cached_parent_cases_fingerprint": self.baseline_fingerprint,
                "external_binding": normalized_baseline_binding,
            },
            generator_identity=generator_identity,
            scorer_identity=task_scorer_identity,
            coverage_policy=self.config.task_score_coverage.as_dict(),
            inference_execution_contract=inference_execution_contract,
        )
        expose_behavior_gate_binding(self, binding)

    def binding_fields(self) -> dict[str, Any]:
        return resolve_behavior_gate_binding(self).fields()

    @staticmethod
    def _candidate(value: str | CandidateWholeOutput) -> CandidateWholeOutput:
        if isinstance(value, CandidateWholeOutput):
            return value
        if isinstance(value, str):
            return CandidateWholeOutput(value)
        raise DynamicBehaviorGateError(
            "Generator values must be strings or CandidateWholeOutput."
        )

    def evaluate_model(
        self,
        model: Any,
        *,
        verify_model_execution: bool = True,
    ) -> FoldDecision:
        if not isinstance(verify_model_execution, bool):
            raise DynamicBehaviorGateError("verify_model_execution must be boolean.")
        execution_counts = {
            "module_forward_calls": 0,
            "root_forward_calls": 0,
        }
        if verify_model_execution:
            execution_contract = resolve_behavior_inference_execution_contract(self)
            with observe_model_execution(
                model,
                contract=execution_contract,
            ) as observed:
                generated = self.generator(model, self.cases)
            execution_counts = dict(observed)
            if not execution_contract_satisfied(execution_counts):
                raise DynamicBehaviorGateError(
                    "Whole-output generator returned without executing the "
                    "candidate model root required by its declared inference "
                    "entrypoint."
                )
        else:
            generated = self.generator(model, self.cases)
        if not isinstance(generated, Mapping):
            raise DynamicBehaviorGateError(
                "Whole-output generator must return a mapping."
            )
        expected_ids = {value.case_id for value in self.cases}
        observed_ids = {str(value) for value in generated}
        if observed_ids != expected_ids:
            raise DynamicBehaviorGateError(
                "Generated case IDs differ from the cached baseline: "
                f"missing={sorted(expected_ids - observed_ids)}, "
                f"unknown={sorted(observed_ids - expected_ids)}."
            )
        candidates = MappingProxyType(
            {
                case_id: self._candidate(generated[case_id])
                for case_id in sorted(expected_ids)
            }
        )
        parent_nonempty = tuple(value for value in self.cases if not value.parent_empty)
        retained_nonempty = sum(
            not candidates[value.case_id].empty for value in parent_nonempty
        )
        nonempty_retention = (
            1.0 if not parent_nonempty else retained_nonempty / len(parent_nonempty)
        )
        new_empty = tuple(
            value.case_id
            for value in parent_nonempty
            if candidates[value.case_id].empty
        )
        critical_empty = tuple(
            value.case_id
            for value in self.cases
            if value.critical_nonempty and candidates[value.case_id].empty
        )
        new_fragments = tuple(
            value.case_id
            for value in self.cases
            if not value.parent_fragment_like
            and candidates[value.case_id].fragment_like
        )
        eos_regressions = []
        if self.config.minimum_eos_margin_delta is not None:
            for value in self.cases:
                candidate = candidates[value.case_id]
                if value.parent_eos_margin is None or candidate.eos_margin is None:
                    continue
                delta = candidate.eos_margin - value.parent_eos_margin
                if delta < self.config.minimum_eos_margin_delta:
                    eos_regressions.append(value.case_id)

        generation_gate_enabled = (
            self.config.maximum_new_generation_limit_saturations is not None
        )
        parent_generation_limit_saturations = 0
        candidate_generation_limit_saturations = 0
        missing_generation_metadata = []
        incompatible_generation_limits = []
        new_generation_limit_saturations = []
        if generation_gate_enabled:
            for value in self.cases:
                parent_metadata = value.parent_metadata
                if parent_metadata is None:  # guarded during construction
                    raise AssertionError("Parent generation metadata vanished.")
                parent_saturated = parent_metadata.generation_limit_saturated
                if parent_saturated is None:  # guarded during construction
                    raise AssertionError("Parent generation metadata is incomplete.")
                parent_generation_limit_saturations += int(parent_saturated)
                candidate_metadata = candidates[value.case_id].metadata
                if (
                    candidate_metadata is None
                    or candidate_metadata.generation_limit_saturated is None
                ):
                    missing_generation_metadata.append(value.case_id)
                    continue
                if (
                    candidate_metadata.generation_limit
                    != parent_metadata.generation_limit
                ):
                    incompatible_generation_limits.append(value.case_id)
                    continue
                candidate_saturated = candidate_metadata.generation_limit_saturated
                candidate_generation_limit_saturations += int(candidate_saturated)
                if candidate_saturated and not parent_saturated:
                    new_generation_limit_saturations.append(value.case_id)

        repetition_gate_enabled = (
            self.config.maximum_repeated_ngram_share_regressions is not None
        )
        missing_repetition_metadata = []
        repeated_ngram_regressions = []
        maximum_repeated_ngram_share_delta_observed = 0.0
        if repetition_gate_enabled:
            maximum_repeated_ngram_share_delta = (
                self.config.maximum_repeated_ngram_share_delta
            )
            if maximum_repeated_ngram_share_delta is None:
                raise AssertionError("Repetition threshold configuration vanished.")
            for value in self.cases:
                parent_metadata = value.parent_metadata
                if parent_metadata is None:  # guarded during construction
                    raise AssertionError("Parent repetition metadata vanished.")
                parent_share = parent_metadata.maximum_repeated_ngram_share
                if parent_share is None:  # guarded during construction
                    raise AssertionError("Parent repetition metadata is incomplete.")
                candidate_metadata = candidates[value.case_id].metadata
                if (
                    candidate_metadata is None
                    or candidate_metadata.maximum_repeated_ngram_share is None
                ):
                    missing_repetition_metadata.append(value.case_id)
                    continue
                delta = candidate_metadata.maximum_repeated_ngram_share - parent_share
                maximum_repeated_ngram_share_delta_observed = max(
                    maximum_repeated_ngram_share_delta_observed,
                    delta,
                )
                if delta > maximum_repeated_ngram_share_delta + 1e-12:
                    repeated_ngram_regressions.append(value.case_id)

        parent_task_scores: dict[str, float] = {}
        candidate_task_scores: dict[str, float] = {}
        parent_task_score_regressions = []
        missing_candidate_task_scores = []
        if self.task_scorer is not None:
            for value in self.cases:
                parent_score = self.task_scorer(
                    value,
                    value.parent_output,
                )
                candidate_score = self.task_scorer(
                    value,
                    candidates[value.case_id].output,
                )
                if parent_score is None:
                    continue
                normalized_parent_score = _task_score(
                    parent_score,
                    "parent task score",
                )
                parent_task_scores[value.case_id] = normalized_parent_score
                if candidate_score is None:
                    missing_candidate_task_scores.append(value.case_id)
                    continue
                normalized_candidate_score = _task_score(
                    candidate_score,
                    "candidate task score",
                )
                candidate_task_scores[value.case_id] = normalized_candidate_score
                if normalized_candidate_score + 1e-12 < normalized_parent_score:
                    parent_task_score_regressions.append(value.case_id)
        parent_task_total = sum(parent_task_scores.values())
        candidate_task_total = sum(
            candidate_task_scores.get(case_id, 0.0) for case_id in parent_task_scores
        )
        task_retention = (
            1.0
            if parent_task_total <= 0.0
            else candidate_task_total / parent_task_total
        )
        coverage_policy = self.config.task_score_coverage
        parent_correct_task_ids = tuple(
            value.case_id
            for value in self.cases
            if parent_task_scores.get(value.case_id, -1.0) + 1e-12
            >= coverage_policy.parent_correct_minimum_score
        )
        parent_correct_task_set = set(parent_correct_task_ids)
        parent_correct_by_split = {
            split: sum(
                value.case_id in parent_correct_task_set and value.split == split
                for value in self.cases
            )
            for split in sorted({value.split for value in self.cases})
        }
        parent_correct_by_category = {
            category: sum(
                value.case_id in parent_correct_task_set and value.category == category
                for value in self.cases
            )
            for category in sorted({value.category for value in self.cases})
        }
        missing_required_task_splits = tuple(
            split
            for split in coverage_policy.required_splits
            if parent_correct_by_split.get(split, 0) <= 0
        )
        missing_required_task_categories = tuple(
            category
            for category in coverage_policy.required_categories
            if parent_correct_by_category.get(category, 0) <= 0
        )
        missing_required_task_split_categories = ()
        if coverage_policy.require_split_category_cross_product:
            missing_required_task_split_categories = tuple(
                f"{split}:{category}"
                for split in coverage_policy.required_splits
                for category in coverage_policy.required_categories
                if not any(
                    value.case_id in parent_correct_task_set
                    and value.split == split
                    and value.category == category
                    for value in self.cases
                )
            )

        reasons = []
        if len(new_empty) > self.config.maximum_new_empty_responses:
            reasons.append("new_empty=" + ",".join(new_empty))
        if critical_empty:
            reasons.append("critical_empty=" + ",".join(critical_empty))
        if len(new_fragments) > self.config.maximum_new_fragment_responses:
            reasons.append("new_fragments=" + ",".join(new_fragments))
        if nonempty_retention < self.config.minimum_parent_nonempty_retention:
            reasons.append("parent_nonempty_retention_below_threshold")
        if eos_regressions:
            reasons.append("eos_margin_regression=" + ",".join(eos_regressions))
        if missing_generation_metadata:
            reasons.append(
                "missing_generation_limit_metadata="
                + ",".join(missing_generation_metadata)
            )
        if incompatible_generation_limits:
            reasons.append(
                "generation_limit_mismatch=" + ",".join(incompatible_generation_limits)
            )
        if (
            generation_gate_enabled
            and len(new_generation_limit_saturations)
            > self.config.maximum_new_generation_limit_saturations
        ):
            reasons.append(
                "new_generation_limit_saturation="
                + ",".join(new_generation_limit_saturations)
            )
        if missing_repetition_metadata:
            reasons.append(
                "missing_repeated_ngram_metadata="
                + ",".join(missing_repetition_metadata)
            )
        if (
            repetition_gate_enabled
            and len(repeated_ngram_regressions)
            > self.config.maximum_repeated_ngram_share_regressions
        ):
            reasons.append(
                "repeated_ngram_regression=" + ",".join(repeated_ngram_regressions)
            )
        if self.task_scorer is not None and not parent_task_scores:
            reasons.append("task_score_parent_coverage_zero")
        if (
            coverage_policy.require_any_scored_case
            and not parent_task_scores
            and "task_score_parent_coverage_zero" not in reasons
        ):
            reasons.append("task_score_parent_coverage_zero")
        if (
            coverage_policy.require_any_parent_correct_case
            and not parent_correct_task_ids
        ):
            reasons.append("task_score_parent_correct_coverage_zero")
        if missing_required_task_splits:
            reasons.append(
                "task_score_required_split_parent_correct_coverage_zero="
                + ",".join(missing_required_task_splits)
            )
        if missing_required_task_categories:
            reasons.append(
                "task_score_required_category_parent_correct_coverage_zero="
                + ",".join(missing_required_task_categories)
            )
        if missing_required_task_split_categories:
            reasons.append(
                "task_score_required_split_category_parent_correct_coverage_zero="
                + ",".join(missing_required_task_split_categories)
            )
        if missing_candidate_task_scores:
            reasons.append(
                "missing_candidate_task_score="
                + ",".join(missing_candidate_task_scores)
            )
        if task_retention < self.config.minimum_task_score_retention:
            reasons.append("task_score_retention_below_threshold")
        if (
            len(parent_task_score_regressions)
            > self.config.maximum_parent_task_score_regressions
        ):
            reasons.append(
                "parent_task_score_regression="
                + ",".join(parent_task_score_regressions)
            )

        metrics = {
            "case_count": len(self.cases),
            "parent_nonempty_count": len(parent_nonempty),
            "retained_parent_nonempty_count": retained_nonempty,
            "parent_nonempty_retention": nonempty_retention,
            "new_empty_count": len(new_empty),
            "critical_empty_count": len(critical_empty),
            "new_fragment_count": len(new_fragments),
            "eos_margin_regression_count": len(eos_regressions),
            "generation_limit_gate_enabled": generation_gate_enabled,
            "configured_maximum_new_generation_limit_saturations": (
                self.config.maximum_new_generation_limit_saturations
            ),
            "parent_generation_limit_saturation_count": (
                parent_generation_limit_saturations
            ),
            "candidate_generation_limit_saturation_count": (
                candidate_generation_limit_saturations
            ),
            "missing_generation_limit_metadata_count": len(missing_generation_metadata),
            "generation_limit_mismatch_count": len(incompatible_generation_limits),
            "new_generation_limit_saturation_count": len(
                new_generation_limit_saturations
            ),
            "repetition_gate_enabled": repetition_gate_enabled,
            "configured_maximum_repeated_ngram_share_regressions": (
                self.config.maximum_repeated_ngram_share_regressions
            ),
            "configured_maximum_repeated_ngram_share_delta": (
                self.config.maximum_repeated_ngram_share_delta
            ),
            "missing_repeated_ngram_metadata_count": len(missing_repetition_metadata),
            "repeated_ngram_regression_count": len(repeated_ngram_regressions),
            "maximum_repeated_ngram_share_delta_observed": (
                maximum_repeated_ngram_share_delta_observed
            ),
            "task_scored_case_count": len(parent_task_scores),
            "task_candidate_scored_case_count": len(candidate_task_scores),
            "task_missing_candidate_score_count": len(missing_candidate_task_scores),
            "task_parent_correct_scored_case_count": len(parent_correct_task_ids),
            "task_parent_correct_by_split_json": json.dumps(
                parent_correct_by_split,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "task_parent_correct_by_category_json": json.dumps(
                parent_correct_by_category,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "task_missing_required_split_count": len(missing_required_task_splits),
            "task_missing_required_category_count": len(
                missing_required_task_categories
            ),
            "task_missing_required_split_category_count": len(
                missing_required_task_split_categories
            ),
            "task_score_coverage_policy_fingerprint": canonical_sha256(
                coverage_policy.as_dict()
            ),
            "task_score_retention": task_retention,
            "configured_maximum_parent_task_score_regressions": (
                self.config.maximum_parent_task_score_regressions
            ),
            "parent_task_score_regression_count": len(parent_task_score_regressions),
            "exact_text_equality_is_diagnostic_only": True,
            "parent_forward_passes": 0,
            "model_execution_verified": verify_model_execution,
            "observed_child_module_forward_calls": (
                execution_counts["module_forward_calls"]
            ),
            "observed_child_root_forward_calls": (
                execution_counts["root_forward_calls"]
            ),
        }
        if reasons:
            return FoldDecision.reject(
                "Whole-output regression gate failed: " + "; ".join(reasons),
                metrics,
            )
        return FoldDecision.accept(
            "Whole-output regression gate accepted the child.",
            metrics,
        )

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        return self.evaluate_model(context.execution_context.child_model)

    def __call__(self, model: Any) -> FoldDecision:
        return self.evaluate_model(model)


__all__ = [
    "AUDIO_HASH_BASIS",
    "BehaviorCallableIdentity",
    "BehaviorGateBinding",
    "BehaviorInferenceExecutionContract",
    "CANONICAL_MEDIA_HASH_BASES",
    "CachedWholeOutputBehaviorGate",
    "CandidateWholeOutput",
    "DynamicBehaviorGateError",
    "IMAGE_HASH_BASIS",
    "MANIFEST_SCHEMA",
    "MEDIA_REFERENCE_SCHEMA",
    "MULTIMODAL_BASELINE_SCHEMA",
    "MULTIMODAL_CANDIDATE_SCHEMA",
    "MULTIMODAL_CASE_SCHEMA",
    "MULTIMODAL_COVERAGE_SCHEMA",
    "MediaReference",
    "MultimodalCandidateOutput",
    "MultimodalWholeOutputCase",
    "SCHEMA",
    "TEXT_HASH_BASIS",
    "TaskScoreCoveragePolicy",
    "WholeOutputCase",
    "WholeOutputGateConfig",
    "WholeOutputMetadata",
    "audio_content_sha256",
    "build_behavior_gate_binding",
    "build_multimodal_behavior_gate_binding",
    "expose_behavior_gate_binding",
    "execution_contract_satisfied",
    "identify_behavior_callable",
    "image_content_sha256",
    "inference_execution_contract_from_manifest",
    "media_content_sha256",
    "multimodal_case_baseline_fingerprint",
    "multimodal_case_coverage",
    "multimodal_cases_from_manifest",
    "observe_model_execution",
    "resolve_behavior_gate_binding",
    "resolve_behavior_inference_execution_contract",
    "text_content_sha256",
]
