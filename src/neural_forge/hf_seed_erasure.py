"""Fail-closed audit for seed erasure across independent HF-style compiles.

The audit is model-family neutral: it inspects a PyTorch module's complete
named parameter/buffer inventory and delegates behavioral evaluation to the
caller.  It does not compile a model and it cannot turn a finite set of runs
into a universal claim about untested seeds.  A successful report proves only
that the supplied, independently seeded compile outcomes are identical across
the evidence surfaces checked here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


_SCHEMA = "neural-forge-seed-erasure-audit-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")
_REPORT_EXCEPTION_SENTINEL = {"__neural_forge_seed_owned__": True}


class SeedErasureError(ValueError):
    """Base class for a seed-erasure audit failure."""


class SeedErasureContractError(SeedErasureError):
    """Raised when supplied outcomes do not satisfy the audit contract."""


class SeedLeakError(SeedErasureError):
    """Raised when an undeclared seed-dependent difference is found."""


class DishonestSeedExceptionError(SeedErasureError):
    """Raised when an exception is broad, missing, or does not differ."""


@dataclass(frozen=True, slots=True)
class SeedOwnedField:
    """One exact, intentionally seed-owned field and its justification."""

    name: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SeedOwnedExceptionScope:
    """Exact exceptions that may differ without hiding adjacent evidence.

    Tensor and output names are exact logical names.  Compiler-report names
    are RFC 6901 JSON pointers to scalar leaves.  Containers and the report
    root cannot be excepted.
    """

    tensors: tuple[SeedOwnedField, ...] = ()
    outputs: tuple[SeedOwnedField, ...] = ()
    compiler_report_paths: tuple[SeedOwnedField, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "tensors": [field.as_dict() for field in self.tensors],
            "outputs": [field.as_dict() for field in self.outputs],
            "compiler_report_paths": [
                field.as_dict() for field in self.compiler_report_paths
            ],
        }


@dataclass(frozen=True, slots=True)
class SeedCompileOutcome:
    """One completed compile outcome supplied to :func:`audit_seed_erasure`."""

    seed: int
    model: Any
    parameter_budget: int
    compiler_report: Any
    label: str | None = None
    structure_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class NamedTensorDigest:
    """Immutable content and structure evidence for one logical tensor."""

    name: str
    kind: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: str
    requires_grad: bool
    persistent: bool | None
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["shape"] = list(self.shape)
        value["stride"] = list(self.stride)
        return value


@dataclass(frozen=True, slots=True)
class NamedOutputFingerprint:
    """One caller-computed behavioral fingerprint."""

    name: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SeedOutcomeAudit:
    """Immutable evidence captured for one compile outcome."""

    label: str
    seed: int
    declared_parameter_budget: int
    unique_parameter_count: int
    structure_sha256: str
    tensor_inventory_sha256: str
    tensor_manifest_sha256: str
    compiler_report_sha256: str
    output_manifest_sha256: str
    named_tensors: tuple[NamedTensorDigest, ...]
    output_fingerprints: tuple[NamedOutputFingerprint, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "seed": self.seed,
            "declared_parameter_budget": self.declared_parameter_budget,
            "unique_parameter_count": self.unique_parameter_count,
            "structure_sha256": self.structure_sha256,
            "tensor_inventory_sha256": self.tensor_inventory_sha256,
            "tensor_manifest_sha256": self.tensor_manifest_sha256,
            "compiler_report_sha256": self.compiler_report_sha256,
            "output_manifest_sha256": self.output_manifest_sha256,
            "named_tensors": [tensor.as_dict() for tensor in self.named_tensors],
            "output_fingerprints": [
                fingerprint.as_dict() for fingerprint in self.output_fingerprints
            ],
        }


@dataclass(frozen=True, slots=True)
class SeedErasureReport:
    """Immutable truth record for the audited finite set of compile outcomes."""

    schema: str
    outcome_count: int
    seeds: tuple[int, ...]
    outcomes: tuple[SeedOutcomeAudit, ...]
    declared_exceptions: SeedOwnedExceptionScope
    common_structure_sha256: str
    exact_structure_match: bool
    exact_parameter_budget_match: bool
    named_tensor_inventory_match: bool
    compiler_reports_equivalent_after_declared_exceptions: bool
    output_inventory_match: bool
    differing_tensor_names: tuple[str, ...]
    differing_output_names: tuple[str, ...]
    differing_compiler_report_paths: tuple[str, ...]
    named_tensors_seed_invariant: bool
    behavioral_outputs_seed_invariant: bool
    initial_randomness_remains_in_named_tensors: bool
    initial_randomness_observable_in_outputs: bool
    seed_leak_detected: bool
    seed_invariant: bool
    seed_erasure_proven: bool
    proof_scope: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "outcome_count": self.outcome_count,
            "seeds": list(self.seeds),
            "outcomes": [outcome.as_dict() for outcome in self.outcomes],
            "declared_exceptions": self.declared_exceptions.as_dict(),
            "common_structure_sha256": self.common_structure_sha256,
            "exact_structure_match": self.exact_structure_match,
            "exact_parameter_budget_match": self.exact_parameter_budget_match,
            "named_tensor_inventory_match": self.named_tensor_inventory_match,
            "compiler_reports_equivalent_after_declared_exceptions": (
                self.compiler_reports_equivalent_after_declared_exceptions
            ),
            "output_inventory_match": self.output_inventory_match,
            "differing_tensor_names": list(self.differing_tensor_names),
            "differing_output_names": list(self.differing_output_names),
            "differing_compiler_report_paths": list(
                self.differing_compiler_report_paths
            ),
            "named_tensors_seed_invariant": self.named_tensors_seed_invariant,
            "behavioral_outputs_seed_invariant": (
                self.behavioral_outputs_seed_invariant
            ),
            "initial_randomness_remains_in_named_tensors": (
                self.initial_randomness_remains_in_named_tensors
            ),
            "initial_randomness_observable_in_outputs": (
                self.initial_randomness_observable_in_outputs
            ),
            "seed_leak_detected": self.seed_leak_detected,
            "seed_invariant": self.seed_invariant,
            "seed_erasure_proven": self.seed_erasure_proven,
            "proof_scope": self.proof_scope,
        }


OutputFingerprintCallback = Callable[[SeedCompileOutcome], Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class _ModelCapture:
    structure_sha256: str
    tensor_inventory_sha256: str
    tensor_manifest_sha256: str
    unique_parameter_count: int
    named_tensors: tuple[NamedTensorDigest, ...]


@dataclass(frozen=True, slots=True)
class _InitialEvidence:
    label: str
    outcome: SeedCompileOutcome
    model: _ModelCapture
    compiler_report: Any
    compiler_report_sha256: str


def _canonical_json(value: Any, *, context: str) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SeedErasureContractError(f"{context} must be strict JSON data.") from exc


def _json_value(value: Any, *, context: str) -> Any:
    """Return a detached JSON-only value with deterministic key handling."""

    serialized = _canonical_json(value, context=context)
    return json.loads(serialized.decode("utf-8"))


def _normalize_model_config_scalars(value: Any) -> Any:
    """Canonicalize standard framework scalars emitted by model configs.

    Recent Transformers configs may retain ``torch.dtype`` objects in nested
    runtime/kernel metadata even though the public config is otherwise JSON
    shaped.  A dtype has one unambiguous textual identity, so normalize only
    that standard scalar and continue to reject every other non-JSON object.
    """

    torch = _torch()
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Mapping):
        return {
            key: _normalize_model_config_scalars(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_model_config_scalars(item) for item in value]
    return value


def _json_sha256(value: Any, *, context: str) -> str:
    return hashlib.sha256(_canonical_json(value, context=context)).hexdigest()


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SeedErasureContractError(
            "audit_seed_erasure requires the neural-forge 'hf' or 'native' extra."
        ) from exc
    return torch


def _named_with_duplicates(model: Any, method_name: str) -> list[tuple[str, Any]]:
    method = getattr(model, method_name, None)
    if not callable(method):
        raise SeedErasureContractError(
            f"model must provide PyTorch Module.{method_name}()."
        )
    try:
        values = list(method(recurse=True, remove_duplicate=False))
    except TypeError as exc:
        raise SeedErasureContractError(
            f"model.{method_name}() must support remove_duplicate=False; "
            "PyTorch 2.5 or newer is required."
        ) from exc
    for item in values:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
        ):
            raise SeedErasureContractError(
                f"model.{method_name}() returned an invalid named tensor."
            )
    return values


def _named_modules_with_duplicates(model: Any) -> list[tuple[str, Any]]:
    method = getattr(model, "named_modules", None)
    if not callable(method):
        raise SeedErasureContractError(
            "model must provide PyTorch Module.named_modules()."
        )
    try:
        values = list(method(remove_duplicate=False))
    except TypeError as exc:
        raise SeedErasureContractError(
            "model.named_modules() must support remove_duplicate=False; "
            "PyTorch 2.5 or newer is required."
        ) from exc
    return values


def _tensor_sha256(torch: Any, tensor: Any, *, name: str) -> str:
    if not isinstance(tensor, torch.Tensor):
        raise SeedErasureContractError(f"Named tensor {name!r} is not a tensor.")
    if tensor.device.type == "meta":
        raise SeedErasureContractError(
            f"Named tensor {name!r} is unmaterialized on the meta device."
        )
    if tensor.layout != torch.strided:
        raise SeedErasureContractError(
            f"Named tensor {name!r} is not a dense strided tensor."
        )

    header = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }
    digest = hashlib.sha256(_canonical_json(header, context=f"tensor {name!r}"))
    try:
        value = tensor.detach().resolve_conj().resolve_neg().cpu().contiguous()
        flat = value.reshape(-1)
        elements_per_chunk = max(1, (8 * 1024 * 1024) // value.element_size())
        for start in range(0, flat.numel(), elements_per_chunk):
            chunk = flat[start : start + elements_per_chunk].contiguous()
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise SeedErasureContractError(
            f"Named tensor {name!r} cannot be hashed exactly."
        ) from exc
    return digest.hexdigest()


def _buffer_persistence(model: Any) -> dict[str, bool]:
    persistence: dict[str, bool] = {}
    for module_name, module in _named_modules_with_duplicates(model):
        buffers = getattr(module, "_buffers", {})
        non_persistent = getattr(module, "_non_persistent_buffers_set", set())
        for local_name, tensor in buffers.items():
            if tensor is None:
                continue
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            value = local_name not in non_persistent
            previous = persistence.get(full_name)
            if previous is not None and previous != value:
                raise SeedErasureContractError(
                    f"Buffer persistence is ambiguous for {full_name!r}."
                )
            persistence[full_name] = value
    return persistence


def _storage_alias_groups(named_values: Sequence[tuple[str, Any]]) -> list[list[str]]:
    groups: dict[tuple[Any, ...], list[str]] = {}
    for name, tensor in named_values:
        try:
            storage = tensor.untyped_storage()
            byte_count = int(storage.nbytes())
            if byte_count:
                key = (
                    tensor.device.type,
                    tensor.device.index,
                    int(storage.data_ptr()),
                    byte_count,
                )
            else:
                key = ("empty-object", id(tensor))
        except (AttributeError, RuntimeError) as exc:
            raise SeedErasureContractError(
                f"Cannot inspect storage ownership for named tensor {name!r}."
            ) from exc
        groups.setdefault(key, []).append(name)
    return sorted(sorted(names) for names in groups.values() if len(names) > 1)


def _model_capture(outcome: SeedCompileOutcome) -> _ModelCapture:
    torch = _torch()
    model = outcome.model
    parameters = _named_with_duplicates(model, "named_parameters")
    buffers = _named_with_duplicates(model, "named_buffers")
    persistence = _buffer_persistence(model)

    all_values: list[tuple[str, Any]] = []
    kinds: dict[str, str] = {}
    for kind, values in (("parameter", parameters), ("buffer", buffers)):
        for name, tensor in values:
            if name in kinds:
                raise SeedErasureContractError(
                    f"Duplicate logical tensor name {name!r} in {kind} inventory."
                )
            kinds[name] = kind
            all_values.append((name, tensor))

    records: list[NamedTensorDigest] = []
    for name, tensor in sorted(all_values):
        kind = kinds[name]
        records.append(
            NamedTensorDigest(
                name=name,
                kind=kind,
                shape=tuple(int(size) for size in tensor.shape),
                stride=tuple(int(size) for size in tensor.stride()),
                storage_offset=int(tensor.storage_offset()),
                dtype=str(tensor.dtype),
                requires_grad=bool(tensor.requires_grad),
                persistent=persistence.get(name) if kind == "buffer" else None,
                sha256=_tensor_sha256(torch, tensor, name=name),
            )
        )

    unique_parameters: dict[tuple[Any, ...], Any] = {}
    for _, parameter in parameters:
        key = (
            parameter.device.type,
            parameter.device.index,
            str(parameter.dtype),
            int(parameter.untyped_storage().data_ptr()),
            int(parameter.storage_offset()),
            tuple(int(size) for size in parameter.shape),
            tuple(int(size) for size in parameter.stride()),
        )
        unique_parameters[key] = parameter
    unique_parameter_count = sum(
        int(parameter.numel()) for parameter in unique_parameters.values()
    )

    module_manifest = []
    for name, module in _named_modules_with_duplicates(model):
        module_manifest.append(
            {
                "name": name,
                "type": _qualified_type(module),
                "training": bool(getattr(module, "training", False)),
            }
        )

    config_value: Any = None
    config = getattr(model, "config", None)
    if config is not None:
        if isinstance(config, Mapping):
            raw_config = config
        elif is_dataclass(config) and not isinstance(config, type):
            raw_config = asdict(config)
        else:
            to_dict = getattr(config, "to_dict", None)
            if callable(to_dict):
                raw_config = to_dict()
            else:
                raw_config = None
        if raw_config is None:
            raise SeedErasureContractError(
                "model.config must be a mapping, dataclass, or provide to_dict() "
                "when present."
            )
        config_value = _json_value(
            _normalize_model_config_scalars(raw_config),
            context="model config",
        )

    metadata_value = None
    if outcome.structure_metadata is not None:
        metadata_value = _json_value(
            outcome.structure_metadata,
            context="outcome.structure_metadata",
        )

    inventory = [
        {
            "name": record.name,
            "kind": record.kind,
            "shape": list(record.shape),
            "stride": list(record.stride),
            "storage_offset": record.storage_offset,
            "dtype": record.dtype,
            "requires_grad": record.requires_grad,
            "persistent": record.persistent,
        }
        for record in records
    ]
    aliases = _storage_alias_groups(all_values)
    structure = {
        "modules": module_manifest,
        "tensor_inventory": inventory,
        "storage_alias_groups": aliases,
        "config": config_value,
        "caller_structure_metadata": metadata_value,
    }
    tensor_manifest = [record.as_dict() for record in records]
    return _ModelCapture(
        structure_sha256=_json_sha256(structure, context="model structure"),
        tensor_inventory_sha256=_json_sha256(
            {"inventory": inventory, "storage_alias_groups": aliases},
            context="tensor inventory",
        ),
        tensor_manifest_sha256=_json_sha256(
            tensor_manifest,
            context="tensor manifest",
        ),
        unique_parameter_count=unique_parameter_count,
        named_tensors=tuple(records),
    )


def _normalize_compiler_report(report: Any) -> Any:
    if is_dataclass(report) and not isinstance(report, type):
        value = asdict(report)
    elif isinstance(report, Mapping):
        value = report
    else:
        as_dict_method = getattr(report, "as_dict", None)
        report_method = getattr(report, "report", None)
        if callable(as_dict_method):
            value = as_dict_method()
        elif callable(report_method):
            value = report_method()
        else:
            raise SeedErasureContractError(
                "compiler_report must be a mapping, dataclass, or expose "
                "as_dict()/report()."
            )
    if not isinstance(value, Mapping):
        raise SeedErasureContractError("compiler_report must normalize to a mapping.")
    return _json_value(value, context="compiler_report")


def _json_topology(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_topology(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_topology(child) for child in value]
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    raise AssertionError("JSON normalization left a non-JSON value")


def _escape_json_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _pointer_tokens(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise DishonestSeedExceptionError(
            f"Compiler-report exception {pointer!r} must be a non-root JSON pointer."
        )
    tokens = pointer[1:].split("/")
    result = []
    for token in tokens:
        index = 0
        while index < len(token):
            if token[index] == "~":
                if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
                    raise DishonestSeedExceptionError(
                        f"Compiler-report exception {pointer!r} has invalid escaping."
                    )
                index += 2
            else:
                index += 1
        result.append(token.replace("~1", "/").replace("~0", "~"))
    return result


def _resolve_pointer(value: Any, pointer: str) -> Any:
    current = value
    for token in _pointer_tokens(pointer):
        if isinstance(current, dict):
            if token not in current:
                raise DishonestSeedExceptionError(
                    f"Compiler-report exception {pointer!r} does not exist in every report."
                )
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise DishonestSeedExceptionError(
                    f"Compiler-report exception {pointer!r} has an invalid list index."
                )
            index = int(token)
            if index >= len(current):
                raise DishonestSeedExceptionError(
                    f"Compiler-report exception {pointer!r} does not exist in every report."
                )
            current = current[index]
        else:
            raise DishonestSeedExceptionError(
                f"Compiler-report exception {pointer!r} crosses a scalar value."
            )
    if isinstance(current, (dict, list)):
        raise DishonestSeedExceptionError(
            f"Compiler-report exception {pointer!r} is a container; only leaves are allowed."
        )
    return current


def _replace_pointer(value: Any, pointer: str) -> None:
    tokens = _pointer_tokens(pointer)
    parent = value
    for token in tokens[:-1]:
        parent = parent[int(token)] if isinstance(parent, list) else parent[token]
    final = tokens[-1]
    replacement = copy.deepcopy(_REPORT_EXCEPTION_SENTINEL)
    if isinstance(parent, list):
        parent[int(final)] = replacement
    else:
        parent[final] = replacement


def _json_differences(left: Any, right: Any, pointer: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(left):
            child_pointer = f"{pointer}/{_escape_json_pointer(key)}"
            differences.extend(_json_differences(left[key], right[key], child_pointer))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        for index, (left_child, right_child) in enumerate(zip(left, right)):
            differences.extend(
                _json_differences(left_child, right_child, f"{pointer}/{index}")
            )
        return differences
    return [] if left == right else [pointer or "/"]


def _normalize_field_group(
    values: Sequence[SeedOwnedField],
    *,
    group_name: str,
) -> tuple[SeedOwnedField, ...]:
    normalized = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, SeedOwnedField):
            raise DishonestSeedExceptionError(
                f"{group_name} exceptions must contain SeedOwnedField values."
            )
        if not isinstance(value.name, str) or not value.name.strip():
            raise DishonestSeedExceptionError(
                f"Every {group_name} exception must have an exact nonempty name."
            )
        if not isinstance(value.reason, str) or not value.reason.strip():
            raise DishonestSeedExceptionError(
                f"Exception {value.name!r} must have a nonempty reason."
            )
        name = value.name.strip()
        if name in seen:
            raise DishonestSeedExceptionError(
                f"Duplicate {group_name} exception {name!r}."
            )
        seen.add(name)
        normalized.append(SeedOwnedField(name=name, reason=value.reason.strip()))
    return tuple(normalized)


def _normalize_scope(
    exceptions: SeedOwnedExceptionScope | None,
) -> SeedOwnedExceptionScope:
    if exceptions is None:
        return SeedOwnedExceptionScope()
    if not isinstance(exceptions, SeedOwnedExceptionScope):
        raise DishonestSeedExceptionError(
            "exceptions must be a SeedOwnedExceptionScope."
        )
    return SeedOwnedExceptionScope(
        tensors=_normalize_field_group(exceptions.tensors, group_name="tensor"),
        outputs=_normalize_field_group(exceptions.outputs, group_name="output"),
        compiler_report_paths=_normalize_field_group(
            exceptions.compiler_report_paths,
            group_name="compiler-report",
        ),
    )


def _normalize_output_fingerprints(
    value: Any,
    *,
    label: str,
) -> tuple[NamedOutputFingerprint, ...]:
    if not isinstance(value, Mapping) or not value:
        raise SeedErasureContractError(
            f"Output fingerprint callback for {label!r} must return a nonempty mapping."
        )
    result = []
    for name, digest in sorted(value.items()):
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise SeedErasureContractError(
                f"Output fingerprint names for {label!r} must be exact nonempty strings."
            )
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise SeedErasureContractError(
                f"Output fingerprint {name!r} for {label!r} must be a SHA-256 hex digest."
            )
        result.append(NamedOutputFingerprint(name=name, sha256=digest.lower()))
    return tuple(result)


def _first_mismatch(values: Sequence[str]) -> int | None:
    first = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value != first:
            return index
    return None


def _validate_outcomes(outcomes: Sequence[SeedCompileOutcome]) -> tuple[str, ...]:
    if isinstance(outcomes, (str, bytes)) or len(outcomes) < 2:
        raise SeedErasureContractError(
            "At least two independently seeded compile outcomes are required."
        )
    seeds: set[int] = set()
    model_ids: set[int] = set()
    labels: list[str] = []
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, SeedCompileOutcome):
            raise SeedErasureContractError(
                f"outcomes[{index}] must be a SeedCompileOutcome."
            )
        if isinstance(outcome.seed, bool) or not isinstance(outcome.seed, int):
            raise SeedErasureContractError(f"outcomes[{index}].seed must be an integer.")
        if outcome.seed in seeds:
            raise SeedErasureContractError("Compile outcome seeds must be unique.")
        seeds.add(outcome.seed)
        if outcome.model is None or id(outcome.model) in model_ids:
            raise SeedErasureContractError(
                "Each seed must provide a distinct compiled model object."
            )
        model_ids.add(id(outcome.model))
        if (
            isinstance(outcome.parameter_budget, bool)
            or not isinstance(outcome.parameter_budget, int)
            or outcome.parameter_budget <= 0
        ):
            raise SeedErasureContractError(
                f"outcomes[{index}].parameter_budget must be a positive integer."
            )
        label = outcome.label if outcome.label is not None else f"seed-{outcome.seed}"
        if not isinstance(label, str) or not label.strip() or label != label.strip():
            raise SeedErasureContractError(
                f"outcomes[{index}].label must be an exact nonempty string."
            )
        if label in labels:
            raise SeedErasureContractError("Compile outcome labels must be unique.")
        labels.append(label)
    return tuple(labels)


def audit_seed_erasure(
    outcomes: Sequence[SeedCompileOutcome],
    output_fingerprints: OutputFingerprintCallback,
    *,
    exceptions: SeedOwnedExceptionScope | None = None,
) -> SeedErasureReport:
    """Audit whether initialization seed information survived compilation.

    The callback is invoked once per outcome and must return a nonempty mapping
    of exact behavioral-suite names to SHA-256 digests.  It should run the same
    deterministic probes for every model and hash their raw result payloads.
    Model or report mutation during the callback is rejected.

    Declared tensor/output differences are reported as seed leaks rather than
    treated as proof of erasure.  Exceptions only make intentional ownership
    explicit; they never turn different compiled models into identical ones.
    """

    if not callable(output_fingerprints):
        raise SeedErasureContractError("output_fingerprints must be callable.")
    if isinstance(outcomes, (str, bytes)):
        raise SeedErasureContractError(
            "At least two independently seeded compile outcomes are required."
        )
    try:
        normalized_outcomes = tuple(outcomes)
    except TypeError as exc:
        raise SeedErasureContractError("outcomes must be an iterable.") from exc
    labels = _validate_outcomes(normalized_outcomes)
    scope = _normalize_scope(exceptions)

    initial: list[_InitialEvidence] = []
    for label, outcome in zip(labels, normalized_outcomes):
        model_capture = _model_capture(outcome)
        report = _normalize_compiler_report(outcome.compiler_report)
        initial.append(
            _InitialEvidence(
                label=label,
                outcome=outcome,
                model=model_capture,
                compiler_report=report,
                compiler_report_sha256=_json_sha256(
                    report,
                    context="compiler_report",
                ),
            )
        )

    structure_hashes = [evidence.model.structure_sha256 for evidence in initial]
    mismatch = _first_mismatch(structure_hashes)
    if mismatch is not None:
        raise SeedErasureContractError(
            "Exact model structure differs between "
            f"{initial[0].label!r} and {initial[mismatch].label!r}; "
            "missing/renamed tensors, shapes, dtypes, aliases, modules, config, "
            "and structure metadata cannot be excepted."
        )

    budgets = [evidence.outcome.parameter_budget for evidence in initial]
    mismatch = _first_mismatch([str(value) for value in budgets])
    if mismatch is not None:
        raise SeedErasureContractError(
            "Declared parameter budgets differ between compile outcomes."
        )
    for evidence in initial:
        if evidence.model.unique_parameter_count != evidence.outcome.parameter_budget:
            raise SeedErasureContractError(
                f"{evidence.label!r} declares parameter budget "
                f"{evidence.outcome.parameter_budget}, but the model has "
                f"{evidence.model.unique_parameter_count} unique parameters."
            )

    report_topologies = [_json_topology(evidence.compiler_report) for evidence in initial]
    mismatch = _first_mismatch(
        [
            _json_sha256(value, context="compiler-report topology")
            for value in report_topologies
        ]
    )
    if mismatch is not None:
        raise SeedErasureContractError(
            "Compiler-report structure differs between "
            f"{initial[0].label!r} and {initial[mismatch].label!r}; "
            "missing/renamed report fields cannot be excepted."
        )

    tensor_maps = [
        {tensor.name: tensor for tensor in evidence.model.named_tensors}
        for evidence in initial
    ]
    tensor_names = tuple(sorted(tensor_maps[0]))
    for mapping in tensor_maps[1:]:
        if tuple(sorted(mapping)) != tensor_names:
            raise SeedErasureContractError(
                "Named tensor inventory differs between compile outcomes."
            )
    differing_tensors = tuple(
        name
        for name in tensor_names
        if len({mapping[name].sha256 for mapping in tensor_maps}) > 1
    )

    declared_tensor_names = {field.name for field in scope.tensors}
    missing_tensor_exceptions = declared_tensor_names.difference(tensor_names)
    if missing_tensor_exceptions:
        raise DishonestSeedExceptionError(
            "Declared tensor exceptions do not exist in every outcome: "
            + ", ".join(sorted(missing_tensor_exceptions))
        )
    unused_tensor_exceptions = declared_tensor_names.difference(differing_tensors)
    if unused_tensor_exceptions:
        raise DishonestSeedExceptionError(
            "Declared tensor exceptions do not actually differ: "
            + ", ".join(sorted(unused_tensor_exceptions))
        )
    undeclared_tensors = set(differing_tensors).difference(declared_tensor_names)
    if undeclared_tensors:
        raise SeedLeakError(
            "Undeclared seed-dependent named tensors: "
            + ", ".join(sorted(undeclared_tensors))
        )

    normalized_reports = [copy.deepcopy(evidence.compiler_report) for evidence in initial]
    differing_report_paths: list[str] = []
    for field in scope.compiler_report_paths:
        values = [_resolve_pointer(report, field.name) for report in normalized_reports]
        serialized_values = [
            _canonical_json(value, context=f"compiler report path {field.name!r}")
            for value in values
        ]
        if len(set(serialized_values)) == 1:
            raise DishonestSeedExceptionError(
                f"Compiler-report exception {field.name!r} does not actually differ."
            )
        differing_report_paths.append(field.name)
        for report in normalized_reports:
            _replace_pointer(report, field.name)

    reference_report = normalized_reports[0]
    undeclared_report_paths: set[str] = set()
    for report in normalized_reports[1:]:
        undeclared_report_paths.update(_json_differences(reference_report, report))
    if undeclared_report_paths:
        raise SeedLeakError(
            "Undeclared compiler-report differences: "
            + ", ".join(sorted(undeclared_report_paths))
        )

    callback_results: list[tuple[NamedOutputFingerprint, ...]] = []
    for evidence in initial:
        try:
            callback_value = output_fingerprints(evidence.outcome)
        except Exception as exc:
            raise SeedErasureContractError(
                f"Output fingerprint callback failed for {evidence.label!r}."
            ) from exc
        callback_results.append(
            _normalize_output_fingerprints(callback_value, label=evidence.label)
        )

    # Recapture all outcomes after all callbacks, including cross-outcome closure
    # side effects.  A callback may observe a model, but it cannot alter evidence.
    for evidence in initial:
        final_model = _model_capture(evidence.outcome)
        if final_model != evidence.model:
            raise SeedErasureContractError(
                f"Output fingerprint callback mutated model {evidence.label!r}."
            )
        final_report = _normalize_compiler_report(evidence.outcome.compiler_report)
        if final_report != evidence.compiler_report:
            raise SeedErasureContractError(
                f"Output fingerprint callback mutated compiler report {evidence.label!r}."
            )

    output_maps = [
        {fingerprint.name: fingerprint.sha256 for fingerprint in fingerprints}
        for fingerprints in callback_results
    ]
    output_names = tuple(sorted(output_maps[0]))
    for mapping in output_maps[1:]:
        if tuple(sorted(mapping)) != output_names:
            raise SeedErasureContractError(
                "Behavioral output fingerprint inventory differs between outcomes."
            )
    differing_outputs = tuple(
        name
        for name in output_names
        if len({mapping[name] for mapping in output_maps}) > 1
    )

    declared_output_names = {field.name for field in scope.outputs}
    missing_output_exceptions = declared_output_names.difference(output_names)
    if missing_output_exceptions:
        raise DishonestSeedExceptionError(
            "Declared output exceptions do not exist in every outcome: "
            + ", ".join(sorted(missing_output_exceptions))
        )
    unused_output_exceptions = declared_output_names.difference(differing_outputs)
    if unused_output_exceptions:
        raise DishonestSeedExceptionError(
            "Declared output exceptions do not actually differ: "
            + ", ".join(sorted(unused_output_exceptions))
        )
    undeclared_outputs = set(differing_outputs).difference(declared_output_names)
    if undeclared_outputs:
        raise SeedLeakError(
            "Undeclared seed-dependent behavioral outputs: "
            + ", ".join(sorted(undeclared_outputs))
        )
    if differing_outputs and not differing_tensors:
        raise DishonestSeedExceptionError(
            "Output differences cannot be declared seed-owned when every named "
            "model tensor is seed-invariant."
        )

    outcome_audits = []
    for evidence, fingerprints in zip(initial, callback_results):
        output_payload = [fingerprint.as_dict() for fingerprint in fingerprints]
        outcome_audits.append(
            SeedOutcomeAudit(
                label=evidence.label,
                seed=evidence.outcome.seed,
                declared_parameter_budget=evidence.outcome.parameter_budget,
                unique_parameter_count=evidence.model.unique_parameter_count,
                structure_sha256=evidence.model.structure_sha256,
                tensor_inventory_sha256=evidence.model.tensor_inventory_sha256,
                tensor_manifest_sha256=evidence.model.tensor_manifest_sha256,
                compiler_report_sha256=evidence.compiler_report_sha256,
                output_manifest_sha256=_json_sha256(
                    output_payload,
                    context="output fingerprint manifest",
                ),
                named_tensors=evidence.model.named_tensors,
                output_fingerprints=fingerprints,
            )
        )

    tensor_invariant = not differing_tensors
    output_invariant = not differing_outputs
    seed_invariant = tensor_invariant and output_invariant
    return SeedErasureReport(
        schema=_SCHEMA,
        outcome_count=len(initial),
        seeds=tuple(evidence.outcome.seed for evidence in initial),
        outcomes=tuple(outcome_audits),
        declared_exceptions=scope,
        common_structure_sha256=structure_hashes[0],
        exact_structure_match=True,
        exact_parameter_budget_match=True,
        named_tensor_inventory_match=True,
        compiler_reports_equivalent_after_declared_exceptions=True,
        output_inventory_match=True,
        differing_tensor_names=differing_tensors,
        differing_output_names=differing_outputs,
        differing_compiler_report_paths=tuple(differing_report_paths),
        named_tensors_seed_invariant=tensor_invariant,
        behavioral_outputs_seed_invariant=output_invariant,
        initial_randomness_remains_in_named_tensors=not tensor_invariant,
        initial_randomness_observable_in_outputs=not output_invariant,
        seed_leak_detected=not seed_invariant,
        seed_invariant=seed_invariant,
        seed_erasure_proven=seed_invariant,
        proof_scope=(
            "Exact named parameter/buffer content plus caller-supplied behavioral "
            "fingerprints for only the independently seeded outcomes supplied to "
            "this audit; this is not a universal proof over untested seeds or inputs."
        ),
    )


__all__ = [
    "DishonestSeedExceptionError",
    "NamedOutputFingerprint",
    "NamedTensorDigest",
    "OutputFingerprintCallback",
    "SeedCompileOutcome",
    "SeedErasureContractError",
    "SeedErasureError",
    "SeedErasureReport",
    "SeedLeakError",
    "SeedOutcomeAudit",
    "SeedOwnedExceptionScope",
    "SeedOwnedField",
    "audit_seed_erasure",
]
