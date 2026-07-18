"""Authenticated inference-entrypoint execution observation.

Behavior evaluators may legitimately enter a model through a wrapper API
instead of ``nn.Module.__call__``.  This module keeps that exception narrow:
the wrapper entrypoint and the exact module root it must execute are sealed in
one canonical contract.  The default contract remains a top-level model call.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from threading import local
from typing import Any

from .dynamic_circuit_ir import canonical_sha256


INFERENCE_EXECUTION_SCHEMA = (
    "neural-forge-behavior-inference-execution-contract-v1"
)


class DynamicBehaviorExecutionError(RuntimeError):
    """A behavior inference contract or its observed execution is invalid."""


def _entrypoint(value: Any) -> str:
    if not isinstance(value, str) or not value or not value.isidentifier():
        raise DynamicBehaviorExecutionError(
            "Inference entrypoint must be one Python identifier."
        )
    if value.startswith("_") and value != "__call__":
        raise DynamicBehaviorExecutionError(
            "Inference entrypoint must be public or __call__."
        )
    return value


def _module_path(value: Any) -> str:
    if not isinstance(value, str):
        raise DynamicBehaviorExecutionError(
            "Required inference module path must be a string."
        )
    if value and any(
        not segment or not segment.isidentifier() for segment in value.split(".")
    ):
        raise DynamicBehaviorExecutionError(
            "Required inference module path must be empty or dot-separated "
            "Python identifiers."
        )
    return value


@dataclass(frozen=True, slots=True)
class BehaviorInferenceExecutionContract:
    """One wrapper entrypoint and exact module root required for acceptance."""

    entrypoint: str = "__call__"
    required_module_path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "entrypoint", _entrypoint(self.entrypoint))
        object.__setattr__(
            self,
            "required_module_path",
            _module_path(self.required_module_path),
        )
        if self.entrypoint == "__call__" and self.required_module_path:
            raise DynamicBehaviorExecutionError(
                "__call__ inference requires the top-level model root."
            )

    @classmethod
    def top_level_forward(cls) -> BehaviorInferenceExecutionContract:
        return cls()

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.body())

    def body(self) -> dict[str, str]:
        return {
            "schema": INFERENCE_EXECUTION_SCHEMA,
            "entrypoint": self.entrypoint,
            "required_module_path": self.required_module_path,
        }

    def as_dict(self) -> dict[str, str]:
        return {
            **self.body(),
            "contract_fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> BehaviorInferenceExecutionContract:
        if not isinstance(value, Mapping):
            raise DynamicBehaviorExecutionError(
                "Inference execution contract must be one JSON object."
            )
        if set(value) != {
            "schema",
            "entrypoint",
            "required_module_path",
            "contract_fingerprint",
        }:
            raise DynamicBehaviorExecutionError(
                "Inference execution contract fields are incomplete or unknown."
            )
        if value.get("schema") != INFERENCE_EXECUTION_SCHEMA:
            raise DynamicBehaviorExecutionError(
                "Inference execution contract schema is unsupported."
            )
        contract = cls(
            entrypoint=value.get("entrypoint"),
            required_module_path=value.get("required_module_path"),
        )
        if value.get("contract_fingerprint") != contract.fingerprint:
            raise DynamicBehaviorExecutionError(
                "Inference execution contract seal is invalid."
            )
        return contract


def inference_execution_contract_from_manifest(
    manifest: Mapping[str, Any],
) -> BehaviorInferenceExecutionContract:
    """Resolve a sealed contract; legacy manifests safely retain root-forward."""

    if not isinstance(manifest, Mapping):
        raise DynamicBehaviorExecutionError(
            "Behavior gate manifest must be one JSON object."
        )
    value = manifest.get("inference_execution")
    if value is None:
        return BehaviorInferenceExecutionContract.top_level_forward()
    return BehaviorInferenceExecutionContract.from_dict(value)


def resolve_behavior_inference_execution_contract(
    evaluator: Any,
) -> BehaviorInferenceExecutionContract:
    """Resolve an exposed evaluator contract, defaulting safely to root-forward."""

    candidates = [evaluator]
    bound_owner = getattr(evaluator, "__self__", None)
    if bound_owner is not None:
        candidates.append(bound_owner)
    explicit_owner = getattr(evaluator, "behavior_gate_owner", None)
    if explicit_owner is not None:
        candidates.append(explicit_owner)
    for candidate in candidates:
        manifest = getattr(candidate, "behavior_gate_manifest", None)
        contract = getattr(
            candidate,
            "behavior_inference_execution_contract",
            None,
        )
        if isinstance(manifest, Mapping):
            sealed = inference_execution_contract_from_manifest(manifest)
            if contract is not None and contract != sealed:
                raise DynamicBehaviorExecutionError(
                    "Evaluator inference execution contract differs from its "
                    "sealed behavior-gate manifest."
                )
            return sealed
        if contract is not None:
            if not isinstance(contract, BehaviorInferenceExecutionContract):
                raise DynamicBehaviorExecutionError(
                    "Evaluator exposes an invalid inference execution contract."
                )
            return contract
    return BehaviorInferenceExecutionContract.top_level_forward()


@contextmanager
def observe_model_execution(
    model: Any,
    *,
    contract: BehaviorInferenceExecutionContract | None = None,
):
    """Count execution of all modules and the contract's exact inference root."""

    selected = (
        BehaviorInferenceExecutionContract.top_level_forward()
        if contract is None
        else contract
    )
    if not isinstance(selected, BehaviorInferenceExecutionContract):
        raise DynamicBehaviorExecutionError(
            "Execution observation contract has the wrong type."
        )
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise DynamicBehaviorExecutionError(
            "Binding behavior evaluation requires a model with named_modules()."
        )
    try:
        modules = tuple(named_modules())
    except (TypeError, ValueError) as exc:
        raise DynamicBehaviorExecutionError(
            "Could not enumerate modules for behavior execution observation."
        ) from exc
    if not modules:
        raise DynamicBehaviorExecutionError(
            "Behavior execution observation found no model modules."
        )
    module_paths = {str(name) for name, _module in modules}
    if selected.required_module_path not in module_paths:
        label = selected.required_module_path or "<root>"
        raise DynamicBehaviorExecutionError(
            f"Required inference module path {label!r} does not exist."
        )
    counts = {
        "module_forward_calls": 0,
        "root_forward_calls": 0,
        "inference_entrypoint_calls": 0,
    }
    entrypoint_state = local()
    handles = []
    entrypoint_had_instance_value = False
    entrypoint_instance_value: Any = None
    entrypoint_patched = False

    def hook(_module: Any, _inputs: Any) -> None:
        counts["module_forward_calls"] += 1

    def required_root_hook(
        _module: Any,
        _inputs: Any,
        _output: Any,
    ) -> None:
        counts["module_forward_calls"] += 1
        if selected.entrypoint == "__call__":
            counts["root_forward_calls"] += 1
            counts["inference_entrypoint_calls"] += 1
        elif int(getattr(entrypoint_state, "depth", 0)) > 0:
            counts["root_forward_calls"] += 1

    try:
        for name, module in modules:
            register = getattr(
                module,
                (
                    "register_forward_hook"
                    if name == selected.required_module_path
                    else "register_forward_pre_hook"
                ),
                None,
            )
            if not callable(register):
                raise DynamicBehaviorExecutionError(
                    f"Model module {name or '<root>'!r} cannot be observed."
                )
            handles.append(
                register(
                    required_root_hook if name == selected.required_module_path else hook
                )
            )
        if selected.entrypoint != "__call__":
            entrypoint_callable = getattr(model, selected.entrypoint, None)
            if not callable(entrypoint_callable):
                raise DynamicBehaviorExecutionError(
                    f"Declared inference entrypoint {selected.entrypoint!r} "
                    "is not callable on the model."
                )
            instance_attributes = getattr(model, "__dict__", {})
            entrypoint_had_instance_value = (
                isinstance(instance_attributes, Mapping)
                and selected.entrypoint in instance_attributes
            )
            if entrypoint_had_instance_value:
                entrypoint_instance_value = instance_attributes[selected.entrypoint]

            def observed_entrypoint(*args: Any, **kwargs: Any) -> Any:
                counts["inference_entrypoint_calls"] += 1
                depth = int(getattr(entrypoint_state, "depth", 0))
                entrypoint_state.depth = depth + 1
                try:
                    return entrypoint_callable(*args, **kwargs)
                finally:
                    entrypoint_state.depth = depth

            object.__setattr__(model, selected.entrypoint, observed_entrypoint)
            entrypoint_patched = True
        yield counts
    finally:
        if entrypoint_patched:
            if entrypoint_had_instance_value:
                object.__setattr__(
                    model,
                    selected.entrypoint,
                    entrypoint_instance_value,
                )
            else:
                object.__delattr__(model, selected.entrypoint)
        for handle in reversed(handles):
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()


def execution_contract_satisfied(counts: Mapping[str, Any]) -> bool:
    """Return whether both the declared API and exact module root executed."""

    for name in (
        "root_forward_calls",
        "inference_entrypoint_calls",
    ):
        value = counts.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return False
    return True


__all__ = [
    "BehaviorInferenceExecutionContract",
    "DynamicBehaviorExecutionError",
    "INFERENCE_EXECUTION_SCHEMA",
    "execution_contract_satisfied",
    "inference_execution_contract_from_manifest",
    "observe_model_execution",
    "resolve_behavior_inference_execution_contract",
]
