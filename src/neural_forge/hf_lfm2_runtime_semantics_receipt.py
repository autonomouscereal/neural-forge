"""Source-authenticated semantics receipt for native LFM2 trace capture.

The module is registry-inert: importing it loads only standard-library helpers
and Neural Forge's canonical JSON implementation.  Torch and Transformers are
loaded lazily when a caller asks to create or verify the receipt.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import sys
import threading
import types
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256


LFM2_RUNTIME_SEMANTICS_RECEIPT_SCHEMA = "neural-forge-lfm2-runtime-semantics-receipt-v2"


class Lfm2RuntimeSemanticsReceiptError(RuntimeError):
    """Raised when live capture semantics differ from installed source."""


_RECEIPT_LOCK = threading.RLock()
_CANONICAL_RECEIPT: Mapping[str, Any] = MappingProxyType({})
_CANONICAL_RECEIPT_SHA256 = ""


def _code_constant_body(value: Any) -> Any:
    if isinstance(value, types.CodeType):
        return _code_object_body(value)
    if value is None or type(value) in {bool, int, float, str, bytes}:
        return value.hex() if isinstance(value, bytes) else value
    if isinstance(value, tuple):
        return [_code_constant_body(item) for item in value]
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _code_object_body(code: types.CodeType) -> dict[str, Any]:
    return {
        "bytecode": code.co_code.hex(),
        "constants": [_code_constant_body(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
    }


def runtime_callable_record(value: Any, name: str) -> dict[str, Any]:
    """Return a stable shallow callable identity used by model receipts."""

    code = getattr(value, "__code__", None)
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if (
        not callable(value)
        or not isinstance(module, str)
        or not module
        or not isinstance(qualname, str)
        or not qualname
    ):
        raise Lfm2RuntimeSemanticsReceiptError(
            f"{name} is not an authenticated Python runtime method."
        )
    return {
        "module": module,
        "qualname": qualname,
        "callable_kind": type(value).__qualname__,
        "code_sha256": (
            canonical_sha256(_code_object_body(code))
            if isinstance(code, types.CodeType)
            else None
        ),
    }


def _runtime_semantic_value_body(
    value: Any,
    name: str,
    seen: set[int],
    *,
    trusted_unloaded_globals: bool,
) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, float):
        return {"kind": "float", "hex": value.hex()}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "sha256": hashlib.sha256(value).hexdigest(),
            "size": len(value),
        }
    if isinstance(value, tuple):
        return [
            _runtime_semantic_value_body(
                item,
                f"{name}[{index}]",
                seen,
                trusted_unloaded_globals=trusted_unloaded_globals,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _runtime_semantic_value_body(
                item,
                f"{name}.{key}",
                seen,
                trusted_unloaded_globals=trusted_unloaded_globals,
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if callable(value):
        if not isinstance(getattr(value, "__module__", None), str) or not isinstance(
            getattr(value, "__qualname__", None), str
        ):
            return {
                "kind": "callable_object",
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
            }
        return {
            "kind": "callable",
            "record": source_callable_semantics_record(
                value,
                name,
                seen,
                trusted_unloaded_globals=trusted_unloaded_globals,
                include_referenced_globals=False,
                require_exact_module_binding=False,
            ),
        }
    return {
        "kind": "value",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }


def _referenced_module_binding_body(
    value: types.ModuleType,
    name: str,
    code_names: frozenset[str],
    seen: set[int],
    *,
    trusted_unloaded_globals: bool,
) -> dict[str, Any]:
    """Authenticate one module global and the attributes this code executes."""

    module_name = getattr(value, "__name__", None)
    if not isinstance(module_name, str) or sys.modules.get(module_name) is not value:
        raise Lfm2RuntimeSemanticsReceiptError(
            f"{name} references a foreign or unregistered module global."
        )
    source_value = getattr(value, "__file__", None)
    source_sha256 = None
    if isinstance(source_value, str):
        source_path = Path(source_value).resolve()
        if not source_path.is_file() or source_path.is_symlink():
            raise Lfm2RuntimeSemanticsReceiptError(
                f"{name} references a module without regular source."
            )
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    attributes = {}
    for attribute_name in sorted(code_names):
        if attribute_name == module_name or not hasattr(value, attribute_name):
            continue
        attribute = getattr(value, attribute_name)
        if attribute is value:
            attributes[attribute_name] = {"kind": "self"}
        else:
            attributes[attribute_name] = _runtime_semantic_value_body(
                attribute,
                f"{name}.{attribute_name}",
                seen,
                trusted_unloaded_globals=trusted_unloaded_globals,
            )

    # These Python names are the public bindings of immutable native entrypoints.
    # A fresh source import still observes the same live ``torch`` module, so the
    # receipt must independently close them over their extension authority.
    if module_name == "torch":
        native = getattr(getattr(value, "_C", None), "_VariableFunctions", None)
        for attribute_name in set(attributes) & {"cat", "chunk", "conv1d"}:
            if native is None or getattr(value, attribute_name) is not getattr(
                native, attribute_name, None
            ):
                raise Lfm2RuntimeSemanticsReceiptError(
                    f"{name}.{attribute_name} differs from its native torch binding."
                )
    if module_name == "torch.nn.functional":
        torch_module = sys.modules.get("torch")
        if "linear" in attributes and (
            torch_module is None
            or value.linear is not getattr(torch_module._C._nn, "linear", None)
        ):
            raise Lfm2RuntimeSemanticsReceiptError(
                f"{name}.linear differs from its native torch binding."
            )
        if "conv1d" in attributes and (
            torch_module is None or value.conv1d is not torch_module.conv1d
        ):
            raise Lfm2RuntimeSemanticsReceiptError(
                f"{name}.conv1d differs from its native torch binding."
            )

    return {
        "kind": "module",
        "module": module_name,
        "source_sha256": source_sha256,
        "referenced_attributes": attributes,
    }


def _referenced_global_binding_body(
    owner: Any,
    value: Any,
    name: str,
    code_names: frozenset[str],
    seen: set[int],
    *,
    trusted_unloaded_globals: bool,
) -> Any:
    if value is owner:
        return {"kind": "self"}
    if isinstance(value, types.ModuleType):
        return _referenced_module_binding_body(
            value,
            name,
            code_names,
            seen,
            trusted_unloaded_globals=trusted_unloaded_globals,
        )
    return _runtime_semantic_value_body(
        value,
        name,
        seen,
        trusted_unloaded_globals=trusted_unloaded_globals,
    )


def source_callable_semantics_record(
    value: Any,
    name: str,
    seen: set[int] | None = None,
    *,
    trusted_unloaded_globals: bool = False,
    include_referenced_globals: bool = True,
    require_exact_module_binding: bool = True,
) -> dict[str, Any]:
    """Describe Python callable semantics, including defaults and closures."""

    active = set() if seen is None else seen
    if id(value) in active:
        raise Lfm2RuntimeSemanticsReceiptError(
            f"{name} contains a recursive callable closure."
        )
    active.add(id(value))
    try:
        record = runtime_callable_record(value, name)
        code = getattr(value, "__code__", None)
        if not isinstance(code, types.CodeType):
            return {
                "qualname": record["qualname"],
                "callable_kind": record["callable_kind"],
                "code_sha256": None,
                "defining_source_sha256": None,
                "defaults": None,
                "kwdefaults": None,
                "closure": None,
            }
        module_name = getattr(value, "__module__", None)
        globals_mapping = getattr(value, "__globals__", None)
        globals_module_name = (
            globals_mapping.get("__name__") if type(globals_mapping) is dict else None
        )
        loaded_module = (
            sys.modules.get(globals_module_name)
            if isinstance(globals_module_name, str)
            else None
        )
        source_path = Path(code.co_filename).resolve()
        declared_module = (
            sys.modules.get(module_name) if isinstance(module_name, str) else None
        )
        declared_source_value = getattr(declared_module, "__file__", None)
        declared_source_path = (
            Path(declared_source_value).resolve()
            if isinstance(declared_source_value, str)
            else None
        )
        directly_defined_in_declared_module = declared_source_path == source_path
        if (
            not isinstance(module_name, str)
            or type(globals_mapping) is not dict
            or not isinstance(globals_module_name, str)
            or (
                require_exact_module_binding
                and not trusted_unloaded_globals
                and directly_defined_in_declared_module
                and module_name != globals_module_name
            )
        ):
            raise Lfm2RuntimeSemanticsReceiptError(
                f"{name} has no exact Python defining-globals mapping."
            )
        if isinstance(loaded_module, types.ModuleType):
            if globals_mapping is not vars(loaded_module):
                raise Lfm2RuntimeSemanticsReceiptError(
                    f"{name} is rebound to a foreign defining-globals mapping."
                )
        elif not trusted_unloaded_globals or not isinstance(
            globals_mapping.get("__file__"), str
        ):
            raise Lfm2RuntimeSemanticsReceiptError(
                f"{name} is not bound to its independently loaded module globals."
            )
        if not source_path.is_file() or source_path.is_symlink():
            raise Lfm2RuntimeSemanticsReceiptError(
                f"{name} does not resolve to a regular defining source file."
            )
        globals_source_value = globals_mapping.get("__file__")
        if (
            not isinstance(globals_source_value, str)
            or Path(globals_source_value).resolve() != source_path
        ):
            raise Lfm2RuntimeSemanticsReceiptError(
                f"{name} code and defining-module source paths differ."
            )
        closure = getattr(value, "__closure__", None)
        closure_body = None
        if closure is not None:
            closure_body = []
            for index, cell in enumerate(closure):
                try:
                    contents = cell.cell_contents
                except ValueError as exc:
                    raise Lfm2RuntimeSemanticsReceiptError(
                        f"{name} has an empty closure cell."
                    ) from exc
                closure_body.append(
                    _runtime_semantic_value_body(
                        contents,
                        f"{name}.closure[{index}]",
                        active,
                        trusted_unloaded_globals=trusted_unloaded_globals,
                    )
                )
        referenced_global_names = sorted(
            global_name
            for global_name in set(code.co_names)
            if global_name in globals_mapping
        )
        code_names = frozenset(code.co_names)
        return {
            "qualname": record["qualname"],
            "callable_kind": record["callable_kind"],
            "code_sha256": record["code_sha256"],
            "defining_source_sha256": hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest(),
            "defaults": _runtime_semantic_value_body(
                getattr(value, "__defaults__", None),
                f"{name}.defaults",
                active,
                trusted_unloaded_globals=trusted_unloaded_globals,
            ),
            "kwdefaults": _runtime_semantic_value_body(
                getattr(value, "__kwdefaults__", None),
                f"{name}.kwdefaults",
                active,
                trusted_unloaded_globals=trusted_unloaded_globals,
            ),
            "closure": closure_body,
            "defining_globals_authenticated": True,
            "referenced_global_names": referenced_global_names,
            "referenced_globals": (
                {
                    global_name: _referenced_global_binding_body(
                        value,
                        globals_mapping[global_name],
                        f"{name}.globals[{global_name!r}]",
                        code_names,
                        active,
                        trusted_unloaded_globals=trusted_unloaded_globals,
                    )
                    for global_name in referenced_global_names
                }
                if include_referenced_globals
                else None
            ),
        }
    finally:
        active.remove(id(value))


def _identity_mapping_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    return set(left) == set(right) and all(left[name] is right[name] for name in left)


def _loaded_attention_registry_state() -> (
    tuple[
        Any,
        dict[str, Any],
        dict[str, Any],
    ]
    | None
):
    modeling_utils = sys.modules.get("transformers.modeling_utils")
    registry = getattr(modeling_utils, "ALL_ATTENTION_FUNCTIONS", None)
    local_mapping = (
        vars(registry).get("_local_mapping") if registry is not None else None
    )
    global_mapping = (
        getattr(type(registry), "_global_mapping", None)
        if registry is not None
        else None
    )
    if type(local_mapping) is not dict or type(global_mapping) is not dict:
        return None
    return registry, dict(local_mapping), dict(global_mapping)


def _restore_attention_registry_state(
    state: tuple[Any, dict[str, Any], dict[str, Any]] | None,
) -> bool:
    if state is None:
        return False
    registry, expected_local, expected_global = state
    local_mapping = vars(registry).get("_local_mapping")
    global_mapping = getattr(type(registry), "_global_mapping", None)
    changed = (
        type(local_mapping) is not dict
        or type(global_mapping) is not dict
        or not _identity_mapping_equal(local_mapping, expected_local)
        or not _identity_mapping_equal(global_mapping, expected_global)
    )
    if type(local_mapping) is dict:
        local_mapping.clear()
        local_mapping.update(expected_local)
    if type(global_mapping) is dict:
        global_mapping.clear()
        global_mapping.update(expected_global)
    return changed


def load_fresh_source_module(
    live_module: types.ModuleType,
    alias: str,
    label: str,
) -> tuple[types.ModuleType, str]:
    """Execute an installed source file under one temporary sibling alias."""

    source_value = getattr(live_module, "__file__", None)
    if not isinstance(source_value, str):
        raise Lfm2RuntimeSemanticsReceiptError(f"{label} has no defining source file.")
    source_path = Path(source_value).resolve()
    if not source_path.is_file() or source_path.is_symlink() or alias in sys.modules:
        raise Lfm2RuntimeSemanticsReceiptError(
            f"{label} cannot be independently loaded from a regular source file."
        )
    before_modules = frozenset(sys.modules)
    attention_registry_state = _loaded_attention_registry_state()
    live_registry_present = "ALL_ATTENTION_FUNCTIONS" in vars(live_module)
    live_registry = vars(live_module).get("ALL_ATTENTION_FUNCTIONS")
    spec = importlib.util.spec_from_file_location(alias, source_path)
    if spec is None or spec.loader is None:
        raise Lfm2RuntimeSemanticsReceiptError(
            f"{label} cannot create an independent source loader."
        )
    fresh = importlib.util.module_from_spec(spec)
    sys.modules[alias] = fresh
    execution_error: BaseException | None = None
    try:
        spec.loader.exec_module(fresh)
    except BaseException as exc:
        execution_error = exc
    finally:
        sys.modules.pop(alias, None)
    unexpected_modules = sorted(set(sys.modules).difference(before_modules))
    for module_name in sorted(
        unexpected_modules,
        key=lambda value: (value.count("."), value),
        reverse=True,
    ):
        imported_module = sys.modules.pop(module_name, None)
        parent_name, separator, child_name = module_name.rpartition(".")
        parent = sys.modules.get(parent_name) if separator else None
        if (
            isinstance(parent, types.ModuleType)
            and getattr(parent, child_name, None) is imported_module
        ):
            delattr(parent, child_name)
    registry_changed = _restore_attention_registry_state(attention_registry_state)
    live_registry_changed = (
        "ALL_ATTENTION_FUNCTIONS" in vars(live_module)
    ) != live_registry_present or (
        live_registry_present
        and vars(live_module).get("ALL_ATTENTION_FUNCTIONS") is not live_registry
    )
    if live_registry_changed:
        if live_registry_present:
            setattr(live_module, "ALL_ATTENTION_FUNCTIONS", live_registry)
        else:
            vars(live_module).pop("ALL_ATTENTION_FUNCTIONS", None)
    if execution_error is not None:
        error = Lfm2RuntimeSemanticsReceiptError(
            f"{label} independent source authentication failed."
        )
        if unexpected_modules:
            error.add_note(
                "Fresh source imports were removed after failure: "
                f"{unexpected_modules}."
            )
        if registry_changed or live_registry_changed:
            error.add_note(
                "Fresh source attention-registry mutations were restored after failure."
            )
        raise error from execution_error
    if unexpected_modules:
        raise Lfm2RuntimeSemanticsReceiptError(
            f"{label} independent source authentication imported persistent modules: "
            f"{unexpected_modules}."
        )
    if registry_changed or live_registry_changed:
        raise Lfm2RuntimeSemanticsReceiptError(
            f"{label} independent source authentication mutated the attention registry."
        )
    return fresh, hashlib.sha256(source_path.read_bytes()).hexdigest()


def _compiled_source_function_record(
    module: types.ModuleType,
    qualname: str,
    label: str,
) -> dict[str, Any]:
    source_value = getattr(module, "__file__", None)
    if not isinstance(source_value, str):
        raise Lfm2RuntimeSemanticsReceiptError(f"{label} has no defining source file.")
    source_path = Path(source_value).resolve()
    if not source_path.is_file() or source_path.is_symlink():
        raise Lfm2RuntimeSemanticsReceiptError(f"{label} source is not a regular file.")
    try:
        root = compile(
            source_path.read_bytes(),
            str(source_path),
            "exec",
            dont_inherit=True,
        )
    except (OSError, SyntaxError, ValueError) as exc:
        raise Lfm2RuntimeSemanticsReceiptError(
            f"{label} source cannot be independently compiled."
        ) from exc
    matches: list[types.CodeType] = []

    def visit(code: types.CodeType) -> None:
        for value in code.co_consts:
            if isinstance(value, types.CodeType):
                if value.co_qualname == qualname:
                    matches.append(value)
                visit(value)

    visit(root)
    if len(matches) != 1:
        raise Lfm2RuntimeSemanticsReceiptError(
            f"{label} source does not define exactly one {qualname!r} callable."
        )
    return {
        "qualname": qualname,
        "code_sha256": canonical_sha256(_code_object_body(matches[0])),
        "defining_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }


def _independent_defining_callable_record(value: Any, label: str) -> dict[str, Any]:
    module_name = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    defining_module = (
        sys.modules.get(module_name) if isinstance(module_name, str) else None
    )
    if not isinstance(defining_module, types.ModuleType) or not isinstance(
        qualname, str
    ):
        raise Lfm2RuntimeSemanticsReceiptError(
            f"{label} has no authenticated defining module."
        )
    resolved_value: Any = defining_module
    for part in qualname.split("."):
        if not part or part == "<locals>" or not hasattr(resolved_value, part):
            raise Lfm2RuntimeSemanticsReceiptError(
                f"{label} is not exported by its authenticated defining module."
            )
        resolved_value = getattr(resolved_value, part)
    if resolved_value is not value:
        raise Lfm2RuntimeSemanticsReceiptError(
            f"{label} differs from its authenticated defining-module export."
        )
    source_value = getattr(defining_module, "__file__", None)
    if not isinstance(source_value, str):
        raise Lfm2RuntimeSemanticsReceiptError(
            f"{label} defining module has no source or extension file."
        )
    defining_path = Path(source_value).resolve()
    if not defining_path.is_file() or defining_path.is_symlink():
        raise Lfm2RuntimeSemanticsReceiptError(
            f"{label} defining module is not a regular file."
        )
    result = {
        "module": module_name,
        "qualname": qualname,
        "callable_kind": type(value).__qualname__,
        "defining_export_authenticated": True,
        "defining_module_sha256": hashlib.sha256(
            defining_path.read_bytes()
        ).hexdigest(),
        "semantics": source_callable_semantics_record(value, label),
    }
    if isinstance(getattr(value, "__code__", None), types.CodeType):
        compiled = _compiled_source_function_record(
            defining_module,
            qualname,
            label,
        )
        if compiled["code_sha256"] != result["semantics"]["code_sha256"]:
            raise Lfm2RuntimeSemanticsReceiptError(
                f"{label} differs from its independently compiled defining source."
            )
        result["compiled_source"] = compiled
    else:
        result["compiled_source"] = None
    return result


def _class_record(
    value: type[Any],
    *,
    name: str,
    module_override: str | None = None,
) -> dict[str, Any]:
    override_names = (
        "__call__",
        "_wrapped_call_impl",
        "_call_impl",
        "_slow_forward",
    )
    bases = []
    for base in value.__bases__:
        module = base.__module__
        if module_override is not None and module == value.__module__:
            module = module_override
        bases.append(f"{module}.{base.__qualname__}")
    return {
        "module": (value.__module__ if module_override is None else module_override),
        "qualname": value.__qualname__,
        "bases": bases,
        "local_call_overrides": [
            method_name for method_name in override_names if method_name in vars(value)
        ],
        "resolved_call": source_callable_semantics_record(
            value.__call__,
            f"class {name}.__call__",
            trusted_unloaded_globals=module_override is not None,
        ),
        "resolved_call_impl": source_callable_semantics_record(
            value._call_impl,
            f"class {name}._call_impl",
            trusted_unloaded_globals=module_override is not None,
        ),
    }


def _class_local_execution_record(
    value: type[Any],
    *,
    name: str,
    module_override: str | None = None,
) -> dict[str, Any]:
    """Seal one class shell and every locally defined Python execution member."""

    def member_semantics(member: Any, member_label: str) -> dict[str, Any]:
        result = source_callable_semantics_record(
            member,
            member_label,
            trusted_unloaded_globals=module_override is not None,
        )
        referenced = result.get("referenced_globals")
        if module_override is not None and isinstance(referenced, dict):
            if "__name__" in referenced:
                referenced["__name__"] = module_override
        return result

    members: dict[str, Any] = {}
    for member_name, raw_member in sorted(vars(value).items()):
        member_kind = "function"
        member = raw_member
        if isinstance(raw_member, staticmethod):
            member_kind = "staticmethod"
            member = raw_member.__func__
        elif isinstance(raw_member, classmethod):
            member_kind = "classmethod"
            member = raw_member.__func__
        if isinstance(member, types.FunctionType):
            members[member_name] = {
                "kind": member_kind,
                "semantics": member_semantics(
                    member,
                    f"class {name}.{member_name}",
                ),
            }
        elif isinstance(raw_member, property):
            property_record: dict[str, Any] = {"kind": "property"}
            for accessor_name, accessor in (
                ("fget", raw_member.fget),
                ("fset", raw_member.fset),
                ("fdel", raw_member.fdel),
            ):
                property_record[accessor_name] = (
                    member_semantics(
                        accessor,
                        f"class {name}.{member_name}.{accessor_name}",
                    )
                    if accessor is not None
                    else None
                )
            members[member_name] = property_record
    module_name = value.__module__ if module_override is None else module_override
    bases = [
        f"{(module_override if base.__module__ == value.__module__ else base.__module__)}."
        f"{base.__qualname__}"
        for base in value.__bases__
    ]
    return {
        "module": module_name,
        "qualname": value.__qualname__,
        "bases": bases,
        "members": members,
    }


def _cache_runtime_record(
    cache_utils: types.ModuleType,
    *,
    module_override: str | None = None,
) -> dict[str, Any]:
    """Seal the cache classes and mapping reached by ``Lfm2Model.forward``."""

    mapping = getattr(cache_utils, "LAYER_TYPE_CACHE_MAPPING", None)
    if not isinstance(mapping, Mapping):
        raise Lfm2RuntimeSemanticsReceiptError(
            "Transformers cache layer mapping is unavailable."
        )
    roots = [getattr(cache_utils, "DynamicCache"), getattr(cache_utils, "Cache")]
    roots.extend(mapping.values())
    classes: dict[str, type[Any]] = {}
    for root in roots:
        if not isinstance(root, type):
            raise Lfm2RuntimeSemanticsReceiptError(
                "Transformers cache layer mapping contains a non-class value."
            )
        for cls in root.__mro__:
            if cls is object or cls.__module__ != cache_utils.__name__:
                continue
            classes[cls.__qualname__] = cls
    normalized_module = cache_utils.__name__ if module_override is None else module_override
    for cls in classes.values():
        if module_override is None and cls.__module__ != normalized_module:
            raise Lfm2RuntimeSemanticsReceiptError(
                "Transformers cache runtime contains a foreign class shell."
            )
    return {
        "module": normalized_module,
        "source_sha256": hashlib.sha256(Path(cache_utils.__file__).read_bytes()).hexdigest(),
        "layer_type_mapping": {
            str(key): f"{normalized_module}.{value.__qualname__}"
            for key, value in sorted(mapping.items())
        },
        "classes": {
            name: _class_local_execution_record(
                cls,
                name=f"Transformers cache {name}",
                module_override=normalized_module,
            )
            for name, cls in sorted(classes.items())
        },
    }


def _authenticated_records(
    pairs: Mapping[str, tuple[Any, Any]],
    namespace: str,
) -> dict[str, Any]:
    records = {}
    for name, (live, canonical) in sorted(pairs.items()):
        expected = source_callable_semantics_record(
            canonical,
            f"canonical.{namespace}.{name}",
            trusted_unloaded_globals=True,
        )
        observed = source_callable_semantics_record(
            live,
            f"live.{namespace}.{name}",
        )
        if observed != expected:
            raise Lfm2RuntimeSemanticsReceiptError(
                f"Installed {namespace} callable {name!r} differs from its "
                "independently loaded defining source."
            )
        records[name] = expected
    return records


def _imported_value_record(value: Any, name: str) -> dict[str, Any]:
    if callable(value):
        return {
            "kind": "callable",
            "record": _independent_defining_callable_record(
                value,
                f"imported LFM2 runtime {name}",
            ),
        }
    if value is None or type(value) in {bool, int, float, str}:
        return {
            "kind": "value",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": value,
        }
    raise Lfm2RuntimeSemanticsReceiptError(
        f"Imported LFM2 runtime {name!r} has an unsupported value type."
    )


def _attention_registry_source_bindings(
    modeling_utils: types.ModuleType,
) -> dict[str, str]:
    source_value = getattr(modeling_utils, "__file__", None)
    if not isinstance(source_value, str):
        raise Lfm2RuntimeSemanticsReceiptError(
            "Transformers modeling_utils has no defining source."
        )
    source_path = Path(source_value).resolve()
    if not source_path.is_file() or source_path.is_symlink():
        raise Lfm2RuntimeSemanticsReceiptError(
            "Transformers modeling_utils source is not regular."
        )
    try:
        tree = ast.parse(source_path.read_bytes(), filename=str(source_path))
    except (OSError, SyntaxError, ValueError) as exc:
        raise Lfm2RuntimeSemanticsReceiptError(
            "Transformers modeling_utils source cannot be parsed."
        ) from exc
    candidates = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "AttentionInterface":
            for statement in node.body:
                if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                    continue
                target = statement.targets[0]
                if (
                    isinstance(target, ast.Name)
                    and target.id == "_global_mapping"
                    and isinstance(statement.value, ast.Dict)
                ):
                    candidates.append(statement.value)
    if len(candidates) != 1:
        raise Lfm2RuntimeSemanticsReceiptError(
            "Transformers AttentionInterface source has no unique global mapping."
        )
    result = {}
    for key, value in zip(candidates[0].keys, candidates[0].values):
        if (
            not isinstance(key, ast.Constant)
            or not isinstance(key.value, str)
            or not isinstance(value, ast.Name)
        ):
            raise Lfm2RuntimeSemanticsReceiptError(
                "Transformers AttentionInterface mapping is not statically auditable."
            )
        result[key.value] = value.id
    if not result:
        raise Lfm2RuntimeSemanticsReceiptError(
            "Transformers AttentionInterface mapping is empty."
        )
    return result


def _attention_registry_snapshot(
    modeling_utils: types.ModuleType,
    registry: Any,
) -> dict[str, Any]:
    source_bindings = _attention_registry_source_bindings(modeling_utils)
    local_mapping = vars(registry).get("_local_mapping")
    global_mapping = getattr(type(registry), "_global_mapping", None)
    if type(local_mapping) is not dict or local_mapping:
        raise Lfm2RuntimeSemanticsReceiptError(
            "LFM2 attention registry carries mutable local overrides."
        )
    if not isinstance(global_mapping, Mapping) or set(global_mapping) != set(
        source_bindings
    ):
        raise Lfm2RuntimeSemanticsReceiptError(
            "LFM2 attention registry keys differ from modeling_utils source."
        )
    observable = dict(registry.items())
    if set(observable) != set(source_bindings):
        raise Lfm2RuntimeSemanticsReceiptError(
            "LFM2 attention registry observable keys differ from its global mapping."
        )
    records = {}
    for key, global_name in sorted(source_bindings.items()):
        expected = getattr(modeling_utils, global_name, None)
        if (
            global_mapping.get(key) is not expected
            or observable.get(key) is not expected
        ):
            raise Lfm2RuntimeSemanticsReceiptError(
                f"LFM2 attention registry binding {key!r} differs from source."
            )
        records[key] = {
            "modeling_utils_global": global_name,
            "callable": _independent_defining_callable_record(
                expected,
                f"attention registry {key}",
            ),
        }
    source_path = Path(modeling_utils.__file__).resolve()
    return {
        "class": f"{type(registry).__module__}.{type(registry).__qualname__}",
        "modeling_utils_source_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "ordered_keys": sorted(source_bindings),
        "bindings": records,
        "local_mapping": {},
    }


def _canonical_receipt_body(torch: Any, modeling_lfm2: Any) -> dict[str, Any]:
    from transformers import cache_utils, modeling_utils
    from torch.nn.modules import container as torch_container
    from torch.nn.modules import conv as torch_conv
    from torch.nn.modules import linear as torch_linear
    from torch.nn.modules import module as torch_module
    from torch.nn.modules import sparse as torch_sparse

    registry = modeling_lfm2.ALL_ATTENTION_FUNCTIONS
    registry_before = {
        backend: registry.get_interface(
            backend,
            modeling_lfm2.eager_attention_forward,
        )
        for backend in ("eager", "sdpa")
    }
    fresh_lfm2, lfm2_source_sha256 = load_fresh_source_module(
        modeling_lfm2,
        "transformers.models.lfm2._neural_forge_capture_auth_modeling_lfm2",
        "Transformers LFM2 capture runtime",
    )
    fresh_cache_utils, cache_utils_source_sha256 = load_fresh_source_module(
        cache_utils,
        "transformers._neural_forge_capture_auth_cache_utils",
        "Transformers cache capture runtime",
    )
    canonical_cache_runtime = _cache_runtime_record(
        fresh_cache_utils,
        module_override="transformers.cache_utils",
    )
    live_cache_runtime = _cache_runtime_record(cache_utils)
    if (
        cache_utils_source_sha256 != canonical_cache_runtime["source_sha256"]
        or live_cache_runtime != canonical_cache_runtime
    ):
        raise Lfm2RuntimeSemanticsReceiptError(
            "Installed Transformers cache runtime differs from source."
        )
    if (
        registry is not fresh_lfm2.ALL_ATTENTION_FUNCTIONS
        or registry is not modeling_utils.ALL_ATTENTION_FUNCTIONS
    ):
        raise Lfm2RuntimeSemanticsReceiptError(
            "Installed LFM2 attention registry is not the canonical shared registry."
        )
    if any(
        registry.get_interface(backend, modeling_lfm2.eager_attention_forward)
        is not value
        for backend, value in registry_before.items()
    ):
        raise Lfm2RuntimeSemanticsReceiptError(
            "Independent LFM2 source authentication changed the attention registry."
        )
    registry_snapshot = _attention_registry_snapshot(modeling_utils, registry)

    live_sdpa = registry_before["sdpa"]
    sdpa_module_name = getattr(live_sdpa, "__module__", None)
    live_sdpa_module = (
        sys.modules.get(sdpa_module_name) if isinstance(sdpa_module_name, str) else None
    )
    if not isinstance(live_sdpa_module, types.ModuleType):
        raise Lfm2RuntimeSemanticsReceiptError(
            "Installed SDPA attention has no loaded defining module."
        )
    fresh_sdpa_module, sdpa_source_sha256 = load_fresh_source_module(
        live_sdpa_module,
        "transformers.integrations._neural_forge_capture_auth_sdpa_attention",
        "Transformers SDPA capture runtime",
    )
    canonical_sdpa = getattr(
        fresh_sdpa_module,
        getattr(live_sdpa, "__name__", ""),
        None,
    )
    if not callable(canonical_sdpa):
        raise Lfm2RuntimeSemanticsReceiptError(
            "Installed SDPA attention cannot be independently resolved."
        )

    fresh_torch_module, torch_module_sha256 = load_fresh_source_module(
        torch_module,
        "torch.nn.modules._neural_forge_capture_auth_module",
        "torch.nn.Module capture runtime",
    )
    fresh_torch_linear, torch_linear_sha256 = load_fresh_source_module(
        torch_linear,
        "torch.nn.modules._neural_forge_capture_auth_linear",
        "torch.nn.Linear capture runtime",
    )
    fresh_torch_conv, torch_conv_sha256 = load_fresh_source_module(
        torch_conv,
        "torch.nn.modules._neural_forge_capture_auth_conv",
        "torch.nn.Conv1d capture runtime",
    )
    fresh_torch_container, torch_container_sha256 = load_fresh_source_module(
        torch_container,
        "torch.nn.modules._neural_forge_capture_auth_container",
        "torch.nn.Sequential capture runtime",
    )
    fresh_torch_sparse, torch_sparse_sha256 = load_fresh_source_module(
        torch_sparse,
        "torch.nn.modules._neural_forge_capture_auth_sparse",
        "torch.nn.Embedding capture runtime",
    )

    native_method_pairs = {
        "Lfm2ForCausalLM.forward": (
            modeling_lfm2.Lfm2ForCausalLM.forward,
            fresh_lfm2.Lfm2ForCausalLM.forward,
        ),
        "Lfm2Model.forward": (
            modeling_lfm2.Lfm2Model.forward,
            fresh_lfm2.Lfm2Model.forward,
        ),
        "Lfm2DecoderLayer.forward": (
            modeling_lfm2.Lfm2DecoderLayer.forward,
            fresh_lfm2.Lfm2DecoderLayer.forward,
        ),
        "Lfm2MLP.forward": (
            modeling_lfm2.Lfm2MLP.forward,
            fresh_lfm2.Lfm2MLP.forward,
        ),
        "Lfm2ShortConv.forward": (
            modeling_lfm2.Lfm2ShortConv.forward,
            fresh_lfm2.Lfm2ShortConv.forward,
        ),
        "Lfm2ShortConv.slow_forward": (
            modeling_lfm2.Lfm2ShortConv.slow_forward,
            fresh_lfm2.Lfm2ShortConv.slow_forward,
        ),
        "Lfm2ShortConv.cuda_kernels_forward": (
            modeling_lfm2.Lfm2ShortConv.cuda_kernels_forward,
            fresh_lfm2.Lfm2ShortConv.cuda_kernels_forward,
        ),
        "Lfm2Attention.forward": (
            modeling_lfm2.Lfm2Attention.forward,
            fresh_lfm2.Lfm2Attention.forward,
        ),
        "Lfm2RMSNorm.forward": (
            modeling_lfm2.Lfm2RMSNorm.forward,
            fresh_lfm2.Lfm2RMSNorm.forward,
        ),
        "Lfm2RotaryEmbedding.forward": (
            modeling_lfm2.Lfm2RotaryEmbedding.forward,
            fresh_lfm2.Lfm2RotaryEmbedding.forward,
        ),
    }
    native_class_pairs = {
        "Lfm2ForCausalLM": (
            modeling_lfm2.Lfm2ForCausalLM,
            fresh_lfm2.Lfm2ForCausalLM,
        ),
        "Lfm2Model": (
            modeling_lfm2.Lfm2Model,
            fresh_lfm2.Lfm2Model,
        ),
        "Lfm2DecoderLayer": (
            modeling_lfm2.Lfm2DecoderLayer,
            fresh_lfm2.Lfm2DecoderLayer,
        ),
        "Lfm2MLP": (modeling_lfm2.Lfm2MLP, fresh_lfm2.Lfm2MLP),
        "Lfm2ShortConv": (
            modeling_lfm2.Lfm2ShortConv,
            fresh_lfm2.Lfm2ShortConv,
        ),
        "Lfm2Attention": (
            modeling_lfm2.Lfm2Attention,
            fresh_lfm2.Lfm2Attention,
        ),
        "Lfm2RMSNorm": (
            modeling_lfm2.Lfm2RMSNorm,
            fresh_lfm2.Lfm2RMSNorm,
        ),
        "Lfm2RotaryEmbedding": (
            modeling_lfm2.Lfm2RotaryEmbedding,
            fresh_lfm2.Lfm2RotaryEmbedding,
        ),
    }
    torch_method_pairs = {
        "torch.nn.Module.__call__": (
            torch.nn.Module.__call__,
            fresh_torch_module.Module.__call__,
        ),
        "torch.nn.Module._wrapped_call_impl": (
            torch.nn.Module._wrapped_call_impl,
            fresh_torch_module.Module._wrapped_call_impl,
        ),
        "torch.nn.Module._call_impl": (
            torch.nn.Module._call_impl,
            fresh_torch_module.Module._call_impl,
        ),
        "torch.nn.Module._slow_forward": (
            torch.nn.Module._slow_forward,
            fresh_torch_module.Module._slow_forward,
        ),
        "torch.nn.Linear.forward": (
            torch.nn.Linear.forward,
            fresh_torch_linear.Linear.forward,
        ),
        "torch.nn.Conv1d.forward": (
            torch.nn.Conv1d.forward,
            fresh_torch_conv.Conv1d.forward,
        ),
        "torch.nn.Conv1d._conv_forward": (
            torch.nn.Conv1d._conv_forward,
            fresh_torch_conv.Conv1d._conv_forward,
        ),
        "torch.nn.Sequential.forward": (
            torch.nn.Sequential.forward,
            fresh_torch_container.Sequential.forward,
        ),
        "torch.nn.Embedding.forward": (
            torch.nn.Embedding.forward,
            fresh_torch_sparse.Embedding.forward,
        ),
    }
    modeling_function_pairs = {
        "apply_rotary_pos_emb": (
            modeling_lfm2.apply_rotary_pos_emb,
            fresh_lfm2.apply_rotary_pos_emb,
        ),
        "rotate_half": (modeling_lfm2.rotate_half, fresh_lfm2.rotate_half),
        "eager_attention_forward": (
            modeling_lfm2.eager_attention_forward,
            fresh_lfm2.eager_attention_forward,
        ),
        "repeat_kv": (modeling_lfm2.repeat_kv, fresh_lfm2.repeat_kv),
        "apply_mask_to_padding_states": (
            modeling_lfm2.apply_mask_to_padding_states,
            fresh_lfm2.apply_mask_to_padding_states,
        ),
    }
    attention_pairs = {
        "eager": (registry_before["eager"], fresh_lfm2.eager_attention_forward),
        "sdpa": (registry_before["sdpa"], canonical_sdpa),
    }

    native_classes = {}
    for name, (live, canonical) in sorted(native_class_pairs.items()):
        expected = _class_record(
            canonical,
            name=name,
            module_override="transformers.models.lfm2.modeling_lfm2",
        )
        observed = _class_record(
            live,
            name=name,
        )
        if observed != expected:
            raise Lfm2RuntimeSemanticsReceiptError(
                f"Installed LFM2 class shell {name!r} differs from source."
            )
        native_classes[name] = expected

    torch_class_pairs = {
        "torch.nn.Module": (
            torch.nn.Module,
            fresh_torch_module.Module,
            "torch.nn.modules.module",
        ),
        "torch.nn.Linear": (
            torch.nn.Linear,
            fresh_torch_linear.Linear,
            "torch.nn.modules.linear",
        ),
        "torch.nn.Conv1d": (
            torch.nn.Conv1d,
            fresh_torch_conv.Conv1d,
            "torch.nn.modules.conv",
        ),
        "torch.nn.Sequential": (
            torch.nn.Sequential,
            fresh_torch_container.Sequential,
            "torch.nn.modules.container",
        ),
        "torch.nn.Embedding": (
            torch.nn.Embedding,
            fresh_torch_sparse.Embedding,
            "torch.nn.modules.sparse",
        ),
    }
    torch_classes = {}
    for name, (live, canonical, expected_module) in sorted(torch_class_pairs.items()):
        expected = _class_record(
            canonical,
            name=name,
            module_override=expected_module,
        )
        observed = _class_record(
            live,
            name=name,
        )
        if observed != expected:
            raise Lfm2RuntimeSemanticsReceiptError(
                f"Installed torch class shell {name!r} differs from source."
            )
        torch_classes[name] = expected

    semantic_global_names = (
        "is_torchdynamo_compiling",
        "is_fast_path_available",
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "create_causal_mask",
        "create_recurrent_attention_mask",
        "maybe_autocast",
        "DynamicCache",
    )
    semantic_globals = {
        name: _imported_value_record(getattr(fresh_lfm2, name), name)
        for name in semantic_global_names
    }
    live_semantic_globals = {
        name: _imported_value_record(getattr(modeling_lfm2, name), name)
        for name in semantic_global_names
    }
    if (
        live_semantic_globals != semantic_globals
        or modeling_lfm2.torch is not torch
        or modeling_lfm2.nn is not torch.nn
        or modeling_lfm2.F is not torch.nn.functional
        or fresh_lfm2.torch is not torch
        or fresh_lfm2.nn is not torch.nn
        or fresh_lfm2.F is not torch.nn.functional
    ):
        raise Lfm2RuntimeSemanticsReceiptError(
            "Installed LFM2 semantic globals differ from independent source."
        )

    silu_source = _compiled_source_function_record(
        torch.nn.functional,
        "silu",
        "torch.nn.functional.silu",
    )
    silu = torch.nn.functional.silu
    silu_code = getattr(silu, "__code__", None)
    silu_semantics = source_callable_semantics_record(
        silu,
        "torch.nn.functional.silu",
    )
    if (
        not isinstance(silu_code, types.CodeType)
        or canonical_sha256(_code_object_body(silu_code)) != silu_source["code_sha256"]
        or getattr(silu, "__module__", None) != "torch.nn.functional"
        or getattr(silu, "__qualname__", None) != "silu"
        or torch.nn.functional.linear is not torch._C._nn.linear
        or torch.nn.functional.conv1d is not torch.conv1d
    ):
        raise Lfm2RuntimeSemanticsReceiptError(
            "Installed torch functional runtime differs from canonical entrypoints."
        )

    return {
        "schema": LFM2_RUNTIME_SEMANTICS_RECEIPT_SCHEMA,
        "torch_distribution_version": importlib.metadata.version("torch"),
        "transformers_distribution_version": importlib.metadata.version("transformers"),
        "lfm2_source_sha256": lfm2_source_sha256,
        "transformers_cache_runtime": canonical_cache_runtime,
        "torch_source_sha256s": {
            "module": torch_module_sha256,
            "linear": torch_linear_sha256,
            "conv": torch_conv_sha256,
            "container": torch_container_sha256,
            "sparse": torch_sparse_sha256,
            "functional": silu_source["defining_source_sha256"],
        },
        "native_lfm2_methods": _authenticated_records(
            native_method_pairs,
            "LFM2",
        ),
        "native_lfm2_classes": native_classes,
        "native_lfm2_functions": _authenticated_records(
            modeling_function_pairs,
            "LFM2 function",
        ),
        "attention_interfaces": _authenticated_records(
            attention_pairs,
            "attention interface",
        ),
        "attention_registry": registry_snapshot,
        "attention_interface_source_sha256s": {
            "eager": lfm2_source_sha256,
            "sdpa": sdpa_source_sha256,
        },
        "semantic_globals": semantic_globals,
        "namespace_bindings": {
            "torch": "torch",
            "nn": "torch.nn",
            "F": "torch.nn.functional",
        },
        "torch_python_methods": _authenticated_records(
            torch_method_pairs,
            "torch",
        ),
        "torch_classes": torch_classes,
        "torch_functional_silu": {
            **silu_source,
            "semantics": silu_semantics,
        },
        "torch_functional_linear": runtime_callable_record(
            torch._C._nn.linear,
            "torch._C._nn.linear",
        ),
        "torch_functional_conv1d": runtime_callable_record(
            torch.conv1d,
            "torch.conv1d",
        ),
        "temporary_aliases_removed": True,
        "persistent_modules_imported": [],
        "attention_registry_preserved": True,
    }


def _ensure_canonical_receipt() -> None:
    global _CANONICAL_RECEIPT
    global _CANONICAL_RECEIPT_SHA256

    if _CANONICAL_RECEIPT:
        return
    with _RECEIPT_LOCK:
        if _CANONICAL_RECEIPT:
            return
        import torch
        from transformers.models.lfm2 import modeling_lfm2

        body = _canonical_receipt_body(torch, modeling_lfm2)
        _CANONICAL_RECEIPT = MappingProxyType(body)
        _CANONICAL_RECEIPT_SHA256 = canonical_sha256(body)


def _live_class_record(
    value: type[Any],
    name: str,
) -> dict[str, Any]:
    return _class_record(
        value,
        name=name,
    )


def verify_lfm2_runtime_semantics_receipt(
    *,
    expected_sha256: str | None = None,
) -> str:
    """Reauthenticate live native capture semantics against installed source."""

    _ensure_canonical_receipt()
    import torch
    from transformers import cache_utils, modeling_utils
    from transformers.models.lfm2 import modeling_lfm2
    from torch.nn.modules import container as torch_container
    from torch.nn.modules import conv as torch_conv
    from torch.nn.modules import linear as torch_linear
    from torch.nn.modules import module as torch_module
    from torch.nn.modules import sparse as torch_sparse

    receipt = _CANONICAL_RECEIPT
    claimed = _CANONICAL_RECEIPT_SHA256
    if (
        not receipt
        or canonical_sha256(dict(receipt)) != claimed
        or (
            expected_sha256 is not None
            and (not isinstance(expected_sha256, str) or expected_sha256 != claimed)
        )
    ):
        raise Lfm2RuntimeSemanticsReceiptError(
            "Native LFM2 runtime receipt is unavailable or inconsistent."
        )

    registry = modeling_lfm2.ALL_ATTENTION_FUNCTIONS
    if registry is not modeling_utils.ALL_ATTENTION_FUNCTIONS:
        raise Lfm2RuntimeSemanticsReceiptError(
            "Live LFM2 attention registry is not the canonical shared registry."
        )
    live_attention = {
        backend: registry.get_interface(
            backend,
            modeling_lfm2.eager_attention_forward,
        )
        for backend in ("eager", "sdpa")
    }
    live_registry_snapshot = _attention_registry_snapshot(modeling_utils, registry)
    sdpa_module_name = getattr(live_attention["sdpa"], "__module__", None)
    sdpa_module = (
        sys.modules.get(sdpa_module_name) if isinstance(sdpa_module_name, str) else None
    )
    if not isinstance(sdpa_module, types.ModuleType):
        raise Lfm2RuntimeSemanticsReceiptError(
            "Live SDPA attention lost its defining module."
        )

    source_modules = {
        "lfm2": modeling_lfm2,
        "sdpa": sdpa_module,
        "module": torch_module,
        "linear": torch_linear,
        "conv": torch_conv,
        "container": torch_container,
        "sparse": torch_sparse,
        "functional": torch.nn.functional,
    }
    source_sha256s = {}
    for name, module in source_modules.items():
        source_value = getattr(module, "__file__", None)
        if not isinstance(source_value, str):
            raise Lfm2RuntimeSemanticsReceiptError(
                f"Runtime module {name!r} lost its defining source."
            )
        source_path = Path(source_value).resolve()
        if not source_path.is_file() or source_path.is_symlink():
            raise Lfm2RuntimeSemanticsReceiptError(
                f"Runtime module {name!r} source is not regular."
            )
        source_sha256s[name] = hashlib.sha256(source_path.read_bytes()).hexdigest()

    native_methods = {
        "Lfm2ForCausalLM.forward": modeling_lfm2.Lfm2ForCausalLM.forward,
        "Lfm2Model.forward": modeling_lfm2.Lfm2Model.forward,
        "Lfm2DecoderLayer.forward": modeling_lfm2.Lfm2DecoderLayer.forward,
        "Lfm2MLP.forward": modeling_lfm2.Lfm2MLP.forward,
        "Lfm2ShortConv.forward": modeling_lfm2.Lfm2ShortConv.forward,
        "Lfm2ShortConv.slow_forward": modeling_lfm2.Lfm2ShortConv.slow_forward,
        "Lfm2ShortConv.cuda_kernels_forward": (
            modeling_lfm2.Lfm2ShortConv.cuda_kernels_forward
        ),
        "Lfm2Attention.forward": modeling_lfm2.Lfm2Attention.forward,
        "Lfm2RMSNorm.forward": modeling_lfm2.Lfm2RMSNorm.forward,
        "Lfm2RotaryEmbedding.forward": (modeling_lfm2.Lfm2RotaryEmbedding.forward),
    }
    native_functions = {
        "apply_rotary_pos_emb": modeling_lfm2.apply_rotary_pos_emb,
        "rotate_half": modeling_lfm2.rotate_half,
        "eager_attention_forward": modeling_lfm2.eager_attention_forward,
        "repeat_kv": modeling_lfm2.repeat_kv,
        "apply_mask_to_padding_states": modeling_lfm2.apply_mask_to_padding_states,
    }
    torch_methods = {
        "torch.nn.Module.__call__": torch.nn.Module.__call__,
        "torch.nn.Module._wrapped_call_impl": torch.nn.Module._wrapped_call_impl,
        "torch.nn.Module._call_impl": torch.nn.Module._call_impl,
        "torch.nn.Module._slow_forward": torch.nn.Module._slow_forward,
        "torch.nn.Linear.forward": torch.nn.Linear.forward,
        "torch.nn.Conv1d.forward": torch.nn.Conv1d.forward,
        "torch.nn.Conv1d._conv_forward": torch.nn.Conv1d._conv_forward,
        "torch.nn.Sequential.forward": torch.nn.Sequential.forward,
        "torch.nn.Embedding.forward": torch.nn.Embedding.forward,
    }

    def live_records(values: Mapping[str, Any], namespace: str) -> dict[str, Any]:
        return {
            name: source_callable_semantics_record(
                value,
                f"live.{namespace}.{name}",
            )
            for name, value in sorted(values.items())
        }

    native_classes = {
        name: _live_class_record(
            value,
            name,
        )
        for name, value in sorted(
            {
                "Lfm2Attention": modeling_lfm2.Lfm2Attention,
                "Lfm2DecoderLayer": modeling_lfm2.Lfm2DecoderLayer,
                "Lfm2ForCausalLM": modeling_lfm2.Lfm2ForCausalLM,
                "Lfm2MLP": modeling_lfm2.Lfm2MLP,
                "Lfm2Model": modeling_lfm2.Lfm2Model,
                "Lfm2RMSNorm": modeling_lfm2.Lfm2RMSNorm,
                "Lfm2RotaryEmbedding": modeling_lfm2.Lfm2RotaryEmbedding,
                "Lfm2ShortConv": modeling_lfm2.Lfm2ShortConv,
            }.items()
        )
    }
    torch_classes = {
        name: _live_class_record(value, name)
        for name, (value, _expected_module) in sorted(
            {
                "torch.nn.Conv1d": (
                    torch.nn.Conv1d,
                    "torch.nn.modules.conv",
                ),
                "torch.nn.Linear": (
                    torch.nn.Linear,
                    "torch.nn.modules.linear",
                ),
                "torch.nn.Embedding": (
                    torch.nn.Embedding,
                    "torch.nn.modules.sparse",
                ),
                "torch.nn.Module": (
                    torch.nn.Module,
                    "torch.nn.modules.module",
                ),
                "torch.nn.Sequential": (
                    torch.nn.Sequential,
                    "torch.nn.modules.container",
                ),
            }.items()
        )
    }
    semantic_globals = {
        name: _imported_value_record(getattr(modeling_lfm2, name), name)
        for name in (
            "is_torchdynamo_compiling",
            "is_fast_path_available",
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "create_causal_mask",
            "create_recurrent_attention_mask",
            "maybe_autocast",
            "DynamicCache",
        )
    }

    silu = torch.nn.functional.silu
    silu_code = getattr(silu, "__code__", None)
    silu_semantics = source_callable_semantics_record(
        silu,
        "live.torch.nn.functional.silu",
    )
    if (
        importlib.metadata.version("torch") != receipt.get("torch_distribution_version")
        or importlib.metadata.version("transformers")
        != receipt.get("transformers_distribution_version")
        or source_sha256s["lfm2"] != receipt.get("lfm2_source_sha256")
        or _cache_runtime_record(cache_utils)
        != receipt.get("transformers_cache_runtime")
        or {
            name: source_sha256s[name]
            for name in (
                "module",
                "linear",
                "conv",
                "container",
                "sparse",
                "functional",
            )
        }
        != receipt.get("torch_source_sha256s")
        or {
            "eager": source_sha256s["lfm2"],
            "sdpa": source_sha256s["sdpa"],
        }
        != receipt.get("attention_interface_source_sha256s")
        or live_records(native_methods, "LFM2") != receipt.get("native_lfm2_methods")
        or native_classes != receipt.get("native_lfm2_classes")
        or live_records(native_functions, "LFM2 function")
        != receipt.get("native_lfm2_functions")
        or live_records(live_attention, "attention interface")
        != receipt.get("attention_interfaces")
        or live_registry_snapshot != receipt.get("attention_registry")
        or semantic_globals != receipt.get("semantic_globals")
        or modeling_lfm2.torch is not torch
        or modeling_lfm2.nn is not torch.nn
        or modeling_lfm2.F is not torch.nn.functional
        or live_records(torch_methods, "torch") != receipt.get("torch_python_methods")
        or torch_classes != receipt.get("torch_classes")
        or not isinstance(silu_code, types.CodeType)
        or canonical_sha256(_code_object_body(silu_code))
        != receipt.get("torch_functional_silu", {}).get("code_sha256")
        or silu_semantics != receipt.get("torch_functional_silu", {}).get("semantics")
        or getattr(silu, "__module__", None) != "torch.nn.functional"
        or getattr(silu, "__qualname__", None) != "silu"
        or torch.nn.functional.linear is not torch._C._nn.linear
        or torch.nn.functional.conv1d is not torch.conv1d
        or runtime_callable_record(torch._C._nn.linear, "torch._C._nn.linear")
        != receipt.get("torch_functional_linear")
        or runtime_callable_record(torch.conv1d, "torch.conv1d")
        != receipt.get("torch_functional_conv1d")
    ):
        raise Lfm2RuntimeSemanticsReceiptError(
            "Live native LFM2 capture runtime differs from its source receipt."
        )
    return claimed


def verify_lfm2_capture_model_runtime(
    model: Any,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Authenticate the exact model instance call path used by trace capture."""

    receipt_sha256 = verify_lfm2_runtime_semantics_receipt(
        expected_sha256=expected_sha256
    )
    import torch
    from transformers.models.lfm2 import modeling_lfm2
    from torch.nn.modules import module as torch_module

    inner_model = getattr(model, "model", None)
    if (
        type(model) is not modeling_lfm2.Lfm2ForCausalLM
        or type(inner_model) is not modeling_lfm2.Lfm2Model
        or type(getattr(inner_model, "embed_tokens", None)) is not torch.nn.Embedding
        or type(getattr(model, "lm_head", None)) is not torch.nn.Linear
        or type(getattr(inner_model, "rotary_emb", None))
        is not modeling_lfm2.Lfm2RotaryEmbedding
    ):
        raise Lfm2RuntimeSemanticsReceiptError(
            "LFM2 trace capture requires the exact native root model call path."
        )

    expected_modules: dict[str, tuple[Any, type[Any]]] = {
        "": (model, modeling_lfm2.Lfm2ForCausalLM),
        "model": (inner_model, modeling_lfm2.Lfm2Model),
        "model.embed_tokens": (inner_model.embed_tokens, torch.nn.Embedding),
        "model.layers": (inner_model.layers, torch.nn.ModuleList),
        "model.rotary_emb": (
            inner_model.rotary_emb,
            modeling_lfm2.Lfm2RotaryEmbedding,
        ),
        "model.embedding_norm": (
            inner_model.embedding_norm,
            modeling_lfm2.Lfm2RMSNorm,
        ),
        "lm_head": (model.lm_head, torch.nn.Linear),
    }
    layers = tuple(inner_model.layers)
    if not layers:
        raise Lfm2RuntimeSemanticsReceiptError(
            "LFM2 trace capture model has no decoder layers."
        )
    for layer_index, layer in enumerate(layers):
        prefix = f"model.layers.{layer_index}"
        expected_modules[prefix] = (layer, modeling_lfm2.Lfm2DecoderLayer)
        expected_modules[f"{prefix}.operator_norm"] = (
            layer.operator_norm,
            modeling_lfm2.Lfm2RMSNorm,
        )
        expected_modules[f"{prefix}.ffn_norm"] = (
            layer.ffn_norm,
            modeling_lfm2.Lfm2RMSNorm,
        )
        feed_forward = layer.feed_forward
        expected_modules[f"{prefix}.feed_forward"] = (
            feed_forward,
            modeling_lfm2.Lfm2MLP,
        )
        if "act_fn" in vars(feed_forward):
            raise Lfm2RuntimeSemanticsReceiptError(
                f"LFM2 capture layer {layer_index} activation was replaced."
            )
        for projection_name in ("w1", "w3", "w2"):
            expected_modules[f"{prefix}.feed_forward.{projection_name}"] = (
                getattr(feed_forward, projection_name),
                torch.nn.Linear,
            )
        if bool(getattr(layer, "is_attention_layer", False)):
            operator = layer.self_attn
            expected_modules[f"{prefix}.self_attn"] = (
                operator,
                modeling_lfm2.Lfm2Attention,
            )
            for projection_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                expected_modules[f"{prefix}.self_attn.{projection_name}"] = (
                    getattr(operator, projection_name),
                    torch.nn.Linear,
                )
            for norm_name in ("q_layernorm", "k_layernorm"):
                expected_modules[f"{prefix}.self_attn.{norm_name}"] = (
                    getattr(operator, norm_name),
                    modeling_lfm2.Lfm2RMSNorm,
                )
        else:
            operator = layer.conv
            expected_modules[f"{prefix}.conv"] = (
                operator,
                modeling_lfm2.Lfm2ShortConv,
            )
            expected_modules[f"{prefix}.conv.conv"] = (
                operator.conv,
                torch.nn.Conv1d,
            )
            for projection_name in ("in_proj", "out_proj"):
                expected_modules[f"{prefix}.conv.{projection_name}"] = (
                    getattr(operator, projection_name),
                    torch.nn.Linear,
                )
    observed_modules = dict(model.named_modules())
    if set(observed_modules) != set(expected_modules) or any(
        observed_modules.get(name) is not module or type(module) is not expected_type
        for name, (module, expected_type) in expected_modules.items()
    ):
        raise Lfm2RuntimeSemanticsReceiptError(
            "LFM2 capture model contains foreign or substituted module classes."
        )

    global_hook_names = (
        "_global_forward_pre_hooks",
        "_global_forward_hooks",
        "_global_forward_hooks_always_called",
        "_global_forward_hooks_with_kwargs",
        "_global_forward_pre_hooks_with_kwargs",
    )
    if any(bool(getattr(torch_module, name, {})) for name in global_hook_names):
        raise Lfm2RuntimeSemanticsReceiptError(
            "LFM2 trace capture cannot authenticate global forward hooks."
        )

    override_names = (
        "forward",
        "__call__",
        "_wrapped_call_impl",
        "_call_impl",
    )
    hook_names = (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_pre_hooks",
        "_backward_hooks",
    )
    for name, module in observed_modules.items():
        label = name or "<root>"
        if any(method_name in vars(module) for method_name in override_names):
            raise Lfm2RuntimeSemanticsReceiptError(
                f"LFM2 capture module {label!r} carries an instance call override."
            )
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise Lfm2RuntimeSemanticsReceiptError(
                f"LFM2 capture module {label!r} carries a compiled call override."
            )
        if any(bool(getattr(module, hook_name, {})) for hook_name in hook_names):
            raise Lfm2RuntimeSemanticsReceiptError(
                f"LFM2 capture module {label!r} carries foreign execution hooks."
            )
    return receipt_sha256


def lfm2_runtime_semantics_receipt() -> dict[str, Any]:
    """Return a detached authenticated native capture receipt body."""

    verify_lfm2_runtime_semantics_receipt()
    return copy.deepcopy(dict(_CANONICAL_RECEIPT))


def lfm2_runtime_semantics_receipt_sha256() -> str:
    """Return the reauthenticated native capture receipt identity."""

    return verify_lfm2_runtime_semantics_receipt()


def lfm2_runtime_semantics_receipt_source_sha256() -> str:
    """Return the exact regular-file identity of this receipt verifier."""

    source_path = Path(__file__).resolve()
    if not source_path.is_file() or source_path.is_symlink():
        raise Lfm2RuntimeSemanticsReceiptError(
            "The native LFM2 receipt verifier source is not a regular file."
        )
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


__all__ = [
    "LFM2_RUNTIME_SEMANTICS_RECEIPT_SCHEMA",
    "Lfm2RuntimeSemanticsReceiptError",
    "lfm2_runtime_semantics_receipt",
    "lfm2_runtime_semantics_receipt_sha256",
    "lfm2_runtime_semantics_receipt_source_sha256",
    "load_fresh_source_module",
    "runtime_callable_record",
    "source_callable_semantics_record",
    "verify_lfm2_capture_model_runtime",
    "verify_lfm2_runtime_semantics_receipt",
]
