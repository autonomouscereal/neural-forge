"""Resident, local-only LFM2 backend for canonical-v2 isolation cells.

The parent is loaded directly from disk once per process and restored after
each cell.  Parent behavior and hidden-state evidence come from one sealed
control artifact; cells never regenerate parent traces.  Candidate evaluation
uses actual model forwards, including post-RoPE attention probabilities and KV
cache state when the focused component is attention.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .dynamic_behavior_execution import (
    BehaviorInferenceExecutionContract,
    execution_contract_satisfied,
    observe_model_execution,
)
from .dynamic_fold_phase_cache import (
    DynamicFoldPhaseCache,
    FoldPhaseCacheAlreadyCompleteError,
    FoldPhaseKey,
)
from .dynamic_fold_snapshot import DynamicFoldSnapshotConfig, capture_model_state
from .hf_lfm2_direct_local import (
    fingerprint_lfm2_direct_local_runtime_tensor_state,
    fingerprint_lfm2_direct_local_state,
    build_lfm2_direct_local_runtime_alias_manifest,
    load_lfm2_direct_local_model,
    load_lfm2_direct_local_tokenizer,
)
from .hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from .hf_lfm2_matrix_cell import MatrixCellRequest
from .matrix_cell_receipt import CellId
from .matrix_technique_registry import (
    CellTechniqueRequest,
    component_spec,
    resolve_level_width,
)

LOCAL_BACKEND_SCHEMA = "neural-forge-lfm2-matrix-local-isolation-backend-v1"
CONTROL_SCHEMA = "neural-forge-lfm2-matrix-copy-exact-control-v4"
_SUPPORTED_COMPONENTS = frozenset(
    {"C03", "C05", "C06", "C07", "C08", "C09", "C11", "C12", "C13"}
)
_RESIDENT_GUARD = threading.Lock()
_RESIDENTS: dict[tuple[str, str, str], "_ResidentModel"] = {}
_CONTROL_GUARD = threading.Lock()
_CONTROLS: dict[
    tuple[str, str, str], tuple[dict[str, Any], dict[str, torch.Tensor]]
] = {}
_EVIDENCE_GUARD = threading.Lock()
_EVIDENCE_MEMORY: dict[str, tuple[Any, Mapping[str, Any]]] = {}
_COMMON_SITE_ENDPOINTS = frozenset(
    {
        "residual.pre",
        "operator.input",
        "operator.output",
        "residual.post_operator",
        "mlp.input",
        "mlp.gate",
        "mlp.up",
        "mlp.post_swiglu",
        "mlp.down",
        "residual.post_layer",
    }
)
_ATTENTION_SITE_ENDPOINTS = frozenset(
    {
        "attention.q_linear",
        "attention.k_linear",
        "attention.v",
        "attention.context",
        "attention.o",
    }
)
_SHORT_CONV_SITE_ENDPOINTS = frozenset(
    {
        "short_conv.b",
        "short_conv.c",
        "short_conv.x",
        "short_conv.bx",
        "short_conv.context",
        "short_conv.o",
    }
)


class Lfm2MatrixLocalBackendError(RuntimeError):
    """Local model, evidence, control, or restoration violated its contract."""


def _runner_module() -> Any:
    import experiments.run_lfm2_matrix_cell as runner

    return runner


def _runner_sha256() -> str:
    runner = _runner_module()
    return hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest()


def _dtype(name: str) -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    try:
        return aliases[str(name).lower()]
    except KeyError as exc:
        raise Lfm2MatrixLocalBackendError(
            f"Unsupported runtime dtype {name!r}."
        ) from exc


def _canonical_dtype_name(name: str) -> str:
    return str(_dtype(name)).removeprefix("torch.")


def _canonical_device(device: str) -> str:
    try:
        parsed = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise Lfm2MatrixLocalBackendError(
            f"Unsupported runtime device {device!r}."
        ) from exc
    if parsed.type == "cuda" and parsed.index is None:
        parsed = torch.device("cuda", torch.cuda.current_device())
    return str(parsed)


def _runtime_device_receipt(device: str) -> Mapping[str, Any]:
    parsed = torch.device(_canonical_device(device))
    if parsed.type == "cuda":
        from .dynamic_replay_scheduler import capture_cuda_replay_device_receipt

        return capture_cuda_replay_device_receipt(int(parsed.index)).as_dict()
    return {
        "schema": "neural-forge-matrix-cpu-runtime-receipt-v1",
        "device": str(parsed),
        "torch_version": str(torch.__version__),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
    }


def _tokenizer_semantics_sha256(tokenizer: Any) -> str:
    from .hf_tokenizer_bridge import fingerprint_hf_tokenizer_semantics

    return fingerprint_hf_tokenizer_semantics(tokenizer)


def _implementation_sha256(value: Any) -> str:
    source = inspect.getsourcefile(type(value))
    if source is None:
        raise Lfm2MatrixLocalBackendError(
            f"{type(value).__qualname__} has no inspectable implementation."
        )
    return _file_sha256(Path(source).resolve())


def _runtime_state_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value[name]
        for name in (
            "state_sha256",
            "alias_abi_sha256",
            "unique_parameter_elements",
            "unique_buffer_elements",
            "named_tensor_occurrence_count",
            "non_persistent_buffers_included",
        )
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = (
        value.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    )
    body = (
        str(value.dtype).encode("ascii")
        + b"\0"
        + repr(tuple(int(item) for item in value.shape)).encode("ascii")
        + b"\0"
        + raw
    )
    return hashlib.sha256(body).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observe_top_level_model_execution(
    model: Any,
    callback: Callable[[], Any],
    *,
    label: str,
) -> tuple[Any, dict[str, Any]]:
    contract = BehaviorInferenceExecutionContract.top_level_forward()
    with observe_model_execution(model, contract=contract) as counts:
        result = callback()
    if not execution_contract_satisfied(counts):
        raise Lfm2MatrixLocalBackendError(
            f"{label} did not execute the authenticated top-level model forward."
        )
    receipt = {
        "label": label,
        "contract": contract.as_dict(),
        **{name: int(value) for name, value in counts.items()},
    }
    return result, receipt


def _atomic_json(path: Path, body: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(canonical_json_bytes(body))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class _LazyParentWeights(Mapping[str, torch.Tensor]):
    """Read-only CPU/FP64 parent values materialized only when a kernel asks."""

    def __init__(self, model: Any) -> None:
        self._state = model.state_dict()
        self._cache: dict[str, torch.Tensor] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._cache:
            self._cache[key] = (
                self._state[key]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .clone()
                .contiguous()
            )
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._state)

    def __len__(self) -> int:
        return len(self._state)


@dataclass(slots=True)
class _ResidentModel:
    model_directory: Path
    device: str
    dtype_name: str
    model: Any
    tokenizer: Any
    tokenizer_semantics_sha256: str
    runtime_device_receipt: Mapping[str, Any]
    parent_identity: Mapping[str, Any]
    runtime_state_fingerprint: Mapping[str, Any]
    model_implementation_sha256: str
    runtime_alias_manifest: Mapping[str, Any]
    parent_state_sha256: str
    parent_weights: _LazyParentWeights
    lock: threading.RLock
    clean: bool
    completed_cells: int


def _resident(
    model_directory: Path, device: str, dtype_name: str
) -> tuple[_ResidentModel, bool]:
    device = _canonical_device(device)
    dtype_name = _canonical_dtype_name(dtype_name)
    key = (str(model_directory.resolve()), device, dtype_name)
    with _RESIDENT_GUARD:
        found = _RESIDENTS.get(key)
        if found is not None:
            return found, True
        model = load_lfm2_direct_local_model(
            model_directory, device=device, dtype=_dtype(dtype_name)
        )
        model.eval()
        tokenizer = load_lfm2_direct_local_tokenizer(model_directory)
        state_sha256 = fingerprint_lfm2_direct_local_state(model)
        parent_identity = fingerprint_lfm2_parent(model).as_dict()
        runtime_state = fingerprint_lfm2_direct_local_runtime_tensor_state(model)
        created = _ResidentModel(
            model_directory=model_directory,
            device=device,
            dtype_name=dtype_name,
            model=model,
            tokenizer=tokenizer,
            tokenizer_semantics_sha256=_tokenizer_semantics_sha256(tokenizer),
            runtime_device_receipt=_runtime_device_receipt(device),
            parent_identity=parent_identity,
            runtime_state_fingerprint=runtime_state,
            model_implementation_sha256=_implementation_sha256(model),
            runtime_alias_manifest=build_lfm2_direct_local_runtime_alias_manifest(
                model
            ),
            parent_state_sha256=state_sha256,
            parent_weights=_LazyParentWeights(model),
            lock=threading.RLock(),
            clean=True,
            completed_cells=0,
        )
        _RESIDENTS[key] = created
        return created, False


def _runtime_versions(model: Any) -> dict[tuple[str, str], tuple[Any, ...]]:
    rows: dict[tuple[str, str], tuple[Any, ...]] = {}
    for kind, method_name in (
        ("parameter", "named_parameters"),
        ("buffer", "named_buffers"),
    ):
        method = getattr(model, method_name)
        try:
            values = method(remove_duplicate=False)
        except TypeError:
            values = method()
        for name, value in values:
            storage = value.untyped_storage()
            rows[(kind, str(name))] = (
                int(storage.data_ptr()),
                int(storage.nbytes()),
                int(value.storage_offset()),
                tuple(int(item) for item in value.shape),
                tuple(int(item) for item in value.stride()),
                str(value.dtype),
                str(value.device),
                int(getattr(value, "_version", 0)),
            )
    return rows


def _attention_layer_indices(model: Any) -> tuple[int, ...]:
    layer_types = tuple(getattr(getattr(model, "config", None), "layer_types", ()))
    indices = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if "attention" in str(layer_type).lower()
    )
    if not indices:
        raise Lfm2MatrixLocalBackendError(
            "LFM2 config exposes no physical attention-layer indices."
        )
    return indices


def _extract_cache(
    past: Any,
    attention_layer_indices: Sequence[int],
    physical_layer_count: int,
) -> tuple[tuple[int, torch.Tensor, torch.Tensor], ...]:
    if past is None:
        return ()
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    if isinstance(past, (tuple, list)):
        tensors = []
        for layer in past:
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                tensors.append((layer[0], layer[1]))
        if len(tensors) != len(attention_layer_indices):
            raise Lfm2MatrixLocalBackendError(
                "Legacy KV cache count differs from physical attention topology."
            )
        return tuple(
            (int(layer_index), key, value)
            for layer_index, (key, value) in zip(
                attention_layer_indices, tensors, strict=True
            )
        )
    layers = getattr(past, "layers", None)
    if isinstance(layers, Sequence):
        rows: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        compressed = len(layers) == len(attention_layer_indices)
        if not compressed and len(layers) != physical_layer_count:
            raise Lfm2MatrixLocalBackendError(
                "KV cache layer count differs from the physical LFM2 topology."
            )
        for stored_index, layer in enumerate(layers):
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
                physical_index = (
                    int(attention_layer_indices[stored_index])
                    if compressed
                    else stored_index
                )
                rows.append((physical_index, key, value))
        if tuple(index for index, _key, _value in rows) != tuple(
            attention_layer_indices
        ):
            raise Lfm2MatrixLocalBackendError(
                "KV cache entries do not match the physical attention-layer set."
            )
        return tuple(rows)
    return ()


@torch.no_grad()
def _capture_forced_trajectory(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    continuation_ids: Sequence[Sequence[int]],
    *,
    device: str,
) -> dict[str, torch.Tensor]:
    if len(prompts) != len(continuation_ids):
        raise Lfm2MatrixLocalBackendError("Prompt/control continuation counts differ.")
    attention_layer_indices = _attention_layer_indices(model)
    physical_layer_count = len(
        tuple(getattr(getattr(model, "config", None), "layer_types", ()))
    )
    buckets: dict[str, list[torch.Tensor]] = {}
    for prompt_index, (prompt, continuation) in enumerate(
        zip(prompts, continuation_ids, strict=True)
    ):
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        current = prompt_ids
        past = None
        for phase_index in range(len(continuation) + 1):
            result = model(
                input_ids=current,
                past_key_values=past,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=True,
                return_dict=True,
            )
            hidden = tuple(result.hidden_states or ())
            if not hidden:
                raise Lfm2MatrixLocalBackendError("Model returned no hidden states.")
            for layer_index, tensor in enumerate(hidden):
                buckets.setdefault(f"hidden|{layer_index:03d}", []).append(
                    tensor[0].detach().to(device="cpu", dtype=torch.float32)
                )
            attentions = tuple(result.attentions or ())
            if len(attentions) == len(attention_layer_indices):
                indexed_attentions = tuple(
                    zip(attention_layer_indices, attentions, strict=True)
                )
            elif len(attentions) == physical_layer_count:
                indexed_attentions = tuple(enumerate(attentions))
            else:
                raise Lfm2MatrixLocalBackendError(
                    "Attention result count differs from physical LFM2 topology."
                )
            observed_attention_indices = tuple(
                int(index)
                for index, tensor in indexed_attentions
                if isinstance(tensor, torch.Tensor)
            )
            if observed_attention_indices != attention_layer_indices:
                raise Lfm2MatrixLocalBackendError(
                    "Attention results do not match the physical attention-layer set."
                )
            for layer_index, tensor in indexed_attentions:
                if isinstance(tensor, torch.Tensor):
                    buckets.setdefault(
                        "attention|"
                        f"{prompt_index:03d}|{phase_index:03d}|{layer_index:03d}",
                        [],
                    ).append(tensor[0].detach().to(device="cpu", dtype=torch.float32))
            for layer_index, key, value in _extract_cache(
                result.past_key_values,
                attention_layer_indices,
                physical_layer_count,
            ):
                buckets.setdefault(
                    f"cache_key|{prompt_index:03d}|{phase_index:03d}|{layer_index:03d}",
                    [],
                ).append(key[0].detach().to(device="cpu", dtype=torch.float32))
                buckets.setdefault(
                    "cache_value|"
                    f"{prompt_index:03d}|{phase_index:03d}|{layer_index:03d}",
                    [],
                ).append(value[0].detach().to(device="cpu", dtype=torch.float32))
            past = result.past_key_values
            if phase_index < len(continuation):
                current = torch.tensor(
                    [[int(continuation[phase_index])]],
                    device=device,
                    dtype=prompt_ids.dtype,
                )
    result: dict[str, torch.Tensor] = {}
    for key, values in buckets.items():
        result[key] = (
            torch.cat(values, dim=0) if key.startswith("hidden|") else values[0]
        )
    return result


def _control_body(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "control_sha256"}


def _load_control(
    root: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], bool]:
    manifest_path = root / "control.json"
    payload_path = root / "control-states.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CONTROL_SCHEMA:
        raise Lfm2MatrixLocalBackendError("Control artifact schema is unsupported.")
    if canonical_sha256(_control_body(manifest)) != manifest.get("control_sha256"):
        raise Lfm2MatrixLocalBackendError("Control manifest authentication failed.")
    cache_key = (
        str(root.resolve()),
        str(manifest["control_sha256"]),
        str(manifest["payload_sha256"]),
    )
    with _CONTROL_GUARD:
        cached = _CONTROLS.get(cache_key)
    if cached is not None:
        return cached[0], cached[1], True
    if _file_sha256(payload_path) != manifest.get("payload_sha256"):
        raise Lfm2MatrixLocalBackendError(
            "Control state payload authentication failed."
        )
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in payload.items()
    ):
        raise Lfm2MatrixLocalBackendError("Control payload is not a tensor mapping.")
    tensor_manifest = {
        key: _tensor_sha256(value) for key, value in sorted(payload.items())
    }
    if canonical_sha256(tensor_manifest) != manifest.get("tensor_manifest_sha256"):
        raise Lfm2MatrixLocalBackendError("Control tensor manifest changed.")
    tensors = dict(payload)
    with _CONTROL_GUARD:
        _CONTROLS[cache_key] = (manifest, tensors)
    return manifest, tensors, False


def _rows_metric(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    runner = _runner_module()
    if candidate.shape != reference.shape:
        raise Lfm2MatrixLocalBackendError(
            f"Metric shapes differ: {tuple(candidate.shape)} != {tuple(reference.shape)}."
        )
    return runner._row_metrics(
        candidate.reshape(-1, int(candidate.shape[-1])).to(torch.float64),
        reference.reshape(-1, int(reference.shape[-1])).to(torch.float64),
    )


def _aggregate_named(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    prefix: str,
) -> dict[str, float]:
    names = sorted(name for name in reference if name.startswith(prefix))
    if not names or any(name not in candidate for name in names):
        raise Lfm2MatrixLocalBackendError(
            f"Actual-forward evidence is missing for prefix {prefix!r}."
        )
    left = torch.cat([candidate[name].reshape(-1) for name in names]).reshape(-1, 1)
    right = torch.cat([reference[name].reshape(-1) for name in names]).reshape(-1, 1)
    return _rows_metric(left, right)


def _aggregate_layer_named(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    prefix: str,
    layer_index: int,
) -> dict[str, float]:
    """Compare only the focused layer in prompt-indexed trajectory tensors."""

    suffix = f"|{int(layer_index):03d}"
    names = sorted(
        name for name in reference if name.startswith(prefix) and name.endswith(suffix)
    )
    if not names or any(name not in candidate for name in names):
        raise Lfm2MatrixLocalBackendError(
            f"Actual-forward evidence is missing for {prefix!r} at layer {layer_index}."
        )
    left = torch.cat([candidate[name].reshape(-1) for name in names]).reshape(-1, 1)
    right = torch.cat([reference[name].reshape(-1) for name in names]).reshape(-1, 1)
    return _rows_metric(left, right)


def _site_evidence_phase_key(
    request: MatrixCellRequest,
    *,
    layer_index: int,
    endpoints: Sequence[str],
    evidence_policy: Mapping[str, Any],
) -> tuple[FoldPhaseKey, str]:
    trace_root = Path(evidence_policy["parent_trace_cache"]).expanduser().resolve()
    trace_manifest = trace_root / "manifest.json"
    hmac_path = (
        Path(evidence_policy["basis_phase_hmac_key_file"]).expanduser().resolve()
    )
    trace_manifest_sha256 = _file_sha256(trace_manifest)
    key = FoldPhaseKey(
        phase_name="lfm2_matrix_site_evidence",
        dependency_sha256s={
            "trace_manifest": trace_manifest_sha256,
            "balanced_basis_phase": str(evidence_policy["basis_phase_key_sha256"]),
            "balanced_basis_hmac_key": _file_sha256(hmac_path),
            "runner": _runner_sha256(),
            "backend": _file_sha256(Path(__file__).resolve()),
        },
        numeric_contract={
            "flatten_dtype": "float64",
            "target_hidden_size": int(evidence_policy["target_hidden_size"]),
            "forced_decode_count": int(evidence_policy.get("forced_decode_count", 8)),
            "fit_batch_limit": evidence_policy.get("fit_batch_limit"),
            "heldout_batch_limit": evidence_policy.get("heldout_batch_limit"),
        },
        abi={
            "schema": "neural-forge-lfm2-matrix-site-evidence-cache-v1",
            "layer_index": int(layer_index),
            "endpoints": sorted(set(map(str, endpoints))),
            "trace_cache_path": str(trace_root),
            "basis_phase_cache_root": str(
                Path(evidence_policy["basis_phase_cache_root"]).expanduser().resolve()
            ),
            "basis_phase_hmac_key_id": str(evidence_policy["basis_phase_hmac_key_id"]),
            "request_evidence_sha256": str(request.evidence_binding["evidence_sha256"]),
        },
    )
    return key, trace_manifest_sha256


def _stable_site_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(name): item
        for name, item in value.items()
        if name not in {"evidence_cache_reused", "payload_sha256"}
    }


def _require_site_evidence_parent_identity(
    binding: Mapping[str, Any], parent_identity: Mapping[str, Any]
) -> str:
    expected = parent_identity.get("identity_sha256")
    observed = binding.get("parent_identity_sha256")
    if observed != expected:
        raise Lfm2MatrixLocalBackendError(
            "Site evidence belongs to a different parent identity."
        )
    return str(observed)


def _load_cached_site_evidence(
    request: MatrixCellRequest,
    *,
    layer_index: int,
    endpoints: Sequence[str],
    evidence_policy: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any], bool, bool, FoldPhaseKey, str, str]:
    key, trace_manifest_sha256 = _site_evidence_phase_key(
        request,
        layer_index=layer_index,
        endpoints=endpoints,
        evidence_policy=evidence_policy,
    )
    cache = DynamicFoldPhaseCache(
        Path(evidence_policy["cache_root"]).expanduser().resolve()
        / "physical-site-evidence-v1"
    )

    def load(cache_directory: Path) -> tuple[Any, Mapping[str, Any]]:
        return _runner_module().load_site_evidence(
            trace_cache_path=Path(evidence_policy["parent_trace_cache"]),
            layer_index=int(layer_index),
            endpoints=endpoints,
            forced_decode_count=int(evidence_policy.get("forced_decode_count", 8)),
            fit_batch_limit=evidence_policy.get("fit_batch_limit"),
            heldout_batch_limit=evidence_policy.get("heldout_batch_limit"),
            cache_directory=cache_directory,
            basis_phase_cache_root=Path(evidence_policy["basis_phase_cache_root"]),
            basis_phase_key_sha256=str(evidence_policy["basis_phase_key_sha256"]),
            basis_phase_hmac_key_file=Path(
                evidence_policy["basis_phase_hmac_key_file"]
            ),
            basis_phase_hmac_key_id=str(evidence_policy["basis_phase_hmac_key_id"]),
            evidence_split_manifest=None,
            target_hidden_size=int(evidence_policy["target_hidden_size"]),
        )

    def verify_entry(entry: Any, evidence: Any, binding: Mapping[str, Any]) -> None:
        expected_result = {
            "selection_sha256": evidence.selection_sha256,
            "binding_sha256": canonical_sha256(_stable_site_binding(binding)),
            "fit_rows": evidence.fit_row_count,
            "heldout_rows": evidence.heldout_row_count,
        }
        if dict(entry.result) != expected_result:
            raise Lfm2MatrixLocalBackendError(
                "Physical site-evidence completion disagrees with its payload."
            )

    entry = cache.lookup(key)
    if entry is not None:
        with _EVIDENCE_GUARD:
            cached = _EVIDENCE_MEMORY.get(entry.completion_sha256)
        if cached is None:
            evidence, binding = load(entry.payload_path)
            verify_entry(entry, evidence, binding)
            with _EVIDENCE_GUARD:
                _EVIDENCE_MEMORY[entry.completion_sha256] = (evidence, binding)
            memory_reused = False
        else:
            evidence, binding = cached
            memory_reused = True
        return (
            evidence,
            binding,
            True,
            memory_reused,
            key,
            entry.completion_sha256,
            trace_manifest_sha256,
        )
    try:
        with cache.begin(key) as transaction:
            evidence, binding = load(transaction.payload_path)
            entry = transaction.complete(
                result={
                    "selection_sha256": evidence.selection_sha256,
                    "binding_sha256": canonical_sha256(_stable_site_binding(binding)),
                    "fit_rows": evidence.fit_row_count,
                    "heldout_rows": evidence.heldout_row_count,
                }
            )
        verify_entry(entry, evidence, binding)
        with _EVIDENCE_GUARD:
            _EVIDENCE_MEMORY[entry.completion_sha256] = (evidence, binding)
        return (
            evidence,
            binding,
            False,
            False,
            key,
            entry.completion_sha256,
            trace_manifest_sha256,
        )
    except FoldPhaseCacheAlreadyCompleteError:
        entry = cache.lookup(key)
        if entry is None:
            raise Lfm2MatrixLocalBackendError(
                "Site evidence cache race produced no completed entry."
            )
        evidence, binding = load(entry.payload_path)
        verify_entry(entry, evidence, binding)
        with _EVIDENCE_GUARD:
            _EVIDENCE_MEMORY[entry.completion_sha256] = (evidence, binding)
        return (
            evidence,
            binding,
            True,
            False,
            key,
            entry.completion_sha256,
            trace_manifest_sha256,
        )


def _site_union_endpoints(
    parent_weights: Mapping[str, torch.Tensor], layer_index: int
) -> tuple[str, ...]:
    attention_key = f"model.layers.{int(layer_index)}.self_attn.q_proj.weight"
    operator = (
        _ATTENTION_SITE_ENDPOINTS
        if attention_key in parent_weights
        else _SHORT_CONV_SITE_ENDPOINTS
    )
    return tuple(sorted(_COMMON_SITE_ENDPOINTS | operator))


def build_lfm2_matrix_control_artifact(
    *,
    model_directory: Path,
    probe_prompts_path: Path,
    output_directory: Path,
    device: str,
    dtype_name: str,
    probe_tokens: int,
) -> Mapping[str, Any]:
    """Generate the one reusable parent control artifact for all slice cells."""

    model_directory = model_directory.expanduser().resolve()
    device = _canonical_device(device)
    dtype_name = _canonical_dtype_name(dtype_name)
    runner = _runner_module()
    prompts, prompt_sha256 = runner._load_prompt_set(
        probe_prompts_path, "free-run-probe"
    )
    resident, _reused = _resident(model_directory, device, dtype_name)
    probe_input_ids = [
        resident.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        for prompt in prompts
    ]
    with resident.lock:
        if not resident.clean or (
            fingerprint_lfm2_direct_local_state(resident.model)
            != resident.parent_state_sha256
        ):
            raise Lfm2MatrixLocalBackendError(
                "Resident parent is not at its base state."
            )
        def capture_control() -> tuple[list[list[int]], dict[str, torch.Tensor]]:
            continuation = runner._greedy_fixed(
                resident.model,
                resident.tokenizer,
                prompts,
                tokens=int(probe_tokens),
                device=device,
            )
            trajectory = _capture_forced_trajectory(
                resident.model,
                resident.tokenizer,
                prompts,
                continuation,
                device=device,
            )
            return continuation, trajectory

        (
            continuation_ids,
            payload,
        ), parent_control_execution = _observe_top_level_model_execution(
            resident.model,
            capture_control,
            label="matrix_parent_control_capture",
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    payload_path = output_directory / "control-states.pt"
    temporary = payload_path.with_name(payload_path.name + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, payload_path)
    tensor_manifest = {
        key: _tensor_sha256(value) for key, value in sorted(payload.items())
    }
    body = {
        "schema": CONTROL_SCHEMA,
        "parent_state_sha256": resident.parent_state_sha256,
        "parent_identity": resident.parent_identity,
        "runtime_state_identity": _runtime_state_identity(
            resident.runtime_state_fingerprint
        ),
        "runtime_alias_manifest_sha256": resident.runtime_alias_manifest[
            "manifest_sha256"
        ],
        "model_implementation_sha256": resident.model_implementation_sha256,
        "layer_types": list(
            getattr(getattr(resident.model, "config", None), "layer_types", ())
        ),
        "attention_implementation": str(
            getattr(
                getattr(resident.model, "config", None),
                "_attn_implementation",
                "unknown",
            )
        ),
        "model_directory": str(model_directory.resolve()),
        "runtime_device": device,
        "runtime_dtype": dtype_name,
        "runtime_device_receipt": resident.runtime_device_receipt,
        "tokenizer_semantics_sha256": resident.tokenizer_semantics_sha256,
        "probe_prompt_sha256": prompt_sha256,
        "probe_prompt_count": len(prompts),
        "probe_input_ids_sha256": canonical_sha256(probe_input_ids),
        "probe_tokens": int(probe_tokens),
        "control_token_ids": [list(map(int, row)) for row in continuation_ids],
        "payload_sha256": _file_sha256(payload_path),
        "tensor_manifest_sha256": canonical_sha256(tensor_manifest),
        "tensor_count": len(payload),
        "parent_forward_cache_generated_once": True,
        "parent_control_execution": parent_control_execution,
        "parent_trace_recapture_required_per_cell": False,
        "helper_runner_sha256": _runner_sha256(),
        "backend_source_sha256": _file_sha256(Path(__file__).resolve()),
    }
    manifest = {**body, "control_sha256": canonical_sha256(body)}
    _atomic_json(output_directory / "control.json", manifest)
    return manifest


class Lfm2LocalIsolationSession:
    def __init__(
        self,
        request: MatrixCellRequest,
        run_directory: Path,
        phase_key_sha256s: Mapping[str, str],
        emit: Callable[..., Mapping[str, Any]],
    ) -> None:
        if request.component_ids[0] not in _SUPPORTED_COMPONENTS:
            raise Lfm2MatrixLocalBackendError(
                f"Local backend has no actual-forward evaluator for {request.component_ids[0]}."
            )
        target = dict(request.execution_policy.get("target_policy", {}))
        evidence_policy = dict(request.execution_policy.get("evidence_policy", {}))
        behavior_policy = dict(request.execution_policy.get("behavior_policy", {}))
        self.request = request
        self.run_directory = run_directory
        self.phase_keys = dict(phase_key_sha256s)
        self._emit = emit
        self._kernel_log: list[str] = []
        self._execution_observations: list[dict[str, Any]] = []
        self.model_directory = Path(target["parent_model"]).expanduser().resolve()
        self.device = _canonical_device(str(target.get("device", "cuda:0")))
        self.dtype_name = _canonical_dtype_name(
            str(target.get("runtime_dtype", "float32"))
        )
        dependency = request.execution_policy.get("backend_dependency_sha256s", {})
        if dependency.get("runner_helpers") != _runner_sha256():
            raise Lfm2MatrixLocalBackendError(
                "Request does not bind the exact local helper implementation."
            )
        self.resident, resident_reused = _resident(
            self.model_directory, self.device, self.dtype_name
        )
        required_target_bindings = {
            "expected_parent_identity_sha256": self.resident.parent_identity[
                "identity_sha256"
            ],
            "expected_runtime_state_sha256": self.resident.runtime_state_fingerprint[
                "state_sha256"
            ],
            "expected_runtime_alias_manifest_sha256": (
                self.resident.runtime_alias_manifest["manifest_sha256"]
            ),
            "expected_tokenizer_semantics_sha256": (
                self.resident.tokenizer_semantics_sha256
            ),
        }
        changed_target_bindings = {
            name: target.get(name)
            for name, expected in required_target_bindings.items()
            if target.get(name) != expected
        }
        if changed_target_bindings:
            raise Lfm2MatrixLocalBackendError(
                "Request does not bind the exact local parent/tokenizer: "
                + ", ".join(sorted(changed_target_bindings))
            )
        self.resident.lock.acquire()
        self._closed = False
        self._resident_reused = resident_reused
        self._technique_result: Any | None = None
        self._restore_values: dict[str, torch.Tensor] = {}
        self._snapshot: Any | None = None
        self._pre_install_versions: dict[tuple[str, str], tuple[Any, ...]] | None = None
        self._pre_install_aliases: Mapping[str, Any] | None = None
        if not self.resident.clean:
            self.resident.lock.release()
            self._closed = True
            raise Lfm2MatrixLocalBackendError(
                "Resident model is quarantined after an unverified prior cell."
            )
        self.resident.clean = False
        self.control_root = (
            Path(behavior_policy["control_artifact"]).expanduser().absolute()
        )
        (
            self.control,
            self.control_tensors,
            control_memory_reused,
        ) = _load_control(self.control_root)
        expected_control_bindings = {
            "parent_state_sha256": self.resident.parent_state_sha256,
            "parent_identity": self.resident.parent_identity,
            "runtime_state_identity": _runtime_state_identity(
                self.resident.runtime_state_fingerprint
            ),
            "runtime_alias_manifest_sha256": self.resident.runtime_alias_manifest[
                "manifest_sha256"
            ],
            "model_implementation_sha256": (self.resident.model_implementation_sha256),
            "layer_types": list(
                getattr(
                    getattr(self.resident.model, "config", None),
                    "layer_types",
                    (),
                )
            ),
            "attention_implementation": str(
                getattr(
                    getattr(self.resident.model, "config", None),
                    "_attn_implementation",
                    "unknown",
                )
            ),
            "model_directory": str(self.model_directory),
            "runtime_device": self.device,
            "runtime_dtype": self.dtype_name,
            "runtime_device_receipt": self.resident.runtime_device_receipt,
            "tokenizer_semantics_sha256": (self.resident.tokenizer_semantics_sha256),
            "helper_runner_sha256": _runner_sha256(),
            "backend_source_sha256": _file_sha256(Path(__file__).resolve()),
        }
        changed_control_bindings = {
            name: self.control.get(name)
            for name, expected in expected_control_bindings.items()
            if self.control.get(name) != expected
        }
        if changed_control_bindings:
            raise Lfm2MatrixLocalBackendError(
                "Control artifact is not bound to the live parent/runtime: "
                + ", ".join(sorted(changed_control_bindings))
            )
        self.probe_prompts, probe_sha = _runner_module()._load_prompt_set(
            Path(behavior_policy["probe_prompts"]), "free-run-probe"
        )
        if probe_sha != self.control["probe_prompt_sha256"]:
            raise Lfm2MatrixLocalBackendError(
                "Control artifact uses another probe set."
            )
        if len(self.probe_prompts) != int(self.control["probe_prompt_count"]):
            raise Lfm2MatrixLocalBackendError(
                "Control prompt count differs from the live probe set."
            )
        live_probe_input_ids = [
            self.resident.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
            for prompt in self.probe_prompts
        ]
        if canonical_sha256(live_probe_input_ids) != self.control.get(
            "probe_input_ids_sha256"
        ):
            raise Lfm2MatrixLocalBackendError(
                "Control prompt tokenization differs from the live tokenizer."
            )
        requested_probe_tokens = int(
            behavior_policy.get("probe_tokens", self.control["probe_tokens"])
        )
        if requested_probe_tokens != int(self.control["probe_tokens"]):
            raise Lfm2MatrixLocalBackendError(
                "Control artifact uses another forced-trajectory length."
            )
        self.smoke_prompts, smoke_sha = _runner_module()._load_prompt_set(
            Path(behavior_policy["smoke_prompts"]), "decode-smoke"
        )
        self._smoke_sha256 = smoke_sha
        required_behavior_bindings = {
            "expected_control_sha256": self.control["control_sha256"],
            "expected_probe_prompt_sha256": probe_sha,
            "expected_smoke_prompt_sha256": smoke_sha,
        }
        changed_behavior_bindings = {
            name: behavior_policy.get(name)
            for name, expected in required_behavior_bindings.items()
            if behavior_policy.get(name) != expected
        }
        if changed_behavior_bindings:
            raise Lfm2MatrixLocalBackendError(
                "Request does not bind the exact control/probe/smoke inputs: "
                + ", ".join(sorted(changed_behavior_bindings))
            )
        self._control_ids = [
            list(map(int, row)) for row in self.control["control_token_ids"]
        ]
        if len(self._control_ids) != len(self.probe_prompts) or any(
            len(row) != requested_probe_tokens for row in self._control_ids
        ):
            raise Lfm2MatrixLocalBackendError(
                "Control continuation geometry does not match its manifest."
            )
        cell = CellId.parse(request.cell_id)
        spec = component_spec(cell, self.resident.parent_weights)
        x4_width = request.technique_configuration.get("x4_width")
        level_width = resolve_level_width(
            spec,
            request.level_id,
            x4_width=None if x4_width is None else int(x4_width),
        )
        endpoints = _site_union_endpoints(
            self.resident.parent_weights, int(cell.layer_index)
        )
        (
            self.site_evidence,
            binding,
            physical_evidence_reused,
            physical_evidence_memory_reused,
            physical_evidence_key,
            physical_evidence_completion_sha256,
            trace_manifest_sha256,
        ) = _load_cached_site_evidence(
            request,
            layer_index=int(cell.layer_index),
            endpoints=endpoints,
            evidence_policy=evidence_policy,
        )
        if binding.get("selection_sha256") != self.site_evidence.selection_sha256:
            raise Lfm2MatrixLocalBackendError(
                "Site evidence selection did not authenticate."
            )
        evidence_parent_identity_sha256 = _require_site_evidence_parent_identity(
            binding, self.resident.parent_identity
        )
        expected_evidence = request.evidence_binding["evidence_sha256"]
        if expected_evidence not in {
            self.site_evidence.selection_sha256,
            canonical_sha256(binding),
            canonical_sha256(_stable_site_binding(binding)),
            trace_manifest_sha256,
            physical_evidence_key.phase_key_sha256,
        }:
            raise Lfm2MatrixLocalBackendError(
                "Canonical evidence binding does not match the opened site evidence."
            )
        knobs = dict(request.technique_configuration)
        knobs.update({"device": self.device, "solve_dtype": "float64"})
        self.technique_request = CellTechniqueRequest(
            cell=cell,
            component=spec,
            level=request.level_id,
            level_width=level_width,
            seed=int(knobs.pop("seed", 0)),
            parent_weights=self.resident.parent_weights,
            evidence=self.site_evidence,
            knobs=knobs,
            workspace=run_directory,
            log=self._log_kernel_detail,
        )
        self.code_identity = {
            "schema": LOCAL_BACKEND_SCHEMA,
            "helper_runner_sha256": _runner_sha256(),
        }
        self.evidence = {
            "selection_sha256": self.site_evidence.selection_sha256,
            "site_binding_sha256": canonical_sha256(_stable_site_binding(binding)),
            "fit_rows": self.site_evidence.fit_row_count,
            "heldout_rows": self.site_evidence.heldout_row_count,
            "parent_state_sha256": self.resident.parent_state_sha256,
            "parent_identity_sha256": evidence_parent_identity_sha256,
            "parent_model_forward_calls": 0,
            "parent_trace_recaptured": False,
            "control_sha256": self.control["control_sha256"],
            "physical_evidence_phase_key_sha256": (
                physical_evidence_key.phase_key_sha256
            ),
            "physical_evidence_completion_sha256": (
                physical_evidence_completion_sha256
            ),
            "trace_manifest_sha256": trace_manifest_sha256,
        }
        self.numeric_execution = {
            "source_storage": {"dtype": self.dtype_name, "device": self.device},
            "source_forward": {"dtype": self.dtype_name, "device": self.device},
            "child_forward": {"dtype": self.dtype_name, "device": self.device},
            "accumulation": {"dtype": "float64", "device": "cpu"},
            "factorization": {"dtype": "float64", "device": "cpu"},
            "target_publication": {"dtype": self.dtype_name, "device": self.device},
            "runtime_verification": {"dtype": "float64", "device": "cpu"},
        }
        self.resources = {
            "resident_model_reused": resident_reused,
            "full_parent_cpu_copy_materialized": False,
            "lazy_parent_tensor_count": 0,
            "runtime_device": self.device,
            "control_memory_reused": control_memory_reused,
            "physical_evidence_memory_reused": (physical_evidence_memory_reused),
        }
        self.operation_audit = {
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "pruning_operations": 0,
            "quantization_operations": 0,
            "parent_trace_recaptures": 0,
            "network_requests": 0,
        }
        self.checkpoint_reuse = {
            "evidence_preparation": {
                "status": "hit" if physical_evidence_reused else "miss",
                "cache_key_sha256": phase_key_sha256s["evidence_preparation"],
                "reason": (
                    "canonical alias to authenticated physical site-evidence phase "
                    f"{physical_evidence_key.phase_key_sha256}"
                ),
            },
            "target_materialization": {
                "status": "hit" if resident_reused else "miss",
                "cache_key_sha256": phase_key_sha256s["target_materialization"],
                "reason": "process-resident local parent shell",
            },
            "technique_solve": {
                "status": "miss",
                "cache_key_sha256": phase_key_sha256s["technique_solve"],
                "reason": "kernel solve has no authenticated payload cache yet",
            },
            "component_install": {
                "status": "miss",
                "cache_key_sha256": phase_key_sha256s["component_install"],
                "reason": "component installation is attempt-local",
            },
            "behavior_evaluation": {
                "status": "miss",
                "cache_key_sha256": phase_key_sha256s["behavior_evaluation"],
                "reason": "candidate behavior is unique to the installed fold",
            },
        }
        self._capture: dict[str, torch.Tensor] | None = None
        self._local_metrics: dict[str, float] | None = None
        self._macro_metrics: dict[str, float] | None = None
        self._terminal_metrics: dict[str, float] | None = None

    def _log_kernel_detail(self, message: str) -> None:
        value = str(message)
        self._kernel_log.append(value)
        self._emit(
            "kernel_diagnostic",
            cell_id=self.request.cell_id,
            message=value,
            diagnostic_index=len(self._kernel_log) - 1,
        )

    def _observe_candidate_execution(
        self,
        label: str,
        callback: Callable[[], Any],
    ) -> Any:
        value, receipt = _observe_top_level_model_execution(
            self.resident.model,
            callback,
            label=label,
        )
        self._execution_observations.append(receipt)
        self.resources["candidate_execution_observations"] = list(
            self._execution_observations
        )
        self.operation_audit["candidate_model_forward_calls"] = sum(
            int(row["root_forward_calls"])
            for row in self._execution_observations
        )
        self._emit(
            "candidate_execution_observed",
            cell_id=self.request.cell_id,
            execution_receipt=receipt,
        )
        return value

    def materialize_copy_exact_control(
        self, phase_key_sha256: str
    ) -> Mapping[str, Any]:
        if phase_key_sha256 != self.phase_keys["target_materialization"]:
            raise Lfm2MatrixLocalBackendError("Control phase key changed.")
        return {
            "materialized": True,
            "authenticated": True,
            "control_sha256": self.control["control_sha256"],
            "parent_state_sha256": self.resident.parent_state_sha256,
            "parent_forwards_reused": True,
        }

    def _heldout_operator_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.technique_request.component
        if spec.input_endpoint is None:
            raise Lfm2MatrixLocalBackendError(
                f"{spec.component_id} has no held-out operator input."
            )
        inputs = self.site_evidence.heldout_rows(spec.input_endpoint)
        if spec.component_id == "C03":
            target = torch.cat(
                [
                    self.site_evidence.heldout_rows("short_conv.b"),
                    self.site_evidence.heldout_rows("short_conv.c"),
                    self.site_evidence.heldout_rows("short_conv.x"),
                ],
                dim=-1,
            )
        else:
            if spec.write_endpoint is None:
                raise Lfm2MatrixLocalBackendError(
                    f"{spec.component_id} has no held-out operator target."
                )
            target = self.site_evidence.heldout_rows(spec.write_endpoint)
        return inputs, target

    def calibrate(self, phase_key_sha256: str) -> Mapping[str, Any]:
        if phase_key_sha256 != self.phase_keys["evidence_preparation"]:
            raise Lfm2MatrixLocalBackendError("Calibration phase key changed.")
        spec = self.technique_request.component
        inputs, target = self._heldout_operator_pair()
        weight = self.resident.parent_weights[spec.weight_keys[0]]
        reproduced = inputs @ weight.T
        metrics = _rows_metric(reproduced, target)
        relative = float(metrics["aggregate_relative_l2"])
        tolerance = float(self.technique_request.knob("calibration_tolerance", 1e-3))
        return {
            "copy_exact_reproduction_rel_l2": relative,
            "copy_exact_reproduction_cosine": float(metrics["mean_cosine"]),
            "measurement_source": "sealed_heldout_operator",
            "passed": relative <= tolerance,
        }

    def install(self, result: Any, phase_key_sha256: str) -> Mapping[str, Any]:
        if phase_key_sha256 != self.phase_keys["component_install"]:
            raise Lfm2MatrixLocalBackendError("Install phase key changed.")
        expected = tuple(self.technique_request.component.weight_keys)
        before_versions = _runtime_versions(self.resident.model)
        before_aliases = build_lfm2_direct_local_runtime_alias_manifest(
            self.resident.model
        )
        if before_aliases != self.resident.runtime_alias_manifest:
            raise Lfm2MatrixLocalBackendError(
                "Resident alias topology differs from its startup manifest."
            )
        owned = set(expected)
        for pair in before_aliases["overlapping_byte_span_pairs"]:
            names = {str(pair["left"]["name"]), str(pair["right"]["name"])}
            if owned.intersection(names) and not names.issubset(owned):
                raise Lfm2MatrixLocalBackendError(
                    "Owned component aliases non-owned state; physical bundle is incomplete."
                )
        state = self.resident.model.state_dict()
        expected_installed_hashes: dict[str, str] = {}
        self._restore_values = {}
        for key in expected:
            target = state[key]
            value = result.installed_weights[key]
            if tuple(value.shape) != tuple(target.shape):
                raise Lfm2MatrixLocalBackendError(
                    f"Installed tensor {key} has the wrong shape."
                )
            if not bool(torch.isfinite(value).all().item()):
                raise Lfm2MatrixLocalBackendError(
                    f"Installed tensor {key} is non-finite."
                )
            original = self.resident.parent_weights[key]
            self._restore_values[key] = original
            expected_installed_hashes[key] = _tensor_sha256(
                value.to(device=target.device, dtype=target.dtype)
            )
        self._pre_install_versions = before_versions
        self._pre_install_aliases = before_aliases
        self._snapshot = capture_model_state(
            [(key, state[key]) for key in expected],
            DynamicFoldSnapshotConfig.memory(),
        )
        self._technique_result = result
        digests = _runner_module()._install_component(
            self.resident.model, result.installed_weights
        )
        after_versions = _runtime_versions(self.resident.model)
        changed_nonowned = sorted(
            identity
            for identity in before_versions
            if not (identity[0] == "parameter" and identity[1] in owned)
            and before_versions[identity] != after_versions[identity]
        )
        if changed_nonowned:
            raise Lfm2MatrixLocalBackendError(
                "Installation mutated non-owned state: "
                + ", ".join(f"{kind}:{name}" for kind, name in changed_nonowned)
            )
        installed_hashes = {
            key: _tensor_sha256(self.resident.model.state_dict()[key])
            for key in expected
        }
        for key in expected:
            if installed_hashes[key] != expected_installed_hashes[key]:
                raise Lfm2MatrixLocalBackendError(f"Installed tensor {key} changed.")
        self.resources["lazy_parent_tensor_count"] = len(
            self.resident.parent_weights._cache
        )
        installed_overlay_sha256 = canonical_sha256(
            {
                "parent_state_sha256": self.resident.parent_state_sha256,
                "installed_tensor_sha256s": installed_hashes,
                "owned_weight_keys": list(expected),
            }
        )
        return {
            "expected_owned_weight_keys": list(expected),
            "delivered_weight_keys": sorted(result.installed_weights),
            "installed_weight_keys": sorted(result.installed_weights),
            "unchanged_nonowned_state_verified": True,
            "alias_closure_verified": True,
            "installed_tensor_sha256s": installed_hashes,
            "install_digest_records": list(digests),
            "pre_state_sha256": self.resident.parent_state_sha256,
            "installed_overlay_sha256": installed_overlay_sha256,
            "state_version_entries_checked": len(before_versions),
            "nonowned_proof_method": (
                "exact-owned-writer-plus-version-storage-and-overlap-alias-proof"
            ),
        }

    def _candidate_capture(self) -> dict[str, torch.Tensor]:
        if self._capture is None:
            if self._technique_result is None:
                raise Lfm2MatrixLocalBackendError("Candidate was not installed.")
            self._capture = self._observe_candidate_execution(
                "matrix_candidate_forced_trajectory",
                lambda: _capture_forced_trajectory(
                    self.resident.model,
                    self.resident.tokenizer,
                    self.probe_prompts,
                    self._control_ids,
                    device=self.device,
                ),
            )
        return self._capture

    def _diagnostics(self) -> dict[str, Any]:
        candidate = self._candidate_capture()
        spec = self.technique_request.component
        result = self._technique_result
        hidden_key = f"hidden|{int(spec.layer_index) + 1:03d}"
        self._macro_metrics = _rows_metric(
            candidate[hidden_key], self.control_tensors[hidden_key]
        )
        inputs, target = self._heldout_operator_pair()
        predict = result.predict or (
            lambda rows: rows @ result.installed_weights[spec.weight_keys[0]].T
        )
        self._local_metrics = _rows_metric(predict(inputs), target)
        local_source = "sealed_heldout_operator"
        local_endpoint = (
            "short_conv.bcx" if spec.component_id == "C03" else str(spec.write_endpoint)
        )
        terminal_key = f"hidden|{max(int(key.split('|')[1]) for key in self.control_tensors if key.startswith('hidden|')):03d}"
        self._terminal_metrics = _rows_metric(
            candidate[terminal_key], self.control_tensors[terminal_key]
        )
        measured_write = {
            "measurement_status": "measured",
            "measurement_source": local_source,
            "measurement_endpoint": local_endpoint,
            "relative_l2": float(self._local_metrics["mean_relative_l2"]),
            "cosine": float(self._local_metrics["mean_cosine"]),
        }
        measured_macro = {
            "measurement_status": "measured",
            "measurement_source": "actual_model_forward",
            "relative_l2": float(self._macro_metrics["mean_relative_l2"]),
            "cosine": float(self._macro_metrics["mean_cosine"]),
        }
        measured_terminal = {
            "measurement_status": "measured",
            "measurement_source": "actual_model_forward",
            "relative_l2": float(self._terminal_metrics["mean_relative_l2"]),
            "cosine": float(self._terminal_metrics["mean_cosine"]),
        }
        groups: dict[str, Any] = {
            "weight": None,
            "component_write": measured_write,
            "macrocell": measured_macro,
            "terminal": measured_terminal,
            "attention_relation": None,
            "mlp_relation": None,
            "state_mixer_relation": None,
            "interface_relation": None,
            "cache_trajectory": None,
        }
        if spec.component_id in {"C06", "C07", "C08", "C09"}:
            if spec.layer_index is None:
                raise Lfm2MatrixLocalBackendError(
                    "Attention diagnostics require a concrete layer."
                )
            attention = _aggregate_layer_named(
                candidate,
                self.control_tensors,
                "attention|",
                spec.layer_index,
            )
            cache_key = _aggregate_layer_named(
                candidate,
                self.control_tensors,
                "cache_key|",
                spec.layer_index,
            )
            cache_value = _aggregate_layer_named(
                candidate,
                self.control_tensors,
                "cache_value|",
                spec.layer_index,
            )
            global_attention = _aggregate_named(
                candidate, self.control_tensors, "attention|"
            )
            global_cache_key = _aggregate_named(
                candidate, self.control_tensors, "cache_key|"
            )
            global_cache_value = _aggregate_named(
                candidate, self.control_tensors, "cache_value|"
            )
            groups["attention_relation"] = {
                "measurement_status": "measured",
                "measurement_source": "actual_model_forward",
                "post_rope_probability_relative_l2": float(
                    attention["aggregate_relative_l2"]
                ),
                "post_rope_probability_cosine": float(attention["mean_cosine"]),
                "global_post_rope_probability_relative_l2": float(
                    global_attention["aggregate_relative_l2"]
                ),
                "residual_relative_l2": float(self._macro_metrics["mean_relative_l2"]),
            }
            groups["cache_trajectory"] = {
                "measurement_status": "measured",
                "measurement_source": "actual_model_forward",
                "key_relative_l2": float(cache_key["aggregate_relative_l2"]),
                "key_cosine": float(cache_key["mean_cosine"]),
                "value_relative_l2": float(cache_value["aggregate_relative_l2"]),
                "value_cosine": float(cache_value["mean_cosine"]),
                "global_key_relative_l2": float(
                    global_cache_key["aggregate_relative_l2"]
                ),
                "global_value_relative_l2": float(
                    global_cache_value["aggregate_relative_l2"]
                ),
                "forced_control_continuation_tokens": int(self.control["probe_tokens"]),
            }
        elif spec.component_id in {"C11", "C12", "C13"}:
            groups["mlp_relation"] = dict(measured_macro)
        else:
            groups["state_mixer_relation"] = dict(measured_macro)
        physical = [spec.component_id]
        if self.request.technique_id == "T7" and spec.component_id in {
            "C06",
            "C07",
        }:
            physical = ["C06", "C07"]
        diagnostics = {
            "measurement_component_ids": [spec.component_id],
            "physical_component_ids": physical,
            "metric_groups": groups,
            "numeric_decision_authority": False,
        }
        self._emit(
            "actual_forward_diagnostics",
            cell_id=self.request.cell_id,
            component_diagnostics=diagnostics,
        )
        return diagnostics

    def evaluate_boundary(self) -> Mapping[str, Any]:
        diagnostics = self._diagnostics()
        return {
            "write_heldout_cos": float(self._local_metrics["mean_cosine"]),
            "write_heldout_rel_l2": float(self._local_metrics["mean_relative_l2"]),
            "post_operator_heldout_cos": float(self._macro_metrics["mean_cosine"]),
            "post_operator_heldout_rel_l2": float(
                self._macro_metrics["mean_relative_l2"]
            ),
            "component_diagnostics": diagnostics,
        }

    def evaluate_composed(self) -> Mapping[str, Any]:
        if self._terminal_metrics is None:
            self._diagnostics()
        relative = float(self._terminal_metrics["mean_relative_l2"])
        return {
            "terminal_heldout_rel_l2": relative,
            "vs_copy_exact_control_delta": relative,
            "terminal_heldout_cos": float(self._terminal_metrics["mean_cosine"]),
            "emit_zero_floor_passed": relative < 1.0,
        }

    def evaluate_free_run(self, phase_key_sha256: str) -> Mapping[str, Any]:
        if phase_key_sha256 != self.phase_keys["behavior_evaluation"]:
            raise Lfm2MatrixLocalBackendError("Behavior phase key changed.")
        candidate = self._observe_candidate_execution(
            "matrix_candidate_fixed_greedy",
            lambda: _runner_module()._greedy_fixed(
                self.resident.model,
                self.resident.tokenizer,
                self.probe_prompts,
                tokens=int(self.control["probe_tokens"]),
                device=self.device,
            ),
        )
        match, lcp = _runner_module()._token_match(candidate, self._control_ids)
        return {
            "probe_prompts": len(self.probe_prompts),
            "tokens_per_prompt": int(self.control["probe_tokens"]),
            "token_match_vs_control": float(match),
            "mean_lcp": float(lcp),
            "candidate_token_ids_sha256": canonical_sha256(candidate),
            "control_token_ids_sha256": canonical_sha256(self._control_ids),
        }

    def evaluate_decode_smoke(self, phase_key_sha256: str) -> list[Mapping[str, Any]]:
        if phase_key_sha256 != self.phase_keys["behavior_evaluation"]:
            raise Lfm2MatrixLocalBackendError("Behavior phase key changed.")
        rows = self._observe_candidate_execution(
            "matrix_candidate_decode_smoke",
            lambda: _runner_module()._decode_smoke(
                self.resident.model,
                self.resident.tokenizer,
                self.smoke_prompts,
                cap=int(
                    self.request.execution_policy.get("behavior_policy", {}).get(
                        "smoke_token_cap", 128
                    )
                ),
                device=self.device,
                eos_id=getattr(self.resident.tokenizer, "eos_token_id", None),
            ),
        )
        return [
            {
                "prompt_id": f"smoke-{index:02d}",
                "prompt_sha256": row["prompt_sha"],
                "termination_reason": row["terminated"],
                "output_verbatim": row["output_verbatim"],
                "generated_tokens": row["generated_tokens"],
                "safety_observation": None,
                "safety_scored_as_intelligence": False,
            }
            for index, row in enumerate(rows)
        ]

    def close(self) -> None:
        if self._closed:
            return
        verified = False
        try:
            if self._technique_result is not None:
                if self._snapshot is None:
                    raise Lfm2MatrixLocalBackendError(
                        "Installed candidate has no rollback snapshot."
                    )
                state = self.resident.model.state_dict()
                self._snapshot.restore(
                    {key: state[key] for key in self._snapshot.names}
                )
                for key, original in self._restore_values.items():
                    expected = original.to(
                        device=state[key].device, dtype=state[key].dtype
                    )
                    if _tensor_sha256(state[key]) != _tensor_sha256(expected):
                        raise Lfm2MatrixLocalBackendError(
                            f"Resident tensor {key} did not restore exactly."
                        )
            if self._pre_install_versions is not None:
                owned = set(self._restore_values)
                after = _runtime_versions(self.resident.model)
                changed_nonowned = sorted(
                    identity
                    for identity in self._pre_install_versions
                    if not (identity[0] == "parameter" and identity[1] in owned)
                    and self._pre_install_versions[identity] != after[identity]
                )
                if changed_nonowned:
                    raise Lfm2MatrixLocalBackendError(
                        "Non-owned state changed during the cell: "
                        + ", ".join(f"{kind}:{name}" for kind, name in changed_nonowned)
                    )
                if (
                    build_lfm2_direct_local_runtime_alias_manifest(self.resident.model)
                    != self._pre_install_aliases
                ):
                    raise Lfm2MatrixLocalBackendError(
                        "State alias topology changed during the cell."
                    )
            verified = True
        finally:
            if self._snapshot is not None:
                self._snapshot.cleanup()
                self._snapshot = None
            if verified:
                self.resident.clean = True
                self.resident.completed_cells += 1
            self._closed = True
            self._capture = None
            self.resident.lock.release()


def open_lfm2_local_isolation_session(
    request: MatrixCellRequest,
    run_directory: Path,
    phase_key_sha256s: Mapping[str, str],
    emit: Callable[..., Mapping[str, Any]],
) -> Lfm2LocalIsolationSession:
    """Canonical backend factory referenced by sealed cell requests."""

    session = Lfm2LocalIsolationSession.__new__(Lfm2LocalIsolationSession)
    try:
        session.__init__(request, run_directory, phase_key_sha256s, emit)
    except BaseException:
        if hasattr(session, "resident") and not getattr(session, "_closed", True):
            try:
                session.close()
            except BaseException:
                pass
        raise
    return session


__all__ = [
    "CONTROL_SCHEMA",
    "LOCAL_BACKEND_SCHEMA",
    "Lfm2LocalIsolationSession",
    "Lfm2MatrixLocalBackendError",
    "build_lfm2_matrix_control_artifact",
    "open_lfm2_local_isolation_session",
]
