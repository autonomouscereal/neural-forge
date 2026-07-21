"""Fast authenticated foundation for the semantic LFM2 fold.

The historical CPCF-v5 run compiled every layer before the newer semantic
layer-stack compiler was invoked.  That work is redundant: the semantic stack
replaces all attention, short-convolution, and MLP producers.  This module
materializes only the pieces which must exist before that stack can run:

* the exact planner-owned dense zero target;
* the closed-form tied tokenizer/interface map;
* neutral RMSNorm coordinates; and
* a standalone dense checkpoint plus a sealed source/plan/evidence receipt.

It deliberately does not impersonate a CPCF-v5 compiler diagnostic.  The
receipt has its own schema and records every direct-local parent tensor, the
selective catalog and cache identities, the analytical interface lineage, and
the exact zero-operation audit.  No parent or child forward is performed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge import dynamic_lfm2_cpcf_resources as cpcf_resources
from neural_forge import hf_lfm2_cpcf_model_fold as cpcf_model_fold
from neural_forge import hf_lfm2_cpcf_v5_model_compiler as cpcf_v5_compiler
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_behavior_execution import observe_model_execution
from neural_forge import dynamic_fold_executor
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_cpcf_blank_target import (
    plan_and_synthesize_lfm2_cpcf_blank,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    install_lfm2_cpcf_neutral_norms,
    install_lfm2_cpcf_tied_interface,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (
    Lfm2CpcfModelFoldPolicy,
    independent_parameter_inventory,
)
from neural_forge.hf_lfm2_cpcf_selective_evidence import (
    open_lfm2_cpcf_selective_evidence_provider_from_catalog,
)
from neural_forge.hf_lfm2_cpcf_v5_quarantine import (
    LFM2_CPCF_V5_ARTIFACT_METADATA,
    Lfm2CpcfV5QuarantineSaver,
    Lfm2CpcfV5StrictLocalLoader,
)
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (
    Lfm2DenseQkvoSourceInventory,
    seal_lfm2_dense_qkvo_source_inventory,
)
from neural_forge.hf_tied_causal_interface_fold import TiedCausalInterfaceFold
from neural_forge.hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    fingerprint_hf_tokenizer_semantics,
)


LFM2_SEMANTIC_FOUNDATION_SCHEMA = "neural-forge-lfm2-semantic-foundation-v1"
LFM2_SEMANTIC_FOUNDATION_STATUS = (
    "standalone-semantic-foundation-saved-and-reloaded"
)
LFM2_SEMANTIC_FOUNDATION_INTERFACE_SCHEMA = (
    "neural-forge-lfm2-semantic-foundation-interface-v1"
)
LFM2_SEMANTIC_FOUNDATION_PARENT_SCHEMA = (
    "neural-forge-lfm2-semantic-foundation-parent-inventory-v1"
)
LFM2_SEMANTIC_FOUNDATION_LINEAGE_SCHEMA = (
    "neural-forge-lfm2-semantic-foundation-attention-lineage-v1"
)
SELECTIVE_CATALOG_PREPARATION_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-selective-catalog-preparation-v1"
)

EXACT_C61_TARGET_PARAMETERS = 57_405_368
EXACT_C61_BLANK_SEED = 0
EXACT_C61_TOPOLOGY = MappingProxyType(
    {
        "vocabulary_size": 4_096,
        "hidden_size": 776,
        "intermediate_size": 797,
        "layer_count": 14,
        "layer_types": [
            "short_conv",
            "short_conv",
            "full_attention",
            "short_conv",
            "full_attention",
            "short_conv",
            "full_attention",
            "short_conv",
            "full_attention",
            "short_conv",
            "full_attention",
            "short_conv",
            "full_attention",
            "short_conv",
        ],
        "query_heads": 10,
        "key_value_heads": 5,
        "head_dim": 64,
        "attention_layer_indices": [2, 4, 6, 8, 10, 12],
    }
)

_ZERO_FOLD_OPERATIONS = MappingProxyType(
    {
        "parent_forward_calls": 0,
        "parent_generate_calls": 0,
        "child_forward_calls": 0,
        "child_generate_calls": 0,
        "parent_cache_generation_calls": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "pruning_operations": 0,
        "quantization_operations": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "network_access_performed": False,
        "hub_loader_used": False,
    }
)


class Lfm2SemanticFoundationError(RuntimeError):
    """The fast semantic foundation failed an authenticated invariant."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2SemanticFoundationError(
            "Semantic-foundation metadata must be finite canonical JSON."
        ) from exc


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticFoundationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _sealed(body: Mapping[str, Any], field: str = "result_sha256") -> dict[str, Any]:
    plain = dict(body)
    if field in plain:
        raise Lfm2SemanticFoundationError(f"Seal field {field!r} already exists.")
    return {**plain, field: canonical_sha256(plain)}


def _verify_sealed(
    value: Any,
    *,
    field: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2SemanticFoundationError(f"{label} is not a mapping.")
    body = dict(value)
    claimed = _digest(body.pop(field, None), f"{label} {field}")
    if canonical_sha256(body) != claimed:
        raise Lfm2SemanticFoundationError(f"{label} seal is invalid.")
    return {**body, field: claimed}


def _ordinary_file(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise Lfm2SemanticFoundationError(f"{label} is missing or unsafe.")
    return path


def _ordinary_directory(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir() or path.is_symlink():
        raise Lfm2SemanticFoundationError(f"{label} is missing or unsafe.")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: Any) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("Semantic foundation requires PyTorch.") from exc
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _read_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SemanticFoundationError(f"Could not parse {label}.") from exc
    if not isinstance(value, dict) or raw != _canonical_json(value) + b"\n":
        raise Lfm2SemanticFoundationError(f"{label} is not canonical JSON.")
    return value, hashlib.sha256(raw).hexdigest()


def write_lfm2_semantic_foundation_receipt(
    path: str | os.PathLike[str],
    receipt: Mapping[str, Any],
) -> str:
    """Create one externally pinnable canonical receipt without replacement."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise Lfm2SemanticFoundationError(
            "Semantic-foundation receipt destination already exists."
        )
    raw = _canonical_json(dict(receipt)) + b"\n"
    try:
        with destination.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if destination.exists() and not destination.is_symlink():
            destination.unlink()
        raise
    return hashlib.sha256(raw).hexdigest()


def _local_safetensor_shards(directory: Path) -> tuple[Path, ...]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file() and not index_path.is_symlink():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Lfm2SemanticFoundationError(
                "Parent safetensors index is invalid."
            ) from exc
        weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise Lfm2SemanticFoundationError(
                "Parent safetensors index has no weight map."
            )
        shards = tuple(directory / name for name in sorted(set(weight_map.values())))
    else:
        single = directory / "model.safetensors"
        shards = (
            (single,)
            if single.is_file()
            else tuple(sorted(directory.glob("model-*.safetensors")))
        )
    if not shards or any(not path.is_file() or path.is_symlink() for path in shards):
        raise Lfm2SemanticFoundationError(
            "Parent direct-local safetensors are incomplete or unsafe."
        )
    return shards


def _tensor_receipt(value: Any) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": [int(size) for size in value.shape],
        "numel": int(value.numel()),
        "tensor_sha256": _tensor_sha256(value),
    }


def inventory_lfm2_semantic_foundation_parent(
    model_directory: str | os.PathLike[str],
    *,
    expected_parent_identity_sha256: str,
) -> Mapping[str, Any]:
    """Hash every direct-local tensor and expose typed attention/FFN subsets."""

    directory = _ordinary_directory(model_directory, "parent model directory")
    expected_parent = _digest(
        expected_parent_identity_sha256,
        "expected parent identity",
    )
    config_path = _ordinary_file(directory / "config.json", "parent config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SemanticFoundationError("Parent config.json is invalid.") from exc
    layer_types = config.get("layer_types") if isinstance(config, Mapping) else None
    if (
        not isinstance(config, Mapping)
        or config.get("model_type") != "lfm2"
        or not isinstance(layer_types, list)
        or not layer_types
    ):
        raise Lfm2SemanticFoundationError(
            "Direct-local parent is not one complete LFM2 checkpoint."
        )
    shards = _local_safetensor_shards(directory)
    shard_records = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "file_sha256": _file_sha256(path),
        }
        for path in shards
    ]
    index_path = directory / "model.safetensors.index.json"
    index_record = (
        None
        if not index_path.exists()
        else {
            "name": index_path.name,
            "size_bytes": index_path.stat().st_size,
            "file_sha256": _file_sha256(index_path),
        }
    )
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("Direct parent inventory requires safetensors.") from exc

    tensors: dict[str, Any] = {}
    tensor_receipts: dict[str, Any] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in tensor_receipts:
                    raise Lfm2SemanticFoundationError(
                        f"Parent tensor {key!r} appears in multiple shards."
                    )
                tensor = handle.get_tensor(key)
                tensor_receipts[key] = _tensor_receipt(tensor)
                if key.endswith(
                    (
                        ".q_proj.weight",
                        ".k_proj.weight",
                        ".v_proj.weight",
                        ".out_proj.weight",
                    )
                ):
                    tensors[key] = tensor.detach().to(device="cpu").contiguous().clone()

    attention: dict[str, Any] = {}
    ffn: dict[str, Any] = {}
    attention_indices = []
    for layer_index, kind in enumerate(layer_types):
        ffn_prefix = f"model.layers.{layer_index}.feed_forward."
        ffn_keys = {
            "ffn_norm": f"model.layers.{layer_index}.ffn_norm.weight",
            "w1": ffn_prefix + "w1.weight",
            "w2": ffn_prefix + "w2.weight",
            "w3": ffn_prefix + "w3.weight",
        }
        if any(key not in tensor_receipts for key in ffn_keys.values()):
            raise Lfm2SemanticFoundationError(
                f"Direct-local parent lacks complete FFN layer {layer_index}."
            )
        ffn[str(layer_index)] = {
            role: tensor_receipts[key] for role, key in ffn_keys.items()
        }
        if kind != "full_attention":
            continue
        attention_indices.append(layer_index)
        prefix = f"model.layers.{layer_index}.self_attn."
        keys = {
            "q": prefix + "q_proj.weight",
            "k": prefix + "k_proj.weight",
            "v": prefix + "v_proj.weight",
            "o": prefix + "out_proj.weight",
        }
        if any(key not in tensors for key in keys.values()):
            raise Lfm2SemanticFoundationError(
                f"Direct-local parent lacks complete Q/K/V/O layer {layer_index}."
            )
        inventory = seal_lfm2_dense_qkvo_source_inventory(
            source_query_weight=tensors[keys["q"]],
            source_key_weight=tensors[keys["k"]],
            source_value_weight=tensors[keys["v"]],
            source_output_weight=tensors[keys["o"]],
        )
        attention[str(layer_index)] = inventory.as_dict()

    checkpoint_manifest = {
        "schema": "neural-forge-direct-parent-checkpoint-manifest-v1",
        "parent_model_identity_sha256": expected_parent,
        "config_file_sha256": _file_sha256(config_path),
        "index": index_record,
        "shards": shard_records,
    }
    body = {
        "schema": LFM2_SEMANTIC_FOUNDATION_PARENT_SCHEMA,
        "parent_model_directory": str(directory),
        "expected_parent_identity_sha256": expected_parent,
        "checkpoint_manifest": checkpoint_manifest,
        "checkpoint_manifest_sha256": canonical_sha256(checkpoint_manifest),
        "tensor_key_inventory_sha256": canonical_sha256(sorted(tensor_receipts)),
        "full_tensor_inventory_sha256": canonical_sha256(tensor_receipts),
        "tensor_count": len(tensor_receipts),
        "tensor_receipts": tensor_receipts,
        "attention_layer_indices": attention_indices,
        "attention_qkvo_inventories": attention,
        "ffn_inventories": ffn,
        "all_checkpoint_tensors_hashed": True,
        "direct_local_safetensors": True,
        "parent_model_forward_calls": 0,
        "network_access_performed": False,
    }
    return MappingProxyType(_sealed(body))


def read_lfm2_semantic_foundation_catalog_preparation(
    receipt_json: str | os.PathLike[str],
    *,
    expected_receipt_sha256: str,
    cache: Lfm2CircuitTraceCache,
    catalog_store_directory: str | os.PathLike[str],
) -> Mapping[str, Any]:
    """Authenticate the sealed tensor-free catalog used by the interface solve."""

    path = _ordinary_file(receipt_json, "selective-catalog preparation receipt")
    receipt, _file_digest = _read_canonical_json(
        path,
        "selective-catalog preparation receipt",
    )
    verified = _verify_sealed(
        receipt,
        field="preparation_receipt_sha256",
        label="selective-catalog preparation receipt",
    )
    store = _ordinary_directory(catalog_store_directory, "selective-catalog store")
    if (
        verified["preparation_receipt_sha256"]
        != _digest(expected_receipt_sha256, "expected catalog receipt")
        or verified.get("schema") != SELECTIVE_CATALOG_PREPARATION_SCHEMA
        or verified.get("status") != "prepared_authenticated_tensor_free_catalog"
        or verified.get("trace_cache_directory")
        != str(Path(cache.path).expanduser().resolve())
        or verified.get("cache_key") != cache.cache_key
        or verified.get("catalog_store_directory") != str(store)
        or verified.get("catalog_receipt_json") != str(path)
        or verified.get("warm_hit_required_for_production_fold") is not True
        or verified.get("cold_rebuild_permitted_in_production_fold") is not False
        or verified.get("tensor_payloads_resident") is not False
    ):
        raise Lfm2SemanticFoundationError(
            "Selective-catalog preparation differs from its pinned cache/store."
        )
    provider = None
    try:
        provider = open_lfm2_cpcf_selective_evidence_provider_from_catalog(
            cache,
            artifact_store=store,
            expected_binding_sha256=_digest(
                verified.get("binding_sha256"),
                "catalog binding SHA-256",
            ),
            expected_manifest_sha256=_digest(
                verified.get("artifact_manifest_sha256"),
                "catalog manifest SHA-256",
            ),
            expected_catalog_sha256=_digest(
                verified.get("artifact_catalog_sha256"),
                "catalog artifact SHA-256",
            ),
        )
        catalog = provider.catalog
        if (
            catalog.catalog_sha256 != verified.get("selective_catalog_sha256")
            or catalog.source_sha256 != verified.get("selective_source_sha256")
            or catalog.provider_sha256 != verified.get("selective_provider_sha256")
        ):
            raise Lfm2SemanticFoundationError(
                "Warm-opened selective catalog differs from its preparation."
            )
    finally:
        if provider is not None:
            provider.close()
    return MappingProxyType(verified)


@dataclass(frozen=True, slots=True)
class Lfm2SemanticFoundationInterfaceSolve:
    interface: TiedCausalInterfaceFold
    report: Mapping[str, Any]


def fit_lfm2_semantic_foundation_interface(
    *,
    compiler_cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    target_tokenizer: Any,
    trajectory_corpus: str | os.PathLike[str],
    catalog_preparation: Mapping[str, Any],
    source_selected_tied_table: Any,
    target_model: Any,
) -> Lfm2SemanticFoundationInterfaceSolve:
    """Run only CPCF's authenticated final-norm tied-interface compiler."""

    preparation = _verify_sealed(
        catalog_preparation,
        field="preparation_receipt_sha256",
        label="selective-catalog preparation",
    )
    if (
        preparation.get("cache_key") != compiler_cache.cache_key
        or preparation.get("trace_cache_directory")
        != str(Path(compiler_cache.path).expanduser().resolve())
    ):
        raise Lfm2SemanticFoundationError(
            "Interface catalog and compiler cache identities diverged."
        )
    trajectory = _ordinary_file(trajectory_corpus, "trajectory corpus")
    rows = cpcf_resources._read_trajectory_rows(trajectory)
    trace_adapter = cpcf_resources._build_trace_adapter(
        rows=rows,
        target_tokenizer=target_tokenizer,
        bridge=bridge,
        cache=compiler_cache,
    )
    provider = None
    final_payload = None
    try:
        provider = open_lfm2_cpcf_selective_evidence_provider_from_catalog(
            compiler_cache,
            artifact_store=_ordinary_directory(
                preparation.get("catalog_store_directory"),
                "selective-catalog store",
            ),
            expected_binding_sha256=_digest(
                preparation.get("binding_sha256"),
                "catalog binding SHA-256",
            ),
            expected_manifest_sha256=_digest(
                preparation.get("artifact_manifest_sha256"),
                "catalog manifest SHA-256",
            ),
            expected_catalog_sha256=_digest(
                preparation.get("artifact_catalog_sha256"),
                "catalog artifact SHA-256",
            ),
        )
        catalog = provider.catalog
        if (
            catalog.catalog_sha256 != preparation.get("selective_catalog_sha256")
            or catalog.source_sha256 != preparation.get("selective_source_sha256")
            or catalog.provider_sha256 != preparation.get("selective_provider_sha256")
        ):
            raise Lfm2SemanticFoundationError(
                "Interface provider differs from the pinned selective catalog."
            )
        final_payload = provider.load_final_norm_payload()
        final_authentication = provider.authenticate_final_norm_payload(final_payload)
        shells = cpcf_v5_compiler.build_lfm2_cpcf_v5_target_phase_shells(
            trace_adapter,
            catalog,
            None,
            final_norm_payload=final_payload,
        )
        final_phases = cpcf_v5_compiler.materialize_lfm2_cpcf_v5_final_norm_phases(
            shells,
            catalog,
            final_payload,
        )
        boundary_compiler = cpcf_v5_compiler._SelectiveBoundaryCompiler(
            shells,
            catalog,
            source_selected_tied_table,
            target_hidden_width=int(target_model.config.hidden_size),
            model_policy=Lfm2CpcfModelFoldPolicy(),
        )
        interface = boundary_compiler.fit_interface(
            final_phases,
            target_norm_epsilon=cpcf_model_fold._final_norm_epsilon(target_model),
        )
    finally:
        if provider is not None and final_payload is not None:
            provider.release_final_norm_payload(final_payload)
        if provider is not None:
            provider.close()

    base = dict(boundary_compiler.base_interface.report)
    anchor = dict(boundary_compiler.anchor_fold.report)
    active = dict(interface.report)
    body = {
        "schema": LFM2_SEMANTIC_FOUNDATION_INTERFACE_SCHEMA,
        "compiler_cache_key": compiler_cache.cache_key,
        "cache_parent_fingerprint": compiler_cache.parent_fingerprint,
        "cache_manifest_sha256": compiler_cache.manifest["manifest_sha256"],
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "trajectory_corpus": str(trajectory),
        "trajectory_corpus_sha256": _file_sha256(trajectory),
        "trace_adapter_sha256": trace_adapter.evidence_sha256,
        "selective_catalog_preparation_receipt_sha256": preparation[
            "preparation_receipt_sha256"
        ],
        "selective_catalog_binding_sha256": preparation["binding_sha256"],
        "selective_catalog_manifest_sha256": preparation[
            "artifact_manifest_sha256"
        ],
        "selective_catalog_artifact_sha256": preparation[
            "artifact_catalog_sha256"
        ],
        "final_norm_authentication_sha256": final_authentication,
        "target_hidden_width": int(target_model.config.hidden_size),
        "target_norm_epsilon": cpcf_model_fold._final_norm_epsilon(target_model),
        "base_interface_report": base,
        "base_interface_report_sha256": base["report_sha256"],
        "causal_anchor_report": anchor,
        "causal_anchor_report_sha256": anchor["report_sha256"],
        "active_interface_report": active,
        "active_interface_report_sha256": active["report_sha256"],
        "source_tied_table_sha256": _tensor_sha256(source_selected_tied_table),
        "active_target_weight_sha256": _tensor_sha256(interface.target_weight),
        "producer_encoder_sha256": _tensor_sha256(interface.producer_encoder),
        "only_final_norm_payload_materialized": True,
        "legacy_layer_compiler_calls": 0,
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return Lfm2SemanticFoundationInterfaceSolve(
        interface=interface,
        report=MappingProxyType(_sealed(body)),
    )


def _balanced_partition(values: Sequence[int], count: int) -> tuple[tuple[int, ...], ...]:
    selected = tuple(int(value) for value in values)
    if count <= 0 or count > len(selected):
        raise Lfm2SemanticFoundationError(
            "A full-coverage head partition cannot create empty groups."
        )
    quotient, remainder = divmod(len(selected), count)
    cursor = 0
    result = []
    for index in range(count):
        size = quotient + int(index < remainder)
        result.append(selected[cursor : cursor + size])
        cursor += size
    if cursor != len(selected) or any(not group for group in result):
        raise AssertionError("Internal balanced partition failed.")
    return tuple(result)


def build_lfm2_semantic_foundation_head_groups(
    *,
    source_query_heads: int,
    source_key_value_heads: int,
    target_query_heads: int,
    target_key_value_heads: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...], Mapping[str, Any]]:
    """Build deterministic full-coverage groups while preserving GQA ownership."""

    values = (
        source_query_heads,
        source_key_value_heads,
        target_query_heads,
        target_key_value_heads,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise Lfm2SemanticFoundationError("Head counts must be positive integers.")
    if (
        source_query_heads % source_key_value_heads
        or target_query_heads % target_key_value_heads
        or target_key_value_heads > source_key_value_heads
    ):
        raise Lfm2SemanticFoundationError(
            "Source and target heads are not a compatible shrinking GQA topology."
        )
    source_q_per_kv = source_query_heads // source_key_value_heads
    target_q_per_kv = target_query_heads // target_key_value_heads
    kv_groups = _balanced_partition(range(source_key_value_heads), target_key_value_heads)
    query_groups = []
    for kv_group in kv_groups:
        owned_queries = tuple(
            source_kv * source_q_per_kv + offset
            for source_kv in kv_group
            for offset in range(source_q_per_kv)
        )
        query_groups.extend(_balanced_partition(owned_queries, target_q_per_kv))
    groups = tuple(query_groups)
    weights = tuple(tuple(1.0 / len(group) for _ in group) for group in groups)
    if (
        len(groups) != target_query_heads
        or sorted(member for group in groups for member in group)
        != list(range(source_query_heads))
    ):
        raise Lfm2SemanticFoundationError(
            "Topology-only GQA grouping did not preserve every source query head."
        )
    body = {
        "schema": "neural-forge-lfm2-semantic-foundation-head-groups-v1",
        "method": "balanced_contiguous_full_coverage_gqa_ownership",
        "source_query_heads": source_query_heads,
        "source_key_value_heads": source_key_value_heads,
        "target_query_heads": target_query_heads,
        "target_key_value_heads": target_key_value_heads,
        "source_queries_per_key_value": source_q_per_kv,
        "target_queries_per_key_value": target_q_per_kv,
        "key_value_groups": [list(group) for group in kv_groups],
        "query_groups": [list(group) for group in groups],
        "query_group_weights": [list(group) for group in weights],
        "full_source_query_coverage": True,
        "all_weights_positive": True,
        "numeric_parent_child_weight_match_objective": False,
    }
    return groups, weights, MappingProxyType(_sealed(body))


def build_lfm2_semantic_foundation_attention_lineage(
    *,
    parent_inventory: Mapping[str, Any],
    target_model: Any,
) -> Mapping[str, Any]:
    """Create the typed source/head lineage consumed by semantic attention."""

    parent = _verify_sealed(
        parent_inventory,
        field="result_sha256",
        label="semantic-foundation parent inventory",
    )
    attention = parent.get("attention_qkvo_inventories")
    if not isinstance(attention, Mapping):
        raise Lfm2SemanticFoundationError("Parent inventory lacks Q/K/V/O layers.")
    config = target_model.config
    target_q = int(config.num_attention_heads)
    target_kv = int(config.num_key_value_heads)
    target_head_dim = int(config.head_dim)
    lineages = {}
    for layer_text, raw_inventory in sorted(attention.items(), key=lambda item: int(item[0])):
        layer_index = int(layer_text)
        inventory = Lfm2DenseQkvoSourceInventory(
            tensor_receipts=raw_inventory.get("tensor_receipts", {}),
            source_state_sha256=raw_inventory.get("source_state_sha256"),
            structural_source_state_sha256=raw_inventory.get(
                "structural_source_state_sha256"
            ),
            logical_parameters=raw_inventory.get("logical_parameters"),
            inventory_sha256=raw_inventory.get("inventory_sha256", ""),
        )
        receipts = inventory.tensor_receipts
        q_shape = tuple(receipts["q_proj.weight"]["shape"])
        k_shape = tuple(receipts["k_proj.weight"]["shape"])
        v_shape = tuple(receipts["v_proj.weight"]["shape"])
        o_shape = tuple(receipts["out_proj.weight"]["shape"])
        source_hidden = int(q_shape[1])
        source_head_dim = target_head_dim
        source_q = int(q_shape[0]) // source_head_dim
        source_kv = int(k_shape[0]) // source_head_dim
        if (
            int(q_shape[0]) % source_head_dim
            or int(k_shape[0]) % source_head_dim
            or source_q <= 0
            or source_kv <= 0
            or int(k_shape[1]) != source_hidden
            or v_shape != k_shape
            or o_shape != (source_hidden, int(q_shape[0]))
        ):
            raise Lfm2SemanticFoundationError(
                f"Parent attention layer {layer_index} has incompatible projections."
            )
        groups, weights, group_report = build_lfm2_semantic_foundation_head_groups(
            source_query_heads=source_q,
            source_key_value_heads=source_kv,
            target_query_heads=target_q,
            target_key_value_heads=target_kv,
        )
        body = {
            "schema": LFM2_SEMANTIC_FOUNDATION_LINEAGE_SCHEMA,
            "layer_index": layer_index,
            "parent_inventory_sha256": parent["result_sha256"],
            "source_inventory": inventory.as_dict(),
            "source_inventory_sha256": inventory.inventory_sha256,
            "query_groups": [list(group) for group in groups],
            "query_group_weights": [list(group) for group in weights],
            "head_group_report": dict(group_report),
            "head_group_report_sha256": group_report["result_sha256"],
            "target_hidden_size": int(config.hidden_size),
            "target_query_heads": target_q,
            "target_key_value_heads": target_kv,
            "target_head_dim": target_head_dim,
            "numeric_parent_child_weight_match_objective": False,
            "legacy_layer_compiler_calls": 0,
        }
        lineages[str(layer_index)] = _sealed(body, "lineage_sha256")
    return MappingProxyType(lineages)


def _target_topology(model: Any) -> Mapping[str, Any]:
    config = model.config
    layer_types = tuple(str(value) for value in config.layer_types)
    return MappingProxyType(
        {
            "vocabulary_size": int(config.vocab_size),
            "hidden_size": int(config.hidden_size),
            "intermediate_size": int(config.intermediate_size),
            "layer_count": len(layer_types),
            "layer_types": list(layer_types),
            "query_heads": int(config.num_attention_heads),
            "key_value_heads": int(config.num_key_value_heads),
            "head_dim": int(config.head_dim),
            "attention_layer_indices": [
                index for index, kind in enumerate(layer_types) if kind == "full_attention"
            ],
        }
    )


def _mapped_parent_tied_table(parent_model: Any, bridge: StrictTokenizerBridgeArtifact) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Semantic foundation requires PyTorch.") from exc
    table = getattr(parent_model.get_input_embeddings(), "weight", None)
    output = getattr(parent_model.get_output_embeddings(), "weight", None)
    mapping = tuple(bridge.transport_profile.target_to_donor_row_indices)
    if (
        not isinstance(table, torch.nn.Parameter)
        or output is not table
        or table.ndim != 2
        or not mapping
        or max(mapping) >= int(table.shape[0])
        or int(table.shape[0]) != bridge.transport_profile.donor_row_capacity
    ):
        raise Lfm2SemanticFoundationError(
            "Parent tied table and tokenizer bridge are incompatible."
        )
    return (
        table.detach()
        .index_select(0, torch.tensor(mapping, device=table.device, dtype=torch.long))
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
        .clone()
    )


def _artifact_metadata(directory: Path) -> dict[str, Any]:
    path = _ordinary_file(
        directory / LFM2_CPCF_V5_ARTIFACT_METADATA,
        "standalone foundation artifact metadata",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SemanticFoundationError(
            "Standalone foundation artifact metadata is invalid."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2SemanticFoundationError(
            "Standalone foundation artifact metadata is not a mapping."
        )
    body = dict(value)
    claimed = _digest(body.pop("artifact_sha256", None), "artifact SHA-256")
    if canonical_sha256(body) != claimed:
        raise Lfm2SemanticFoundationError(
            "Standalone foundation artifact metadata seal is invalid."
        )
    return {**body, "artifact_sha256": claimed}


def materialize_lfm2_semantic_foundation(
    *,
    parent_model: Any,
    parent_model_directory: str | os.PathLike[str],
    parent_name: str,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    bridge: StrictTokenizerBridgeArtifact,
    compiler_cache: Lfm2CircuitTraceCache,
    trajectory_corpus: str | os.PathLike[str],
    catalog_preparation: Mapping[str, Any],
    output_checkpoint_directory: str | os.PathLike[str],
    expected_parent_identity_sha256: str,
    expected_target_plan_fingerprint: str,
    expected_selected_candidate_fingerprint: str,
    child_name: str = "lfm2-semantic-foundation-c61-4x",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
) -> Mapping[str, Any]:
    """Materialize, install, save, and reload the exact c61 semantic foundation."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Semantic foundation requires PyTorch.") from exc
    expected_parent = _digest(
        expected_parent_identity_sha256,
        "expected parent identity",
    )
    expected_plan = _digest(
        expected_target_plan_fingerprint,
        "expected target plan",
    )
    expected_candidate = _digest(
        expected_selected_candidate_fingerprint,
        "expected selected candidate",
    )
    if (
        compiler_cache.parent_fingerprint != expected_parent
        or bridge.donor_tokenizer_semantics_sha256
        != fingerprint_hf_tokenizer_semantics(donor_tokenizer)
        or bridge.target_tokenizer_semantics_sha256
        != fingerprint_hf_tokenizer_semantics(target_tokenizer)
    ):
        raise Lfm2SemanticFoundationError(
            "Parent cache or live tokenizer semantics differ from the bridge."
        )
    parent_before = fingerprint_lfm2_parent(parent_model).identity_sha256
    if parent_before != expected_parent:
        raise Lfm2SemanticFoundationError(
            "Direct-local parent model differs from the sealed cache identity."
        )
    parent_inventory = inventory_lfm2_semantic_foundation_parent(
        parent_model_directory,
        expected_parent_identity_sha256=expected_parent,
    )
    source_table = _mapped_parent_tied_table(parent_model, bridge)
    plan_guard = dynamic_fold_executor._FoldOnlyTorchGuard()
    with observe_model_execution(parent_model) as parent_execution, plan_guard:
        blank = plan_and_synthesize_lfm2_cpcf_blank(
            parent_model,
            parent_name=parent_name,
            parent_tokenizer_fingerprint=(
                bridge.donor_tokenizer_semantics_sha256
            ),
            evidence_fingerprint=compiler_cache.cache_key,
            child_name=child_name,
            compression_ratio="4",
            target_vocabulary_size=bridge.transport_profile.target_row_capacity,
            child_tokenizer_fingerprint=(
                bridge.target_tokenizer_semantics_sha256
            ),
            tokenizer_bridge_artifact=bridge,
            minimum_hidden_size=32,
            maximum_candidates=8,
            selected_candidate_rank=None,
            blank_seed=EXACT_C61_BLANK_SEED,
            device=torch.device(device),
            dtype=getattr(torch, dtype),
        )
    if any(parent_execution.values()):
        raise Lfm2SemanticFoundationError(
            "Blank planning or synthesis executed the parent model."
        )
    target = blank.model
    target.requires_grad_(False)
    target.eval()
    blank_report = blank.as_dict()
    blank_inventory = dict(independent_parameter_inventory(target))
    blank_state = fingerprint_model_state(target)
    topology = dict(_target_topology(target))
    exact_topology = dict(EXACT_C61_TOPOLOGY)
    observed_exact = {
        name: topology[name]
        for name in exact_topology
    }
    if (
        blank.plan.fingerprint != expected_plan
        or blank_report.get("selected_candidate_fingerprint") != expected_candidate
        or blank.unique_parameters != EXACT_C61_TARGET_PARAMETERS
        or int(blank_inventory.get("unique_parameters", -1))
        != EXACT_C61_TARGET_PARAMETERS
        or observed_exact != exact_topology
        or blank_report.get("all_parameters_zero") is not True
        or blank_report.get("blank_state_fingerprint") != blank_state
    ):
        raise Lfm2SemanticFoundationError(
            "Synthesized blank differs from the exact sealed c61 target."
        )
    install_guard = dynamic_fold_executor._FoldOnlyTorchGuard()
    with observe_model_execution(target) as child_execution, install_guard:
        interface_solve = fit_lfm2_semantic_foundation_interface(
            compiler_cache=compiler_cache,
            bridge=bridge,
            target_tokenizer=target_tokenizer,
            trajectory_corpus=trajectory_corpus,
            catalog_preparation=catalog_preparation,
            source_selected_tied_table=source_table,
            target_model=target,
        )
        interface_install = dict(
            install_lfm2_cpcf_tied_interface(target, interface_solve.interface)
        )
        neutral_norm_install = dict(install_lfm2_cpcf_neutral_norms(target))
    if any(child_execution.values()):
        raise Lfm2SemanticFoundationError(
            "Interface or neutral-norm folding executed the child model."
        )
    forbidden_counts = {
        "optimizer_constructions": (
            plan_guard.optimizer_constructions
            + install_guard.optimizer_constructions
        ),
        "optimizer_steps": plan_guard.optimizer_steps + install_guard.optimizer_steps,
        "backward_calls": plan_guard.backward_calls + install_guard.backward_calls,
        "autograd_grad_calls": (
            plan_guard.autograd_grad_calls + install_guard.autograd_grad_calls
        ),
        "pruning_operations": (
            plan_guard.pruning_operations + install_guard.pruning_operations
        ),
        "quantization_operations": (
            plan_guard.quantization_operations
            + install_guard.quantization_operations
        ),
    }
    if any(forbidden_counts.values()):
        raise Lfm2SemanticFoundationError(
            "Foundation attempted a forbidden training or compression operation."
        )
    guard_report = _sealed(
        {
            "schema": "neural-forge-lfm2-semantic-foundation-fold-only-guard-v1",
            "parent_execution_counts": dict(parent_execution),
            "child_execution_counts": dict(child_execution),
            **forbidden_counts,
            "all_forbidden_operation_counts_zero": True,
        }
    )
    installed_inventory = dict(independent_parameter_inventory(target))
    installed_state = fingerprint_model_state(target)
    attention_lineage = build_lfm2_semantic_foundation_attention_lineage(
        parent_inventory=parent_inventory,
        target_model=target,
    )
    if (
        installed_inventory != blank_inventory
        or installed_state == blank_state
        or interface_install.get("accepted") is not True
        or neutral_norm_install.get("accepted") is not True
        or set(map(int, attention_lineage))
        != set(EXACT_C61_TOPOLOGY["attention_layer_indices"])
        or fingerprint_lfm2_parent(parent_model).identity_sha256 != parent_before
    ):
        raise Lfm2SemanticFoundationError(
            "Foundation installation changed topology, missed a layer, or mutated parent."
        )
    context_body = {
        "schema": "neural-forge-lfm2-semantic-foundation-context-v1",
        "parent_identity_sha256": expected_parent,
        "parent_inventory_sha256": parent_inventory["result_sha256"],
        "compiler_cache_key": compiler_cache.cache_key,
        "cache_manifest_sha256": compiler_cache.manifest["manifest_sha256"],
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "catalog_preparation_receipt_sha256": catalog_preparation[
            "preparation_receipt_sha256"
        ],
        "target_plan_fingerprint": blank.plan.fingerprint,
        "selected_candidate_fingerprint": expected_candidate,
        "blank_report_sha256": blank_report["report_sha256"],
        "blank_state_sha256": blank_state,
        "interface_solve_sha256": interface_solve.report["result_sha256"],
        "interface_install_sha256": interface_install["report_sha256"],
        "neutral_norm_install_sha256": neutral_norm_install["report_sha256"],
        "attention_lineage_sha256": canonical_sha256(dict(attention_lineage)),
        "installed_model_state_sha256": installed_state,
        "parameter_inventory_sha256": installed_inventory["inventory_sha256"],
        "operation_audit": dict(_ZERO_FOLD_OPERATIONS),
        "fold_only_guard_sha256": guard_report["result_sha256"],
    }
    context_sha = canonical_sha256(context_body)
    saver = Lfm2CpcfV5QuarantineSaver(
        target_tokenizer=target_tokenizer,
        context_binding_sha256=context_sha,
        expected_ffn_width=EXACT_C61_TOPOLOGY["intermediate_size"],
        expected_model_state_sha256=installed_state,
        expected_parameter_inventory_sha256=installed_inventory["inventory_sha256"],
        expected_tokenizer_semantics_sha256=(
            bridge.target_tokenizer_semantics_sha256
        ),
    )
    loader = Lfm2CpcfV5StrictLocalLoader(
        context_binding_sha256=context_sha,
        device=device,
        dtype=dtype,
        expected_ffn_width=EXACT_C61_TOPOLOGY["intermediate_size"],
        expected_model_state_sha256=installed_state,
        expected_parameter_inventory_sha256=installed_inventory["inventory_sha256"],
        expected_tokenizer_semantics_sha256=(
            bridge.target_tokenizer_semantics_sha256
        ),
    )
    output = Path(output_checkpoint_directory).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise Lfm2SemanticFoundationError(
            "Foundation checkpoint destination must be new."
        )
    staging = Path(
        tempfile.mkdtemp(prefix=".semantic-foundation-", dir=str(output.parent))
    ).resolve()
    moved = False
    try:
        saver(target, staging)
        first_reload = loader(staging)
        if fingerprint_model_state(first_reload) != installed_state:
            raise Lfm2SemanticFoundationError(
                "Staged foundation reload changed model bytes."
            )
        os.replace(staging, output)
        moved = True
        second_reload = loader(output)
        if fingerprint_model_state(second_reload) != installed_state:
            raise Lfm2SemanticFoundationError(
                "Published foundation reload changed model bytes."
            )
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        if moved and output.exists() and not output.is_symlink():
            shutil.rmtree(output)
        raise
    metadata = _artifact_metadata(output)
    if (
        metadata.get("context_binding_sha256") != context_sha
        or metadata.get("model_state_sha256") != installed_state
        or metadata.get("parameter_inventory_sha256")
        != installed_inventory["inventory_sha256"]
        or metadata.get("tokenizer_semantics_sha256")
        != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2SemanticFoundationError(
            "Published foundation metadata differs from the installed child."
        )
    body = {
        "schema": LFM2_SEMANTIC_FOUNDATION_SCHEMA,
        "status": LFM2_SEMANTIC_FOUNDATION_STATUS,
        "parent_model_directory": str(
            _ordinary_directory(parent_model_directory, "parent model directory")
        ),
        "parent_identity_sha256": expected_parent,
        "parent_identity_verified_by_direct_local_model_load": True,
        "parent_source_inventory": dict(parent_inventory),
        "compiler_cache_directory": str(
            Path(compiler_cache.path).expanduser().resolve()
        ),
        "compiler_cache_key": compiler_cache.cache_key,
        "cache_parent_fingerprint": compiler_cache.parent_fingerprint,
        "cache_manifest_sha256": compiler_cache.manifest["manifest_sha256"],
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "trajectory_corpus": str(_ordinary_file(trajectory_corpus, "trajectory corpus")),
        "catalog_preparation": dict(catalog_preparation),
        "target_plan_fingerprint": blank.plan.fingerprint,
        "selected_candidate_fingerprint": expected_candidate,
        "blank_seed": EXACT_C61_BLANK_SEED,
        "blank_target": blank_report,
        "blank_model_state_sha256": blank_state,
        "target_topology": topology,
        "target_unique_parameters": EXACT_C61_TARGET_PARAMETERS,
        "blank_parameter_inventory": blank_inventory,
        "source_selected_tied_table_sha256": _tensor_sha256(source_table),
        "interface_solve": dict(interface_solve.report),
        "interface_install": interface_install,
        "neutral_norm_install": neutral_norm_install,
        "attention_lineage": dict(attention_lineage),
        "installed_model_state_sha256": installed_state,
        "installed_parameter_inventory": installed_inventory,
        "foundation_context": {**context_body, "context_sha256": context_sha},
        "fold_only_guard": guard_report,
        "standalone_artifact": {
            "checkpoint_directory": str(output),
            "artifact_metadata": metadata,
            "artifact_sha256": metadata["artifact_sha256"],
            "model_state_sha256": metadata["model_state_sha256"],
            "parameter_inventory_sha256": metadata[
                "parameter_inventory_sha256"
            ],
            "staged_reload_verified": True,
            "published_reload_verified": True,
        },
        "legacy_cpcf_layer_compiler_executed": False,
        "semantic_layer_stack_is_authoritative_next_stage": True,
        "operation_audit": {
            **dict(_ZERO_FOLD_OPERATIONS),
            "parent_model_objects_constructed": 1,
            "child_model_objects_constructed": 3,
        },
        "claims": {
            "exact_dense_zero_blank_materialized": True,
            "tied_tokenizer_interface_folded": True,
            "neutral_norm_coordinates_installed": True,
            "attention_conv_mlp_layers_folded": False,
            "standalone_child_saved_and_reloaded": True,
            "training_used": False,
            "pruning_used": False,
            "quantization_used": False,
            "parent_forward_used": False,
        },
    }
    return MappingProxyType(_sealed(body, "receipt_sha256"))


def read_lfm2_semantic_foundation_receipt(
    receipt_json: str | os.PathLike[str],
    *,
    expected_receipt_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    """Read an externally pinned foundation receipt without ancestry checks."""

    path = _ordinary_file(receipt_json, "semantic-foundation receipt")
    value, file_sha256 = _read_canonical_json(path, "semantic-foundation receipt")
    receipt = _verify_sealed(
        value,
        field="receipt_sha256",
        label="semantic-foundation receipt",
    )
    if (
        receipt["receipt_sha256"]
        != _digest(expected_receipt_sha256, "expected foundation receipt")
        or receipt.get("schema") != LFM2_SEMANTIC_FOUNDATION_SCHEMA
        or receipt.get("status") != LFM2_SEMANTIC_FOUNDATION_STATUS
    ):
        raise Lfm2SemanticFoundationError(
            "Semantic-foundation receipt differs from its external pin."
        )
    return MappingProxyType(receipt), file_sha256


def authenticate_lfm2_semantic_foundation_receipt(
    receipt_json: str | os.PathLike[str],
    *,
    expected_receipt_sha256: str,
    semantic_cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    input_metadata: Mapping[str, Any],
    parent_model_directory: str | os.PathLike[str],
) -> tuple[Mapping[str, Any], str]:
    """Recompute direct-local ancestry and bind a saved input child."""

    receipt, file_sha256 = read_lfm2_semantic_foundation_receipt(
        receipt_json,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    parent_inventory = _verify_sealed(
        receipt.get("parent_source_inventory"),
        field="result_sha256",
        label="foundation parent inventory",
    )
    observed_parent = inventory_lfm2_semantic_foundation_parent(
        parent_model_directory,
        expected_parent_identity_sha256=semantic_cache.parent_fingerprint,
    )
    artifact = receipt.get("standalone_artifact")
    context = receipt.get("foundation_context")
    topology = receipt.get("target_topology")
    operation = receipt.get("operation_audit")
    blank = _verify_sealed(
        receipt.get("blank_target"),
        field="report_sha256",
        label="foundation blank target",
    )
    interface_solve = _verify_sealed(
        receipt.get("interface_solve"),
        field="result_sha256",
        label="foundation interface solve",
    )
    interface_install = _verify_sealed(
        receipt.get("interface_install"),
        field="report_sha256",
        label="foundation interface installation",
    )
    norm_install = _verify_sealed(
        receipt.get("neutral_norm_install"),
        field="report_sha256",
        label="foundation neutral-norm installation",
    )
    guard = _verify_sealed(
        receipt.get("fold_only_guard"),
        field="result_sha256",
        label="foundation fold-only guard",
    )
    preparation = _verify_sealed(
        receipt.get("catalog_preparation"),
        field="preparation_receipt_sha256",
        label="foundation catalog preparation",
    )
    attention_lineage = receipt.get("attention_lineage")
    blank_inventory = receipt.get("blank_parameter_inventory")
    installed_inventory = receipt.get("installed_parameter_inventory")
    artifact_metadata = (
        artifact.get("artifact_metadata")
        if isinstance(artifact, Mapping)
        else None
    )
    artifact_metadata_verified = _verify_sealed(
        artifact_metadata,
        field="artifact_sha256",
        label="foundation artifact metadata",
    )
    if not isinstance(blank_inventory, Mapping) or not isinstance(
        installed_inventory, Mapping
    ):
        raise Lfm2SemanticFoundationError(
            "Foundation parameter inventories are missing."
        )
    blank_inventory_body = dict(blank_inventory)
    blank_inventory_sha = _digest(
        blank_inventory_body.pop("inventory_sha256", None),
        "foundation blank inventory",
    )
    installed_inventory_body = dict(installed_inventory)
    installed_inventory_sha = _digest(
        installed_inventory_body.pop("inventory_sha256", None),
        "foundation installed inventory",
    )
    if (
        dict(observed_parent) != parent_inventory
        or receipt.get("parent_model_directory")
        != str(_ordinary_directory(parent_model_directory, "parent model directory"))
        or receipt.get("parent_identity_sha256") != semantic_cache.parent_fingerprint
        or receipt.get("cache_parent_fingerprint") != semantic_cache.parent_fingerprint
        or receipt.get("bridge_artifact_sha256") != bridge.artifact_sha256
        or not isinstance(artifact, Mapping)
        or artifact.get("artifact_sha256") != input_metadata.get("artifact_sha256")
        or artifact.get("model_state_sha256") != input_metadata.get("model_state_sha256")
        or artifact.get("parameter_inventory_sha256")
        != input_metadata.get("parameter_inventory_sha256")
        or receipt.get("installed_model_state_sha256")
        != input_metadata.get("model_state_sha256")
        or not isinstance(context, Mapping)
        or context.get("context_sha256")
        != canonical_sha256(
            {key: value for key, value in context.items() if key != "context_sha256"}
        )
        or input_metadata.get("context_binding_sha256") != context.get("context_sha256")
        or context.get("parent_identity_sha256") != semantic_cache.parent_fingerprint
        or context.get("parent_inventory_sha256") != parent_inventory["result_sha256"]
        or context.get("compiler_cache_key") != receipt.get("compiler_cache_key")
        or context.get("cache_manifest_sha256")
        != receipt.get("cache_manifest_sha256")
        or context.get("bridge_artifact_sha256") != bridge.artifact_sha256
        or context.get("catalog_preparation_receipt_sha256")
        != preparation["preparation_receipt_sha256"]
        or context.get("target_plan_fingerprint")
        != receipt.get("target_plan_fingerprint")
        or context.get("selected_candidate_fingerprint")
        != receipt.get("selected_candidate_fingerprint")
        or context.get("blank_report_sha256") != blank["report_sha256"]
        or context.get("blank_state_sha256")
        != receipt.get("blank_model_state_sha256")
        or context.get("interface_solve_sha256")
        != interface_solve["result_sha256"]
        or context.get("interface_install_sha256")
        != interface_install["report_sha256"]
        or context.get("neutral_norm_install_sha256")
        != norm_install["report_sha256"]
        or not isinstance(attention_lineage, Mapping)
        or context.get("attention_lineage_sha256")
        != canonical_sha256(attention_lineage)
        or context.get("installed_model_state_sha256")
        != receipt.get("installed_model_state_sha256")
        or context.get("parameter_inventory_sha256") != installed_inventory_sha
        or context.get("fold_only_guard_sha256") != guard["result_sha256"]
        or context.get("operation_audit") != dict(_ZERO_FOLD_OPERATIONS)
        or not isinstance(topology, Mapping)
        or {name: topology.get(name) for name in EXACT_C61_TOPOLOGY}
        != dict(EXACT_C61_TOPOLOGY)
        or receipt.get("target_unique_parameters") != EXACT_C61_TARGET_PARAMETERS
        or blank.get("target_plan_fingerprint")
        != receipt.get("target_plan_fingerprint")
        or blank.get("selected_candidate_fingerprint")
        != receipt.get("selected_candidate_fingerprint")
        or blank.get("target_unique_parameters") != EXACT_C61_TARGET_PARAMETERS
        or blank.get("blank_state_fingerprint")
        != receipt.get("blank_model_state_sha256")
        or blank.get("all_parameters_zero") is not True
        or blank.get("dense_materialized_parameters") is not True
        or blank.get("training_used") is not False
        or blank.get("pruning_used") is not False
        or blank.get("quantization_used") is not False
        or blank_inventory_sha != canonical_sha256(blank_inventory_body)
        or installed_inventory_sha != canonical_sha256(installed_inventory_body)
        or blank_inventory.get("unique_parameters") != EXACT_C61_TARGET_PARAMETERS
        or installed_inventory.get("unique_parameters")
        != EXACT_C61_TARGET_PARAMETERS
        or blank_inventory != installed_inventory
        or interface_solve.get("legacy_layer_compiler_calls") != 0
        or interface_solve.get("training_steps") != 0
        or interface_install.get("accepted") is not True
        or norm_install.get("accepted") is not True
        or guard.get("all_forbidden_operation_counts_zero") is not True
        or any(guard.get(name) != 0 for name in (
            "optimizer_constructions",
            "optimizer_steps",
            "backward_calls",
            "autograd_grad_calls",
            "pruning_operations",
            "quantization_operations",
        ))
        or any(
            not isinstance(guard.get(name), Mapping)
            or any(value != 0 for value in guard[name].values())
            for name in ("parent_execution_counts", "child_execution_counts")
        )
        or receipt.get("blank_seed") != EXACT_C61_BLANK_SEED
        or receipt.get("legacy_cpcf_layer_compiler_executed") is not False
        or not isinstance(operation, Mapping)
        or any(operation.get(name) != value for name, value in _ZERO_FOLD_OPERATIONS.items())
        or artifact_metadata_verified.get("artifact_sha256")
        != artifact.get("artifact_sha256")
        or artifact_metadata_verified.get("model_state_sha256")
        != artifact.get("model_state_sha256")
        or artifact_metadata_verified.get("parameter_inventory_sha256")
        != artifact.get("parameter_inventory_sha256")
        or artifact_metadata_verified.get("context_binding_sha256")
        != context.get("context_sha256")
        or artifact.get("staged_reload_verified") is not True
        or artifact.get("published_reload_verified") is not True
    ):
        raise Lfm2SemanticFoundationError(
            "Foundation plan, source, child, or zero-operation ancestry diverged."
        )
    return receipt, file_sha256


def rehydrate_lfm2_semantic_foundation_interface(
    target_model: Any,
    source_selected_tied_table: Any,
    *,
    semantic_cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    target_tokenizer: Any,
    parent_model_directory: str | os.PathLike[str],
    input_metadata: Mapping[str, Any],
    foundation_receipt_json: str | os.PathLike[str],
    expected_foundation_receipt_sha256: str,
) -> tuple[TiedCausalInterfaceFold, Mapping[str, Any]]:
    """Recompute and authenticate the foundation's analytical producer map."""

    receipt, file_sha256 = authenticate_lfm2_semantic_foundation_receipt(
        foundation_receipt_json,
        expected_receipt_sha256=expected_foundation_receipt_sha256,
        semantic_cache=semantic_cache,
        bridge=bridge,
        input_metadata=input_metadata,
        parent_model_directory=parent_model_directory,
    )
    compiler_cache_directory = _ordinary_directory(
        receipt.get("compiler_cache_directory"),
        "foundation compiler cache",
    )
    from neural_forge.hf_lfm2_circuit_trace_cache import (
        open_lfm2_circuit_trace_cache,
    )

    compiler_cache = open_lfm2_circuit_trace_cache(
        compiler_cache_directory,
        verify=True,
    )
    if (
        compiler_cache.cache_key != receipt.get("compiler_cache_key")
        or compiler_cache.parent_fingerprint != semantic_cache.parent_fingerprint
    ):
        raise Lfm2SemanticFoundationError(
            "Foundation compiler cache differs from its receipt."
        )
    solve = fit_lfm2_semantic_foundation_interface(
        compiler_cache=compiler_cache,
        bridge=bridge,
        target_tokenizer=target_tokenizer,
        trajectory_corpus=receipt.get("trajectory_corpus"),
        catalog_preparation=receipt.get("catalog_preparation"),
        source_selected_tied_table=source_selected_tied_table,
        target_model=target_model,
    )
    expected_solve = _verify_sealed(
        receipt.get("interface_solve"),
        field="result_sha256",
        label="foundation interface solve",
    )
    target = target_model.get_input_embeddings().weight
    output = target_model.get_output_embeddings().weight
    expected_target = solve.interface.target_weight.to(
        device=target.device,
        dtype=target.dtype,
    )
    if (
        dict(solve.report) != expected_solve
        or output is not target
        or not bool((target == expected_target).all().item())
        or receipt.get("source_selected_tied_table_sha256")
        != _tensor_sha256(source_selected_tied_table)
    ):
        raise Lfm2SemanticFoundationError(
            "Recomputed foundation interface differs from saved child or source."
        )
    body = {
        "schema": "neural-forge-lfm2-semantic-foundation-interface-rehydration-v1",
        "foundation_receipt_sha256": receipt["receipt_sha256"],
        "foundation_receipt_file_sha256": file_sha256,
        "compiler_cache_key": compiler_cache.cache_key,
        "semantic_evidence_cache_key": semantic_cache.cache_key,
        "common_parent_fingerprint": semantic_cache.parent_fingerprint,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "interface_solve_sha256": solve.report["result_sha256"],
        "active_interface_report_sha256": solve.interface.report["report_sha256"],
        "saved_target_weight_sha256": _tensor_sha256(target),
        "producer_encoder_sha256": _tensor_sha256(solve.interface.producer_encoder),
        "compiler_and_semantic_cache_identities_bound_separately": True,
        "producer_encoder_recomputed_from_authenticated_final_norm_evidence": True,
        "legacy_cpcf_layer_compiler_calls": 0,
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "training_steps": 0,
    }
    return solve.interface, MappingProxyType(_sealed(body))


__all__ = [
    "EXACT_C61_BLANK_SEED",
    "EXACT_C61_TARGET_PARAMETERS",
    "EXACT_C61_TOPOLOGY",
    "LFM2_SEMANTIC_FOUNDATION_INTERFACE_SCHEMA",
    "LFM2_SEMANTIC_FOUNDATION_LINEAGE_SCHEMA",
    "LFM2_SEMANTIC_FOUNDATION_PARENT_SCHEMA",
    "LFM2_SEMANTIC_FOUNDATION_SCHEMA",
    "LFM2_SEMANTIC_FOUNDATION_STATUS",
    "Lfm2SemanticFoundationError",
    "Lfm2SemanticFoundationInterfaceSolve",
    "authenticate_lfm2_semantic_foundation_receipt",
    "build_lfm2_semantic_foundation_attention_lineage",
    "build_lfm2_semantic_foundation_head_groups",
    "fit_lfm2_semantic_foundation_interface",
    "inventory_lfm2_semantic_foundation_parent",
    "materialize_lfm2_semantic_foundation",
    "read_lfm2_semantic_foundation_catalog_preparation",
    "read_lfm2_semantic_foundation_receipt",
    "rehydrate_lfm2_semantic_foundation_interface",
    "write_lfm2_semantic_foundation_receipt",
]
