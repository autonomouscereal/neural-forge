"""Non-promotable real H776 layer-2 residual-attention probe.

This module intentionally does not implement another model compiler.  It
opens one exact authenticated k=2 CPCF-v5 prefix, reconstructs its actual-child
causal phase bank, runs the ordinary layer compiler exactly once for the first
real attention layer, and stops before any k=3 publication.

The checkpoint was produced by an older immutable compiler runtime.  Normal
v5 resume must therefore continue rejecting it when the live implementation
fingerprint differs.  This probe uses the lower-level authenticated artifact,
boundary, and state-bank APIs and labels its result permanently
non-promotable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


LFM2_CPCF_V5_REAL_ATTENTION_PROBE_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-real-attention-layer-probe-v1"
)
LFM2_CPCF_V5_REAL_ATTENTION_RESOURCE_REPORT_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-real-attention-resource-report-v1"
)
LFM2_CPCF_V5_REAL_ATTENTION_CHECKPOINT_AUDIT_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-real-attention-k2-audit-v1"
)
LFM2_CPCF_V5_REAL_ATTENTION_PCG_AUDIT_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-real-attention-pcg-audit-v2"
)

H776_LAYER_INDEX = 2
H776_LAYER_COUNT = 14
H776_SOURCE_HIDDEN_SIZE = 1024
H776_TARGET_HIDDEN_SIZE = 776
H776_TARGET_INTERMEDIATE_SIZE = 797
H776_TARGET_QUERY_HEADS = 10
H776_TARGET_KEY_VALUE_HEADS = 5
H776_HEAD_DIM = 64
H776_TARGET_UNIQUE_PARAMETERS = 57_405_368
H776_PARENT_UNIQUE_PARAMETERS = 229_693_184
H776_PARAMETER_COMPRESSION_RATIO = (
    H776_PARENT_UNIQUE_PARAMETERS / H776_TARGET_UNIQUE_PARAMETERS
)
H776_LAYER_TYPES = (
    "conv",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
)
H776_OPERATOR_KINDS = tuple(
    "full_attention" if value == "full_attention" else "short_conv"
    for value in H776_LAYER_TYPES
)
H776_ATTENTION_CANDIDATE_IDS = (
    "direct_monolithic",
    "residual_attention_split",
    "fixed_qk_sequence_effective_vo_als",
)
H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID = H776_ATTENTION_CANDIDATE_IDS[-1]
H776_PHASE_COUNT = 2_304
H776_TRAJECTORY_COUNT = 256
H776_PHASES_PER_TRAJECTORY = 9
H776_SPLIT_PHASE_COUNTS = MappingProxyType(
    {"fit": 1_386, "selection": 468, "validation": 450}
)
H776_SPLIT_TRAJECTORY_COUNTS = MappingProxyType(
    {"fit": 154, "selection": 52, "validation": 50}
)
H776_FIT_QUERY_ROWS = 40_656
H776_VALIDATION_QUERY_ROWS = 13_199
H776_MAXIMUM_PROCESS_RSS_BYTES = 64 * 1024**3
H776_MAXIMUM_CUDA_BYTES_PER_DEVICE = 20 * 1024**3
H776_PCG_MAXIMUM_ITERATIONS = 64
H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL = 1e-8

H776_TARGET_PLAN_FINGERPRINT = (
    "3f27369567214c7f01d3f437e9c4ec50bce05c8345bbee4bf23570c204be6153"
)
H776_TARGET_TOPOLOGY_SHA256 = (
    "eb2d8124c66eb3d928c64567dc5de68193c727fbb5edd031bdb498552fdd0e54"
)
H776_SELECTIVE_CATALOG_SHA256 = (
    "0d12eab6f5641f24ed2d8173adb69fd727b772cd5ac5db47b3747017cfbf2694"
)
H776_SELECTIVE_SOURCE_SHA256 = (
    "298aca003880a465dcc412b3d7dd80ee9e3b5c8d291484f670f18ffe76a21eda"
)
H776_SELECTIVE_PROVIDER_SHA256 = (
    "cf690ab6cccbf1c44d3f8e40c576589ba020f53a4ffe2c4e1d93455bcbcf8582"
)
H776_CACHE_SHA256 = (
    "76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95"
)
H776_TOKEN_BRIDGE_SHA256 = (
    "5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384"
)
H776_TOKENIZER_SHA256 = (
    "5e76c6c0f9180ac69fae4729ecf2239100d7409fd148be14d80574340160675b"
)
H776_SOURCE_TIED_TABLE_SHA256 = (
    "d3ddf11aec3de8ed385080058f3b6677a18d6cc8524fdc3eaa6fcfe52e801356"
)
H776_SOURCE_STATE_SHA256 = (
    "7a4bc29edb9a4dfca9bdfaa507a36b2bc4c0cd81e2145d483221656686166665"
)
H776_SOURCE_INVENTORY_SHA256 = (
    "bd60d29cc6dd79b0eceaf1a3e626912e199355f7ff18024e16c3b2cd48913311"
)
H776_BLANK_REPORT_SHA256 = (
    "f6e958771a87270b2d728ad9ee3f19f4a8e6fc2da11b616fd1a0e05257f44b96"
)
H776_BLANK_STATE_SHA256 = (
    "8e83843b34cde2381fd4ed561411c3ef17eedf9460895873a21a7fb9d09f4caa"
)
H776_FULL_POLICY_SHA256 = (
    "7e0ece20c9aae06db5636b9c053089081786f0485e92dc8e61169e76553563af"
)
H776_ORIGIN_COMPILER_SHA256 = (
    "79a10961eee4b51ebf35b66a71f7295b0c4311d35895002964723dae3ed3f28e"
)
H776_SEQUENCE_SOLVER_PLAN_SHA256 = (
    "f26bd9d409e8cee9bbc11612ac89e7d509bc28d0aa3654f3fe60199b28c23f82"
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_EXPECTED_PHASE_NAMES = (
    "prefill",
    "decode.000",
    "decode.001",
    "decode.002",
    "decode.003",
    "decode.004",
    "decode.005",
    "decode.006",
    "decode.007",
)
_ZERO_EXECUTION_FIELDS = (
    "parent_model_forward_calls",
    "target_model_forward_calls",
    "target_layer_forward_calls",
    "generation_calls",
    "training_steps",
    "backward_calls",
    "optimizer_steps",
    "parameters_pruned",
    "parameters_quantized",
    "autograd_graphs",
)
_STRUCTURED_VALUE_PCG_ROLE = re.compile(
    r"sequence_effective_structured_value_iteration_([12])_group_([0-4])\Z"
)


class Lfm2CpcfV5RealAttentionProbeError(RuntimeError):
    """The real one-layer probe failed an authenticated stop condition."""


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
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Probe metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _plain(child) for name, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_plain(child) for child in value]
    return value


def _self_sealed(
    value: Mapping[str, Any],
    *,
    seal_field: str,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5RealAttentionProbeError(f"{name} must be a mapping.")
    result = _plain(value)
    body = dict(result)
    claimed = _digest(body.pop(seal_field, None), f"{name}.{seal_field}")
    if _sha256(body) != claimed:
        raise Lfm2CpcfV5RealAttentionProbeError(f"{name} is not self-sealed.")
    return result


def _local_directory(value: Path, name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be pathlib.Path.")
    raw = os.fspath(value.expanduser())
    if not raw or "://" in raw or raw.startswith(("\\\\", "//")):
        raise Lfm2CpcfV5RealAttentionProbeError(
            f"{name} must be an explicit local path."
        )
    try:
        result = value.expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfV5RealAttentionProbeError(
            f"{name} does not exist."
        ) from exc
    if result.is_symlink() or not result.is_dir():
        raise Lfm2CpcfV5RealAttentionProbeError(
            f"{name} must be a nonsymlink directory."
        )
    return result


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5RealAttentionProbePins:
    """Every out-of-band identity needed to run the H776 k=2 probe."""

    checkpoint_store_directory: Path
    checkpoint_key: str
    checkpoint_manifest_sha256: str
    expected_trace_sha256: str
    expected_runtime_compiler_implementation_sha256: str
    expected_sequence_source_sha256: str
    expected_target_plan_fingerprint: str = H776_TARGET_PLAN_FINGERPRINT
    expected_target_topology_sha256: str = H776_TARGET_TOPOLOGY_SHA256
    expected_selective_catalog_sha256: str = H776_SELECTIVE_CATALOG_SHA256
    expected_selective_source_sha256: str = H776_SELECTIVE_SOURCE_SHA256
    expected_selective_provider_sha256: str = H776_SELECTIVE_PROVIDER_SHA256
    expected_cache_sha256: str = H776_CACHE_SHA256
    expected_token_bridge_sha256: str = H776_TOKEN_BRIDGE_SHA256
    expected_tokenizer_sha256: str = H776_TOKENIZER_SHA256
    expected_source_tied_table_sha256: str = H776_SOURCE_TIED_TABLE_SHA256
    expected_source_state_sha256: str = H776_SOURCE_STATE_SHA256
    expected_source_inventory_sha256: str = H776_SOURCE_INVENTORY_SHA256
    expected_blank_report_sha256: str = H776_BLANK_REPORT_SHA256
    expected_blank_state_sha256: str = H776_BLANK_STATE_SHA256
    expected_full_policy_sha256: str = H776_FULL_POLICY_SHA256
    expected_origin_compiler_sha256: str = H776_ORIGIN_COMPILER_SHA256
    expected_sequence_solver_plan_sha256: str = H776_SEQUENCE_SOLVER_PLAN_SHA256
    layer_index: int = H776_LAYER_INDEX
    expected_target_unique_parameters: int = H776_TARGET_UNIQUE_PARAMETERS
    maximum_process_rss_bytes: int = H776_MAXIMUM_PROCESS_RSS_BYTES
    maximum_cuda_bytes_per_device: int = H776_MAXIMUM_CUDA_BYTES_PER_DEVICE
    require_zero_swap: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint_store_directory",
            _local_directory(
                self.checkpoint_store_directory,
                "checkpoint_store_directory",
            ),
        )
        for name in (
            "checkpoint_key",
            "checkpoint_manifest_sha256",
            "expected_trace_sha256",
            "expected_runtime_compiler_implementation_sha256",
            "expected_sequence_source_sha256",
            "expected_target_plan_fingerprint",
            "expected_target_topology_sha256",
            "expected_selective_catalog_sha256",
            "expected_selective_source_sha256",
            "expected_selective_provider_sha256",
            "expected_cache_sha256",
            "expected_token_bridge_sha256",
            "expected_tokenizer_sha256",
            "expected_source_tied_table_sha256",
            "expected_source_state_sha256",
            "expected_source_inventory_sha256",
            "expected_blank_report_sha256",
            "expected_blank_state_sha256",
            "expected_full_policy_sha256",
            "expected_origin_compiler_sha256",
            "expected_sequence_solver_plan_sha256",
        ):
            _digest(getattr(self, name), name)
        fixed = {
            "expected_target_plan_fingerprint": H776_TARGET_PLAN_FINGERPRINT,
            "expected_target_topology_sha256": H776_TARGET_TOPOLOGY_SHA256,
            "expected_selective_catalog_sha256": H776_SELECTIVE_CATALOG_SHA256,
            "expected_selective_source_sha256": H776_SELECTIVE_SOURCE_SHA256,
            "expected_selective_provider_sha256": H776_SELECTIVE_PROVIDER_SHA256,
            "expected_cache_sha256": H776_CACHE_SHA256,
            "expected_token_bridge_sha256": H776_TOKEN_BRIDGE_SHA256,
            "expected_tokenizer_sha256": H776_TOKENIZER_SHA256,
            "expected_source_tied_table_sha256": H776_SOURCE_TIED_TABLE_SHA256,
            "expected_source_state_sha256": H776_SOURCE_STATE_SHA256,
            "expected_source_inventory_sha256": H776_SOURCE_INVENTORY_SHA256,
            "expected_blank_report_sha256": H776_BLANK_REPORT_SHA256,
            "expected_blank_state_sha256": H776_BLANK_STATE_SHA256,
            "expected_full_policy_sha256": H776_FULL_POLICY_SHA256,
            "expected_origin_compiler_sha256": H776_ORIGIN_COMPILER_SHA256,
            "expected_sequence_solver_plan_sha256": (
                H776_SEQUENCE_SOLVER_PLAN_SHA256
            ),
            "layer_index": H776_LAYER_INDEX,
            "expected_target_unique_parameters": H776_TARGET_UNIQUE_PARAMETERS,
            "maximum_process_rss_bytes": H776_MAXIMUM_PROCESS_RSS_BYTES,
            "maximum_cuda_bytes_per_device": (
                H776_MAXIMUM_CUDA_BYTES_PER_DEVICE
            ),
            "require_zero_swap": True,
        }
        changed = [
            name for name, expected in fixed.items() if getattr(self, name) != expected
        ]
        if changed:
            raise Lfm2CpcfV5RealAttentionProbeError(
                "The real probe is pinned to the exact H776 layer-2 lane; "
                "changed fields: " + ", ".join(changed) + "."
            )
        if (
            self.expected_runtime_compiler_implementation_sha256
            == self.expected_origin_compiler_sha256
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "The probe requires the explicitly pinned new compiler runtime, "
                "not ordinary same-runtime resume."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_store_directory": str(self.checkpoint_store_directory),
            "checkpoint_key": self.checkpoint_key,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "expected_trace_sha256": self.expected_trace_sha256,
            "expected_runtime_compiler_implementation_sha256": (
                self.expected_runtime_compiler_implementation_sha256
            ),
            "expected_sequence_source_sha256": self.expected_sequence_source_sha256,
            "expected_target_plan_fingerprint": (
                self.expected_target_plan_fingerprint
            ),
            "expected_target_topology_sha256": (
                self.expected_target_topology_sha256
            ),
            "expected_selective_catalog_sha256": (
                self.expected_selective_catalog_sha256
            ),
            "expected_selective_source_sha256": (
                self.expected_selective_source_sha256
            ),
            "expected_selective_provider_sha256": (
                self.expected_selective_provider_sha256
            ),
            "expected_cache_sha256": self.expected_cache_sha256,
            "expected_token_bridge_sha256": self.expected_token_bridge_sha256,
            "expected_tokenizer_sha256": self.expected_tokenizer_sha256,
            "expected_source_tied_table_sha256": (
                self.expected_source_tied_table_sha256
            ),
            "expected_source_state_sha256": self.expected_source_state_sha256,
            "expected_source_inventory_sha256": (
                self.expected_source_inventory_sha256
            ),
            "expected_blank_report_sha256": self.expected_blank_report_sha256,
            "expected_blank_state_sha256": self.expected_blank_state_sha256,
            "expected_full_policy_sha256": self.expected_full_policy_sha256,
            "expected_origin_compiler_sha256": (
                self.expected_origin_compiler_sha256
            ),
            "expected_sequence_solver_plan_sha256": (
                self.expected_sequence_solver_plan_sha256
            ),
            "layer_index": self.layer_index,
            "expected_target_unique_parameters": (
                self.expected_target_unique_parameters
            ),
            "maximum_process_rss_bytes": self.maximum_process_rss_bytes,
            "maximum_cuda_bytes_per_device": (
                self.maximum_cuda_bytes_per_device
            ),
            "require_zero_swap": self.require_zero_swap,
        }


@dataclass(slots=True)
class Lfm2CpcfV5RealAttentionProbeRuntime:
    """Caller-built local runtime; no loader or capture path exists here."""

    source_model: Any
    target_model: Any
    trace_adapter: Any
    evidence_provider: Any
    source_tied_table: Any
    blank_target_report: Mapping[str, Any]
    target_plan_fingerprint: str
    runtime_policy: Any
    post_operator_chart_transport_shadow: bool = False


def _checkpoint_reference_mapping(verified: Any) -> dict[str, str]:
    reference = getattr(verified, "reference", None)
    if reference is None or not callable(getattr(reference, "as_dict", None)):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Checkpoint did not return one typed artifact reference."
        )
    return dict(reference.as_dict())


def authenticate_h776_k2_checkpoint(
    store: Any,
    verified: Any,
    pins: Lfm2CpcfV5RealAttentionProbePins,
) -> Mapping[str, Any]:
    """Validate the exact causal k=2 prefix without accepting k=1 or k=3."""

    reference = _checkpoint_reference_mapping(verified)
    if reference != {
        "artifact_kind": "checkpoint",
        "artifact_key": pins.checkpoint_key,
        "manifest_sha256": pins.checkpoint_manifest_sha256,
    }:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Opened checkpoint differs from the independent k=2 reference."
        )
    manifest = _plain(getattr(verified, "manifest", None))
    completed = _plain(getattr(verified, "completed_layers", None))
    if not isinstance(manifest, dict) or not isinstance(completed, dict):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Checkpoint manifest or completed-layer document is absent."
        )
    bindings = manifest.get("bindings")
    receipts = completed.get("completed_layer_receipts")
    incoming = completed.get("incoming_state_bank_metadata")
    if (
        not isinstance(bindings, dict)
        or not isinstance(receipts, list)
        or not isinstance(incoming, dict)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Checkpoint progress bindings are incomplete."
        )
    if (
        bindings.get("next_layer_index") != H776_LAYER_INDEX
        or completed.get("next_layer_index") != H776_LAYER_INDEX
        or completed.get("incoming_state_layer_index") != H776_LAYER_INDEX
        or completed.get("completed_layer_indices") != [0, 1]
        or [receipt.get("layer_index") for receipt in receipts] != [0, 1]
        or [receipt.get("operator_kind") for receipt in receipts]
        != ["short_conv", "short_conv"]
        or incoming.get("layer_index") != H776_LAYER_INDEX
        or incoming.get("boundary") != "layer.001.post_layer"
        or incoming.get("actual_folded_upstream_states") is not True
        or receipts[-1].get("next_bank_sha256")
        != incoming.get("state_bank_sha256")
        or receipts[1].get("input_bank_sha256")
        != receipts[0].get("next_bank_sha256")
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Checkpoint is not the exact contiguous short-conv k=2 causal prefix."
        )
    for receipt in receipts:
        execution = receipt.get("execution")
        if (
            not isinstance(execution, dict)
            or execution.get("closed_form_compilers") is not True
            or any(execution.get(name) != 0 for name in execution if name != "closed_form_compilers")
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "A completed prefix receipt does not prove fold-only execution."
            )
    topology = completed.get("target_topology_snapshot")
    if (
        not isinstance(topology, dict)
        or topology.get("model_type") != "lfm2"
        or topology.get("layer_count") != H776_LAYER_COUNT
        or topology.get("topology_sha256")
        != pins.expected_target_topology_sha256
        or completed.get("target_topology_sha256")
        != pins.expected_target_topology_sha256
        or bindings.get("target_topology_sha256")
        != pins.expected_target_topology_sha256
        or bindings.get("full_policy_sha256")
        != pins.expected_full_policy_sha256
        or bindings.get("compiler_fingerprint_sha256")
        != pins.expected_origin_compiler_sha256
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Checkpoint topology, policy, or origin compiler differs from H776."
        )
    boundary_reference = bindings.get("boundary")
    if (
        not isinstance(boundary_reference, dict)
        or boundary_reference.get("artifact_kind") != "boundary"
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Checkpoint does not bind one authenticated boundary frontier."
        )
    body = {
        "schema": LFM2_CPCF_V5_REAL_ATTENTION_CHECKPOINT_AUDIT_SCHEMA,
        "checkpoint_reference": reference,
        "boundary_reference": dict(boundary_reference),
        "next_layer_index": H776_LAYER_INDEX,
        "completed_layer_indices": [0, 1],
        "completed_operator_kinds": ["short_conv", "short_conv"],
        "incoming_boundary": "layer.001.post_layer",
        "incoming_state_bank_sha256": incoming["state_bank_sha256"],
        "target_topology_sha256": pins.expected_target_topology_sha256,
        "full_policy_sha256": pins.expected_full_policy_sha256,
        "origin_compiler_sha256": pins.expected_origin_compiler_sha256,
        "k1_rejected_as_insufficient": True,
        "k3_not_opened": True,
    }
    # Reauthenticate immediately after inspecting the progress document.
    final = store.final_reauthenticate_checkpoint(verified)
    if _checkpoint_reference_mapping(final) != reference:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Checkpoint identity changed during k=2 authentication."
        )
    return MappingProxyType({**body, "audit_sha256": _sha256(body)})


def authenticate_h776_boundary_frontier(
    frontier: Any,
    pins: Lfm2CpcfV5RealAttentionProbePins,
) -> Mapping[str, Any]:
    report = _plain(getattr(frontier, "report", None))
    if not isinstance(report, dict):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Boundary artifact did not reconstruct a typed v5 frontier."
        )
    context = report.get("context")
    if not isinstance(context, dict):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Boundary frontier has no authenticated runtime context."
        )
    expected_context = {
        "source_model_state_sha256": pins.expected_source_state_sha256,
        "source_model_inventory_sha256": pins.expected_source_inventory_sha256,
        "cache_sha256": pins.expected_cache_sha256,
        "selective_catalog_sha256": pins.expected_selective_catalog_sha256,
        "selective_source_sha256": pins.expected_selective_source_sha256,
        "selective_provider_sha256": pins.expected_selective_provider_sha256,
        "trace_sha256": pins.expected_trace_sha256,
        "tokenizer_sha256": pins.expected_tokenizer_sha256,
        "token_bridge_sha256": pins.expected_token_bridge_sha256,
        "source_tied_table_sha256": pins.expected_source_tied_table_sha256,
        "blank_target_report_sha256": pins.expected_blank_report_sha256,
        "blank_target_state_sha256": pins.expected_blank_state_sha256,
        "target_topology_sha256": pins.expected_target_topology_sha256,
        "planned_target_unique_parameters": H776_TARGET_UNIQUE_PARAMETERS,
        "runtime_policy_sha256": pins.expected_full_policy_sha256,
        "compiler_implementation_sha256": pins.expected_origin_compiler_sha256,
    }
    mismatches = [
        name
        for name, expected in expected_context.items()
        if context.get(name) != expected
    ]
    if (
        report.get("next_layer_index") != H776_LAYER_INDEX
        or report.get("expected_layer_count") != H776_LAYER_COUNT
        or report.get("source_width") != H776_SOURCE_HIDDEN_SIZE
        or report.get("target_width") != H776_TARGET_HIDDEN_SIZE
        or tuple(report.get("layer_operator_kinds", ())) != H776_OPERATOR_KINDS
        or len(report.get("ordered_phase_sha256", ())) != H776_PHASE_COUNT
        or [
            layer.get("operator_kind")
            for layer in report.get("completed_layer_charts", ())
        ]
        != ["short_conv", "short_conv"]
        or mismatches
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Boundary frontier differs from the exact H776 k=2 lane"
            + ("" if not mismatches else ": " + ", ".join(mismatches))
            + "."
        )
    body = {
        "frontier_sha256": report["frontier_sha256"],
        "context_sha256": context["context_sha256"],
        "next_layer_index": report["next_layer_index"],
        "expected_layer_count": report["expected_layer_count"],
        "source_width": report["source_width"],
        "target_width": report["target_width"],
        "phase_count": len(report["ordered_phase_sha256"]),
        "split_row_counts": dict(report["split_row_counts"]),
        "layer_operator_kinds": list(report["layer_operator_kinds"]),
        "cross_runtime_probe_only": True,
        "ordinary_resume_gate_weakened": False,
    }
    return MappingProxyType({**body, "audit_sha256": _sha256(body)})


def authenticate_h776_phase_inventory(
    phases: Sequence[Any],
    *,
    frontier: Any,
) -> Mapping[str, Any]:
    ordered = tuple(phases)
    split_counts = {
        split: sum(getattr(phase, "split", None) == split for phase in ordered)
        for split in H776_SPLIT_PHASE_COUNTS
    }
    split_rows = {
        split: sum(
            int(getattr(phase, "batch_size"))
            * int(getattr(phase, "sampled_suffix_tokens"))
            for phase in ordered
            if getattr(phase, "split", None) == split
        )
        for split in H776_SPLIT_PHASE_COUNTS
    }
    trajectories: dict[tuple[int, str], list[str]] = {}
    trajectory_splits: dict[tuple[int, str], str] = {}
    for phase in ordered:
        identity = (
            int(getattr(phase, "batch_index")),
            str(getattr(phase, "record_id")),
        )
        trajectories.setdefault(identity, []).append(str(getattr(phase, "phase")))
        previous = trajectory_splits.setdefault(identity, str(getattr(phase, "split")))
        if previous != getattr(phase, "split"):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "A causal trajectory crosses evaluation splits."
            )
    trajectory_counts = {
        split: sum(value == split for value in trajectory_splits.values())
        for split in H776_SPLIT_TRAJECTORY_COUNTS
    }
    frontier_rows = dict(frontier.report["split_row_counts"])
    if (
        len(ordered) != H776_PHASE_COUNT
        or len(trajectories) != H776_TRAJECTORY_COUNT
        or split_counts != dict(H776_SPLIT_PHASE_COUNTS)
        or trajectory_counts != dict(H776_SPLIT_TRAJECTORY_COUNTS)
        or any(tuple(names) != _EXPECTED_PHASE_NAMES for names in trajectories.values())
        or any(len(names) != H776_PHASES_PER_TRAJECTORY for names in trajectories.values())
        or split_rows != frontier_rows
        or split_rows["fit"] != H776_FIT_QUERY_ROWS
        or split_rows["validation"] != H776_VALIDATION_QUERY_ROWS
        or split_rows["selection"] <= 0
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Live phases differ from the authenticated complete H776 causal bank."
        )
    body = {
        "phase_count": len(ordered),
        "trajectory_count": len(trajectories),
        "phases_per_trajectory": H776_PHASES_PER_TRAJECTORY,
        "ordered_phase_names": list(_EXPECTED_PHASE_NAMES),
        "split_phase_counts": split_counts,
        "split_trajectory_counts": trajectory_counts,
        "split_query_rows": split_rows,
        "complete_prefill_plus_eight_decode_phases": True,
    }
    return MappingProxyType({**body, "audit_sha256": _sha256(body)})


class Lfm2SequenceEffectivePcgCollector:
    """Collect and authenticate the scoped observation-only PCG events."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def __call__(self, event: Mapping[str, Any]) -> None:
        value = _plain(event)
        body = dict(value)
        claimed = _digest(
            body.pop("observation_sha256", None),
            "PCG observation SHA-256",
        )
        report = body.get("report")
        report_fields = {
            "role",
            "solver",
            "maximum_iterations",
            "iterations_executed",
            "required_relative_normal_residual",
            "initial_relative_normal_residual",
            "final_relative_normal_residual",
            "relative_normal_residual_history",
            "solution_sha256",
            "report_sha256",
        }
        if (
            set(value)
            != {
                "schema",
                "solver_kind",
                "report",
                "report_sha256",
                "solver_result_or_arguments_modified",
                "observation_sha256",
            }
            or body.get("schema")
            != "neural-forge-lfm2-sequence-effective-pcg-observation-v1"
            or body.get("solver_kind") != "single_tensor"
            or body.get("solver_result_or_arguments_modified") is not False
            or not isinstance(report, dict)
            or set(report) != report_fields
            or report.get("report_sha256") != body.get("report_sha256")
            or _sha256({k: v for k, v in report.items() if k != "report_sha256"})
            != report.get("report_sha256")
            or _sha256(body) != claimed
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Sequence-effective PCG telemetry is not authenticated."
            )
        role = report.get("role")
        match = (
            None
            if not isinstance(role, str)
            else _STRUCTURED_VALUE_PCG_ROLE.fullmatch(role)
        )
        iterations = report.get("iterations_executed")
        required = report.get("required_relative_normal_residual")
        initial = report.get("initial_relative_normal_residual")
        final = report.get("final_relative_normal_residual")
        history = report.get("relative_normal_residual_history")
        if (
            match is None
            or report.get("solver")
            != "deterministic_jacobi_pcg_single_tensor"
            or report.get("maximum_iterations") != H776_PCG_MAXIMUM_ITERATIONS
            or isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or not 0 <= iterations <= H776_PCG_MAXIMUM_ITERATIONS
            or required != H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
            or isinstance(initial, bool)
            or not isinstance(initial, (int, float))
            or not math.isfinite(float(initial))
            or float(initial) < 0.0
            or isinstance(final, bool)
            or not isinstance(final, (int, float))
            or not math.isfinite(float(final))
            or float(final) < 0.0
            or float(final) > H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
            or not isinstance(history, list)
            or len(history) != iterations
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or float(item) < 0.0
                for item in history
            )
            or (
                iterations == 0
                and (
                    history
                    or float(initial) != float(final)
                )
            )
            or (
                iterations > 0
                and float(history[-1]) != float(final)
            )
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "A sequence-effective PCG solve missed its convergence gate."
            )
        _digest(report.get("solution_sha256"), "PCG solution SHA-256")
        if any(
            previous["report"]["role"] == role
            or previous["report_sha256"] == body["report_sha256"]
            or previous["observation_sha256"] == claimed
            for previous in self._events
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "A duplicate sequence-effective PCG event was observed."
            )
        self._events.append(value)

    def finalize(
        self,
        *,
        sequence_artifact: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        artifact = _self_sealed(
            sequence_artifact,
            seal_field="report_sha256",
            name="sequence-effective operator artifact",
        )
        frontier = artifact.get("candidate_frontier")
        solve_hashes = artifact.get("solve_report_sha256")
        effective_solve = artifact.get("effective_operator_solve")
        if (
            artifact.get("candidate_id")
            != H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID
            or not isinstance(frontier, list)
            or len(frontier) not in {3, 4}
            or artifact.get("candidate_frontier_sha256") != _sha256(frontier)
            or artifact.get("bounded_proposal_count") != len(frontier)
            or artifact.get("selected_proposal_index") != len(frontier) - 1
            or not isinstance(solve_hashes, list)
            or not isinstance(effective_solve, dict)
            or frontier[0].get("proposal_index") != 0
            or frontier[0].get("stage")
            != "scalable_producer_shared_value_sequence_output_seed"
            or frontier[-1].get("proposal_index") != len(frontier) - 1
            or frontier[-1].get("stage")
            != "fixed_producer_value_subspace_anchor"
            or frontier[-1].get("selected_for_materialization") is not True
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Sequence-effective artifact lacks the exact streamed ALS "
                "proposal inventory."
            )
        als = frontier[1:-1]
        iteration_count = len(als)
        accepted = 0
        artifact_value_hashes: list[str] = []
        reconstructed_solve_hashes = [
            _digest(
                effective_solve.get("report_sha256"),
                "effective operator solve SHA-256",
            ),
            _digest(
                frontier[0].get("solve_report_sha256"),
                "sequence seed output solve SHA-256",
            ),
        ]
        for iteration, proposal in enumerate(als, start=1):
            value_hashes = proposal.get("value_solve_report_sha256")
            output_hash = proposal.get("output_solve_report_sha256")
            if (
                proposal.get("proposal_index") != iteration
                or proposal.get("stage")
                != "monotone_streamed_effective_space_value_output_als"
                or not isinstance(proposal.get("accepted"), bool)
                or not isinstance(value_hashes, list)
                or len(value_hashes) != H776_TARGET_KEY_VALUE_HEADS
            ):
                raise Lfm2CpcfV5RealAttentionProbeError(
                    "Sequence-effective ALS proposal inventory is incomplete."
                )
            if proposal["accepted"]:
                accepted += 1
            elif iteration != iteration_count:
                raise Lfm2CpcfV5RealAttentionProbeError(
                    "A rejected ALS proposal was followed by another iteration."
                )
            normalized = [
                _digest(
                    value,
                    f"ALS iteration {iteration} value solve SHA-256",
                )
                for value in value_hashes
            ]
            if len(set(normalized)) != len(normalized):
                raise Lfm2CpcfV5RealAttentionProbeError(
                    "ALS value-solve report hashes must be unique per iteration."
                )
            artifact_value_hashes.extend(normalized)
            reconstructed_solve_hashes.extend(normalized)
            reconstructed_solve_hashes.append(
                _digest(
                    output_hash,
                    f"ALS iteration {iteration} output solve SHA-256",
                )
            )
        reconstructed_solve_hashes.append(
            _digest(
                frontier[-1].get("output_solve_report_sha256"),
                "anchored output solve SHA-256",
            )
        )
        normalized_solve_hashes = [
            _digest(value, "sequence solve report SHA-256")
            for value in solve_hashes
        ]
        if (
            normalized_solve_hashes != reconstructed_solve_hashes
            or len(set(artifact_value_hashes)) != len(artifact_value_hashes)
            or artifact.get("accepted_als_iterations") != accepted
            or artifact.get("executed_als_iteration_limit") != 2
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Sequence-effective solve list does not exactly bind its ALS "
                "proposal inventory."
            )
        expected_roles = [
            (
                f"sequence_effective_structured_value_iteration_{iteration}_"
                f"group_{group}"
            )
            for iteration in range(1, iteration_count + 1)
            for group in range(H776_TARGET_KEY_VALUE_HEADS)
        ]
        observed_roles = [event["report"]["role"] for event in self._events]
        if observed_roles != expected_roles:
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Observed PCG roles differ from the exact unique H776 "
                "iteration/group inventory."
            )
        event_bindings = []
        for event, role, outer_hash in zip(
            self._events,
            expected_roles,
            artifact_value_hashes,
            strict=True,
        ):
            match = _STRUCTURED_VALUE_PCG_ROLE.fullmatch(role)
            assert match is not None
            event_bindings.append(
                {
                    "role": role,
                    "iteration": int(match.group(1)),
                    "group": int(match.group(2)),
                    "observation_sha256": event["observation_sha256"],
                    "pcg_report_sha256": event["report_sha256"],
                    "artifact_value_solve_report_sha256": outer_hash,
                }
            )
        body = {
            "schema": LFM2_CPCF_V5_REAL_ATTENTION_PCG_AUDIT_SCHEMA,
            "event_count": len(self._events),
            "multi_rhs_solve_count": 0,
            "single_tensor_solve_count": len(self._events),
            "als_iteration_count": iteration_count,
            "expected_roles": expected_roles,
            "observed_roles": observed_roles,
            "events": list(self._events),
            "event_to_artifact_solve_bindings": event_bindings,
            "sequence_artifact_report_sha256": artifact["report_sha256"],
            "sequence_candidate_frontier_sha256": (
                artifact["candidate_frontier_sha256"]
            ),
            "sequence_solve_report_sha256": normalized_solve_hashes,
            "sequence_solve_report_inventory_sha256": _sha256(
                normalized_solve_hashes
            ),
            "artifact_value_solve_report_sha256": artifact_value_hashes,
            "all_roles_unique": True,
            "all_event_report_hashes_unique": True,
            "all_observation_hashes_unique": True,
            "all_iterations_at_most": H776_PCG_MAXIMUM_ITERATIONS,
            "all_final_relative_normal_residuals_at_most": (
                H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
            ),
            "solver_results_or_arguments_modified": False,
        }
        return MappingProxyType({**body, "audit_sha256": _sha256(body)})


def _linux_process_memory_sample() -> Mapping[str, int]:
    path = Path("/proc/self/status")
    if not path.is_file():
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The real probe requires Linux /proc/self/status memory evidence."
        )
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Could not read process memory evidence."
        ) from exc
    for line in lines:
        name, separator, raw = line.partition(":")
        if name not in {"VmRSS", "VmHWM", "VmSwap"} or not separator:
            continue
        fields = raw.split()
        if len(fields) != 2 or fields[1] != "kB":
            raise Lfm2CpcfV5RealAttentionProbeError(
                f"Unexpected {name} format in process memory evidence."
            )
        values[name] = int(fields[0]) * 1024
    if set(values) != {"VmRSS", "VmHWM", "VmSwap"}:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Process memory evidence lacks RSS, high-water, or swap."
        )
    return MappingProxyType(values)


class Lfm2CpcfV5RealAttentionResourceMonitor:
    """Sample process/CUDA peaks during the one-layer transaction."""

    def __init__(
        self,
        *,
        process_sampler: Callable[[], Mapping[str, int]] = (
            _linux_process_memory_sample
        ),
        cuda_sampler: Callable[[], Mapping[str, int]],
        sample_interval_seconds: float = 0.05,
    ) -> None:
        if not callable(process_sampler) or not callable(cuda_sampler):
            raise TypeError("Resource samplers must be callable.")
        self._process_sampler = process_sampler
        self._cuda_sampler = cuda_sampler
        self._interval = float(sample_interval_seconds)
        if not 0.0 < self._interval <= 1.0:
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Resource sample interval must be in (0, 1]."
            )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._failure: BaseException | None = None

    def _sample(self, label: str) -> None:
        process = {str(k): int(v) for k, v in self._process_sampler().items()}
        cuda = {str(k): int(v) for k, v in self._cuda_sampler().items()}
        if (
            process["VmRSS"] > H776_MAXIMUM_PROCESS_RSS_BYTES
            or process["VmHWM"] > H776_MAXIMUM_PROCESS_RSS_BYTES
            or process["VmSwap"] != 0
            or any(
                value > H776_MAXIMUM_CUDA_BYTES_PER_DEVICE
                for value in cuda.values()
            )
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "The H776 probe exceeded its RSS/CUDA cap or used swap."
            )
        sample = {
            "label": label,
            "monotonic_ns": time.monotonic_ns(),
            "process": process,
            "cuda_peak_reserved_bytes": cuda,
        }
        with self._lock:
            self._samples.append(sample)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._sample("periodic")
            except BaseException as exc:
                with self._lock:
                    if self._failure is None:
                        self._failure = exc
                return

    def __enter__(self) -> Lfm2CpcfV5RealAttentionResourceMonitor:
        if self._thread is not None:
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Resource monitor cannot be entered twice."
            )
        self._sample("enter")
        self._thread = threading.Thread(
            target=self._run,
            name="lfm2-h776-attention-resource-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def checkpoint(self, label: str) -> None:
        self._sample(str(label))
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise Lfm2CpcfV5RealAttentionProbeError(
                "The resource sampler failed between probe checkpoints."
            ) from failure

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join(timeout=max(1.0, self._interval * 4))
        if self._thread.is_alive():
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Resource monitor did not stop deterministically."
            )
        if exc is None:
            self.checkpoint("exit")

    def report(self) -> Mapping[str, Any]:
        with self._lock:
            samples = _plain(self._samples)
            failure = self._failure
        if failure is not None or not samples:
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Resource monitor has no clean complete sample history."
            ) from failure
        body = {
            "schema": LFM2_CPCF_V5_REAL_ATTENTION_RESOURCE_REPORT_SCHEMA,
            "sample_interval_seconds": self._interval,
            "sample_count": len(samples),
            "maximum_process_rss_bytes": H776_MAXIMUM_PROCESS_RSS_BYTES,
            "maximum_cuda_bytes_per_device": (
                H776_MAXIMUM_CUDA_BYTES_PER_DEVICE
            ),
            "peak_vm_rss_bytes": max(
                sample["process"]["VmRSS"] for sample in samples
            ),
            "peak_vm_hwm_bytes": max(
                sample["process"]["VmHWM"] for sample in samples
            ),
            "peak_vm_swap_bytes": max(
                sample["process"]["VmSwap"] for sample in samples
            ),
            "peak_cuda_reserved_bytes": {
                device: max(
                    sample["cuda_peak_reserved_bytes"].get(device, 0)
                    for sample in samples
                )
                for device in sorted(
                    {
                        device
                        for sample in samples
                        for device in sample["cuda_peak_reserved_bytes"]
                    }
                )
            },
            "samples": samples,
            "rss_cap_passed": True,
            "cuda_cap_passed": True,
            "zero_swap_passed": True,
        }
        return MappingProxyType({**body, "report_sha256": _sha256(body)})


def _store_inventory(root: Path) -> Mapping[str, tuple[str, int, int]]:
    result: dict[str, tuple[str, int, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Checkpoint store acquired a symbolic link."
            )
        stat = path.stat()
        result[path.relative_to(root).as_posix()] = (
            "directory" if path.is_dir() else "file",
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )
    return MappingProxyType(result)


def _target_geometry(target_model: Any) -> dict[str, Any]:
    config = getattr(target_model, "config", None)
    layers = getattr(getattr(target_model, "model", None), "layers", None)
    layer_types = tuple(getattr(config, "layer_types", ()))
    if (
        config is None
        or layers is None
        or len(layers) != H776_LAYER_COUNT
        or int(getattr(config, "hidden_size", -1)) != H776_TARGET_HIDDEN_SIZE
        or int(getattr(config, "intermediate_size", -1))
        != H776_TARGET_INTERMEDIATE_SIZE
        or int(getattr(config, "num_attention_heads", -1))
        != H776_TARGET_QUERY_HEADS
        or int(getattr(config, "num_key_value_heads", -1))
        != H776_TARGET_KEY_VALUE_HEADS
        or layer_types != H776_LAYER_TYPES
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Live blank is not the exact H776 dense target architecture."
        )
    attention = layers[H776_LAYER_INDEX].self_attn
    shapes = {
        "q_proj.weight": list(attention.q_proj.weight.shape),
        "k_proj.weight": list(attention.k_proj.weight.shape),
        "v_proj.weight": list(attention.v_proj.weight.shape),
        "out_proj.weight": list(attention.out_proj.weight.shape),
    }
    expected_shapes = {
        "q_proj.weight": [640, 776],
        "k_proj.weight": [320, 776],
        "v_proj.weight": [320, 776],
        "out_proj.weight": [776, 640],
    }
    if (
        shapes != expected_shapes
        or int(getattr(attention, "head_dim", -1)) != H776_HEAD_DIM
        or not bool(getattr(layers[H776_LAYER_INDEX], "is_attention_layer", False))
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Live layer 2 is not the exact H776 attention macrocell."
        )
    return {
        "hidden_size": H776_TARGET_HIDDEN_SIZE,
        "intermediate_size": H776_TARGET_INTERMEDIATE_SIZE,
        "query_heads": H776_TARGET_QUERY_HEADS,
        "key_value_heads": H776_TARGET_KEY_VALUE_HEADS,
        "head_dim": H776_HEAD_DIM,
        "layer_count": H776_LAYER_COUNT,
        "layer_types": list(H776_LAYER_TYPES),
        "layer_2_attention_shapes": shapes,
    }


def authenticate_h776_attention_result(
    result: Any,
    *,
    input_bank_sha256: str,
    pcg_audit: Mapping[str, Any],
    pins: Lfm2CpcfV5RealAttentionProbePins,
) -> Mapping[str, Any]:
    authenticated_pcg = _self_sealed(
        pcg_audit,
        seal_field="audit_sha256",
        name="PCG audit",
    )
    if (
        authenticated_pcg.get("schema")
        != LFM2_CPCF_V5_REAL_ATTENTION_PCG_AUDIT_SCHEMA
        or authenticated_pcg.get("multi_rhs_solve_count") != 0
        or authenticated_pcg.get("single_tensor_solve_count") not in {5, 10}
        or authenticated_pcg.get("event_count")
        != authenticated_pcg.get("single_tensor_solve_count")
        or authenticated_pcg.get("all_roles_unique") is not True
        or authenticated_pcg.get("all_event_report_hashes_unique") is not True
        or authenticated_pcg.get("all_observation_hashes_unique") is not True
        or authenticated_pcg.get("solver_results_or_arguments_modified")
        is not False
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "PCG audit seal or exact H776 solve inventory is invalid."
        )
    report = _self_sealed(
        getattr(result, "report", None),
        seal_field="result_sha256",
        name="layer result",
    )
    policy = report.get("solver_evidence_policy")
    if not isinstance(policy, dict):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Layer result lacks solver evidence policy."
        )
    selection = policy.get("attention_candidate_selection")
    evaluations = policy.get("attention_candidate_evaluations")
    if not isinstance(selection, dict) or not isinstance(evaluations, dict):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Layer result lacks the three-arm attention tournament."
        )
    candidate_ids = tuple(selection.get("candidate_ids", ()))
    selected_id = selection.get("selected_candidate_id")
    if (
        report.get("layer_index") != H776_LAYER_INDEX
        or report.get("operator_kind") != "full_attention"
        or report.get("input_bank_sha256") != input_bank_sha256
        or report.get("split_phase_counts") != dict(H776_SPLIT_PHASE_COUNTS)
        or candidate_ids != H776_ATTENTION_CANDIDATE_IDS
        or set(evaluations) != set(H776_ATTENTION_CANDIDATE_IDS)
        or selection.get("candidate_count") != 3
        or selection.get("selection_split_only") is not True
        or selection.get("winner_frozen_before_validation") is not True
        or selection.get("validation_phase_count_consumed_before_freeze") != 0
        or policy.get("attention_each_candidate_has_own_fit_only_mlp") is not True
        or policy.get("attention_winner_frozen_before_validation") is not True
        or policy.get("attention_validation_replays_before_winner_freeze") != 0
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Attention tournament, split isolation, or layer identity changed."
        )
    for candidate_id, evaluation in evaluations.items():
        if (
            evaluation.get("candidate_id") != candidate_id
            or evaluation.get("operator_compiled_from_splits") != ["fit"]
            or evaluation.get("mlp_compiled_from_splits") != ["fit"]
            or evaluation.get("candidate_evaluated_on_splits") != ["selection"]
            or evaluation.get("validation_phase_count_consumed") != 0
            or evaluation.get("training_steps") != 0
            or evaluation.get("parameters_pruned") != 0
            or evaluation.get("parameters_quantized") != 0
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                f"Attention candidate {candidate_id!r} crossed a split or mutation gate."
            )
    sequence = evaluations[H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]
    raw_artifact = sequence.get("operator_artifact")
    artifact = (
        None
        if not isinstance(raw_artifact, dict)
        else _self_sealed(
            raw_artifact,
            seal_field="report_sha256",
            name="sequence-effective operator artifact",
        )
    )
    solver_plan = None if not isinstance(artifact, dict) else artifact.get("solver_plan")
    if (
        not isinstance(artifact, dict)
        or artifact.get("candidate_id") != H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID
        or artifact.get("fixed_query_key_state_reused_exactly") is not True
        or artifact.get("fit_only_compilation") is not True
        or artifact.get("compiled_from_splits") != ["fit"]
        or artifact.get("selection_case_count_consumed") != 0
        or artifact.get("validation_case_count_consumed") != 0
        or artifact.get("dense_target_native_qkvo") is not True
        or artifact.get("ordinary_dense_target_value_output") is not True
        or artifact.get("forbidden_value_coefficient_tensor_materialized") is not False
        or artifact.get("forbidden_hidden_head_dim_gram_materialized") is not False
        or not isinstance(solver_plan, dict)
        or solver_plan.get("strategy") != "streamed_cross_head_statistics_pcg"
        or solver_plan.get("report_sha256")
        != pins.expected_sequence_solver_plan_sha256
        or solver_plan.get("fit_rows") != H776_FIT_QUERY_ROWS
        or solver_plan.get("target_query_heads") != H776_TARGET_QUERY_HEADS
        or solver_plan.get("target_hidden_size") != H776_TARGET_HIDDEN_SIZE
        or solver_plan.get("target_output_size") != H776_TARGET_HIDDEN_SIZE
        or solver_plan.get("head_dim") != H776_HEAD_DIM
        or solver_plan.get("forbidden_value_coefficient_materialized") is not False
        or solver_plan.get("forbidden_value_dense_gram_materialized") is not False
        or solver_plan.get("complete_operator_dense_factorization_avoided") is not True
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Sequence-effective arm did not execute the pinned scalable H776 plan."
        )
    artifact_frontier = artifact.get("candidate_frontier")
    artifact_solve_hashes = artifact.get("solve_report_sha256")
    bindings = authenticated_pcg.get("event_to_artifact_solve_bindings")
    events = authenticated_pcg.get("events")
    expected_roles = authenticated_pcg.get("expected_roles")
    observed_roles = authenticated_pcg.get("observed_roles")
    if (
        not isinstance(artifact_frontier, list)
        or artifact.get("candidate_frontier_sha256")
        != _sha256(artifact_frontier)
        or not isinstance(artifact_solve_hashes, list)
        or authenticated_pcg.get("sequence_artifact_report_sha256")
        != artifact["report_sha256"]
        or authenticated_pcg.get("sequence_candidate_frontier_sha256")
        != artifact["candidate_frontier_sha256"]
        or authenticated_pcg.get("sequence_solve_report_sha256")
        != artifact_solve_hashes
        or authenticated_pcg.get("sequence_solve_report_inventory_sha256")
        != _sha256(artifact_solve_hashes)
        or not isinstance(bindings, list)
        or not isinstance(events, list)
        or not isinstance(expected_roles, list)
        or expected_roles != observed_roles
        or not bindings
        or len(bindings) != len(events)
        or len(bindings) != len(expected_roles)
        or any(
            binding.get("role") != role
            or event.get("report", {}).get("role") != role
            or binding.get("observation_sha256")
            != event.get("observation_sha256")
            or binding.get("pcg_report_sha256")
            != event.get("report_sha256")
            for binding, event, role in zip(
                bindings,
                events,
                expected_roles,
                strict=True,
            )
        )
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "PCG event hashes are not cross-bound to the sealed sequence "
            "artifact and ALS proposal inventory."
        )
    replayed_collector = Lfm2SequenceEffectivePcgCollector()
    for event in events:
        replayed_collector(event)
    replayed_pcg = replayed_collector.finalize(
        sequence_artifact=artifact,
    )
    if _plain(replayed_pcg) != authenticated_pcg:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "PCG audit does not reproduce exactly from its sealed events and "
            "sequence artifact."
        )
    execution = artifact.get("execution")
    if (
        not isinstance(execution, dict)
        or set(execution) != set(_ZERO_EXECUTION_FIELDS)
        or any(
            type(execution.get(name)) is not int or execution[name] != 0
            for name in _ZERO_EXECUTION_FIELDS
        )
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Sequence-effective artifact does not prove fold-only execution."
        )
    keys = {}
    behavioral_keys = {}
    for candidate_id in H776_ATTENTION_CANDIDATE_IDS:
        raw_key = evaluations[candidate_id].get("selection_key")
        if (
            not isinstance(raw_key, list)
            or len(raw_key) != 11
            or raw_key[-1] != candidate_id
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in raw_key[:-1]
            )
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                f"Attention candidate {candidate_id!r} has an invalid "
                "behavioral selection key."
            )
        keys[candidate_id] = tuple(raw_key)
        behavioral_keys[candidate_id] = tuple(
            float(value) for value in raw_key[:-1]
        )
    sequence_won = selected_id == H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID
    sequence_strictly_better = all(
        behavioral_keys[H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]
        < behavioral_keys[candidate_id]
        for candidate_id in H776_ATTENTION_CANDIDATE_IDS[:-1]
    )
    next_bank = getattr(result, "next_bank", None)
    if (
        next_bank is None
        or getattr(next_bank, "layer_index", None) != H776_LAYER_INDEX + 1
        or report.get("next_bank_sha256")
        != getattr(next_bank, "state_bank_sha256", None)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Layer result did not produce the in-memory layer-3 input bank."
        )
    body = {
        "layer_result_sha256": report["result_sha256"],
        "input_bank_sha256": input_bank_sha256,
        "next_bank_sha256": report["next_bank_sha256"],
        "candidate_ids": list(candidate_ids),
        "selected_candidate_id": selected_id,
        "selection_keys": {
            candidate_id: list(keys[candidate_id])
            for candidate_id in H776_ATTENTION_CANDIDATE_IDS
        },
        "behavioral_selection_keys": {
            candidate_id: list(behavioral_keys[candidate_id])
            for candidate_id in H776_ATTENTION_CANDIDATE_IDS
        },
        "lexical_tie_break_excluded_from_strict_improvement": True,
        "sequence_effective_won": sequence_won,
        "sequence_effective_strictly_better_than_both_controls": (
            sequence_strictly_better
        ),
        "scientific_outcome": (
            "positive_sequence_effective_selection"
            if sequence_won and sequence_strictly_better
            else "honest_negative_or_control_selected"
        ),
        "full_validation_metrics": _plain(report["split_metrics"]["validation"]),
        "sequence_solver_plan": _plain(solver_plan),
        "pcg_convergence": authenticated_pcg,
        "in_memory_next_bank_created_by_layer_replay": True,
        "k3_checkpoint_published": False,
        "layer_3_compiled": False,
        "automatic_retry_or_retuning_performed": False,
    }
    return MappingProxyType({**body, "audit_sha256": _sha256(body)})


def execute_lfm2_cpcf_v5_real_attention_probe(
    runtime: Lfm2CpcfV5RealAttentionProbeRuntime,
    pins: Lfm2CpcfV5RealAttentionProbePins,
    *,
    resource_monitor: Lfm2CpcfV5RealAttentionResourceMonitor,
) -> Mapping[str, Any]:
    """Compile/evaluate exactly real attention layer 2 and stop in memory."""

    if not isinstance(runtime, Lfm2CpcfV5RealAttentionProbeRuntime):
        raise TypeError("runtime must be Lfm2CpcfV5RealAttentionProbeRuntime.")
    if not isinstance(pins, Lfm2CpcfV5RealAttentionProbePins):
        raise TypeError("pins must be Lfm2CpcfV5RealAttentionProbePins.")
    if not isinstance(
        resource_monitor,
        Lfm2CpcfV5RealAttentionResourceMonitor,
    ):
        raise TypeError(
            "resource_monitor must be Lfm2CpcfV5RealAttentionResourceMonitor."
        )
    if type(runtime.post_operator_chart_transport_shadow) is not bool:
        raise TypeError(
            "post_operator_chart_transport_shadow must be bool."
        )

    from . import hf_lfm2_cpcf_model_fold as model_fold
    from . import hf_lfm2_cpcf_v5_model_compiler as v5
    from . import hf_lfm2_sequence_effective_attention_fold as sequence_fold
    from .hf_lfm2_cpcf_layer_compiler import (
        compile_and_install_lfm2_cpcf_layer,
    )
    from .hf_lfm2_cpcf_model_compiler import (
        rebind_lfm2_cpcf_phase_state_bank,
    )
    from .hf_lfm2_cpcf_residual_error_controller import (
        apply_lfm2_cpcf_post_operator_chart_transport_shadow,
    )
    from .hf_lfm2_cpcf_resume_artifact import (
        Lfm2CpcfArtifactReference,
        Lfm2CpcfResumeArtifactStore,
    )
    from .hf_lfm2_cpcf_resume_bridge import (
        install_lfm2_cpcf_prefix_checkpoint,
        reconstruct_lfm2_cpcf_phase_state_bank,
    )
    from .hf_lfm2_cpcf_v5_resume import Lfm2CpcfV5BoundaryFrontier

    implementation = dict(v5.lfm2_cpcf_v5_compiler_implementation())
    if (
        implementation.get("implementation_sha256")
        != pins.expected_runtime_compiler_implementation_sha256
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Live new compiler runtime differs from its invocation pin."
        )
    sequence_path = Path(sequence_fold.__file__).resolve(strict=True)
    sequence_source_sha256 = hashlib.sha256(sequence_path.read_bytes()).hexdigest()
    if sequence_source_sha256 != pins.expected_sequence_source_sha256:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Live sequence-effective implementation differs from its source pin."
        )
    geometry = _target_geometry(runtime.target_model)
    source_before = model_fold._snapshot_model_auth(runtime.source_model)
    target_before = model_fold._snapshot_model(runtime.target_model)
    blank_report = _self_sealed(
        runtime.blank_target_report,
        seal_field="report_sha256",
        name="blank target report",
    )
    source_parameter_inventory = model_fold.independent_parameter_inventory(
        runtime.source_model
    )
    target_inventory = model_fold.independent_parameter_inventory(
        runtime.target_model
    )
    if (
        runtime.target_plan_fingerprint != pins.expected_target_plan_fingerprint
        or blank_report["report_sha256"] != pins.expected_blank_report_sha256
        or target_before.state_fingerprint != pins.expected_blank_state_sha256
        or source_before.state_fingerprint != pins.expected_source_state_sha256
        or int(source_parameter_inventory["unique_parameters"])
        != H776_PARENT_UNIQUE_PARAMETERS
        or int(target_inventory["unique_parameters"])
        != H776_TARGET_UNIQUE_PARAMETERS
        or {str(parameter.dtype) for parameter in runtime.source_model.parameters()}
        != {"torch.bfloat16"}
        or {str(parameter.dtype) for parameter in runtime.target_model.parameters()}
        != {"torch.bfloat16"}
        or any(
            bool(getattr(parameter, "is_quantized", False))
            for parameter in runtime.source_model.parameters()
        )
        or any(
            bool(getattr(parameter, "is_quantized", False))
            for parameter in runtime.target_model.parameters()
        )
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Live source state or fresh H776 blank differs from the pinned lane."
        )
    model_fold._assert_target_is_zero_blank(runtime.target_model)
    runtime_policy = runtime.runtime_policy
    model_policy = getattr(runtime_policy, "model_fold_policy", None)
    if (
        model_policy is None
        or not callable(getattr(runtime_policy, "as_dict", None))
        or not callable(getattr(model_policy, "as_dict", None))
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The probe requires one typed complete v5 runtime policy."
        )
    policy_sha256 = _sha256(runtime_policy.as_dict())
    if policy_sha256 != pins.expected_full_policy_sha256:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Live complete v5 runtime policy differs from the k=2 checkpoint policy."
        )
    source_tied_sha = model_fold._tensor_sha256(runtime.source_tied_table)
    if source_tied_sha != pins.expected_source_tied_table_sha256:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Live bridged source table differs from the k=2 boundary context."
        )
    runtime.trace_adapter.verify()
    if (
        getattr(runtime.trace_adapter, "evidence_sha256", None)
        != pins.expected_trace_sha256
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Live target tokenizer trace differs from its invocation pin."
        )
    provider = runtime.evidence_provider
    catalog = provider.catalog
    if (
        catalog.catalog_sha256 != pins.expected_selective_catalog_sha256
        or catalog.source_sha256 != pins.expected_selective_source_sha256
        or catalog.provider_sha256 != pins.expected_selective_provider_sha256
        or catalog.cache_key != pins.expected_cache_sha256
        or catalog.layer_count != H776_LAYER_COUNT
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Selective provider differs from the pinned H776 catalog."
        )

    store = Lfm2CpcfResumeArtifactStore(pins.checkpoint_store_directory)
    store_before = _store_inventory(store.root)
    checkpoint_reference = Lfm2CpcfArtifactReference(
        artifact_kind="checkpoint",
        artifact_key=pins.checkpoint_key,
        manifest_sha256=pins.checkpoint_manifest_sha256,
    )
    checkpoint = store.open_checkpoint(checkpoint_reference)
    checkpoint = store.final_reauthenticate_checkpoint(checkpoint)
    checkpoint_audit = authenticate_h776_k2_checkpoint(store, checkpoint, pins)
    boundary_reference = Lfm2CpcfArtifactReference(
        **checkpoint_audit["boundary_reference"]
    )

    final_payload = None
    layer_payload = None
    result = None
    chart_transport_controller = None
    with resource_monitor:
        resource_monitor.checkpoint("authenticated_inputs_ready")
        try:
            final_payload = provider.load_final_norm_payload()
            final_authentication = provider.authenticate_final_norm_payload(
                final_payload
            )
            phases = v5.build_lfm2_cpcf_v5_target_phase_shells(
                runtime.trace_adapter,
                catalog,
                None,
                final_norm_payload=final_payload,
            )
            final_phases = v5.materialize_lfm2_cpcf_v5_final_norm_phases(
                phases,
                catalog,
                final_payload,
            )
            boundary_compiler = v5._SelectiveBoundaryCompiler(
                phases,
                catalog,
                runtime.source_tied_table,
                target_hidden_width=H776_TARGET_HIDDEN_SIZE,
                model_policy=model_policy,
            )
            boundary_compiler.fit_interface(
                final_phases,
                target_norm_epsilon=model_fold._final_norm_epsilon(
                    runtime.target_model
                ),
            )
            del final_phases
            provider.release_final_norm_payload(final_payload)
            final_payload = None
            resource_monitor.checkpoint("interface_recomputed_final_norm_released")

            boundary = store.open_boundary(boundary_reference)
            boundary = store.final_reauthenticate_boundary(boundary)
            frontier = Lfm2CpcfV5BoundaryFrontier(
                report=boundary.reports,
                tensors=boundary.tensors,
            )
            frontier_audit = authenticate_h776_boundary_frontier(frontier, pins)
            if (
                final_authentication
                != frontier.context["final_norm_authentication_sha256"]
            ):
                raise Lfm2CpcfV5RealAttentionProbeError(
                    "Recomputed final-normalization evidence differs from k=2."
                )
            frontier.hydrate(boundary_compiler)
            phase_audit = authenticate_h776_phase_inventory(
                phases,
                frontier=frontier,
            )
            resource_monitor.checkpoint("boundary_frontier_hydrated")

            install_report = install_lfm2_cpcf_prefix_checkpoint(
                store,
                checkpoint,
                runtime.target_model,
                expected_blank_state_sha256=pins.expected_blank_state_sha256,
            )
            state_bank = reconstruct_lfm2_cpcf_phase_state_bank(
                store,
                checkpoint,
                phases=phases,
            )
            if (
                state_bank.layer_index != H776_LAYER_INDEX
                or state_bank.state_bank_sha256
                != checkpoint_audit["incoming_state_bank_sha256"]
                or install_report["next_layer_index"] != H776_LAYER_INDEX
            ):
                raise Lfm2CpcfV5RealAttentionProbeError(
                    "Installed prefix and reconstructed k=2 state bank diverged."
                )
            resource_monitor.checkpoint("k2_prefix_installed")

            reference = catalog.dynamic_catalog.references[H776_LAYER_INDEX]
            layer_payload = provider.load_layer_payload(reference)
            authentication = provider.authenticate_layer_payload(
                reference,
                layer_payload,
            )
            if (
                layer_payload.layer_index != H776_LAYER_INDEX
                or layer_payload.operator_kind != "full_attention"
                or authentication.evidence_sha256 != reference.evidence_sha256
            ):
                raise Lfm2CpcfV5RealAttentionProbeError(
                    "Resident payload is not authenticated attention layer 2."
                )
            materialized_phases = v5.materialize_lfm2_cpcf_v5_layer_phases(
                phases,
                catalog,
                layer_payload,
            )
            layer_bank = rebind_lfm2_cpcf_phase_state_bank(
                state_bank,
                materialized_phases,
            )
            charts = boundary_compiler.fit_layer(
                materialized_phases,
                layer_index=H776_LAYER_INDEX,
            )
            source_charts_sha256 = charts.layer_charts_sha256
            chart_transport_controller = (
                apply_lfm2_cpcf_post_operator_chart_transport_shadow(
                    layer_bank,
                    charts,
                    enabled=(
                        runtime.post_operator_chart_transport_shadow
                    ),
                )
            )
            if chart_transport_controller is not None:
                charts = chart_transport_controller.boundary_charts
                resource_monitor.checkpoint(
                    "layer_2_post_operator_chart_transport_frozen"
                )
            collector = Lfm2SequenceEffectivePcgCollector()
            resource_monitor.checkpoint("layer_2_evidence_and_charts_ready")
            with sequence_fold.observe_lfm2_sequence_effective_pcg(collector):
                result = compile_and_install_lfm2_cpcf_layer(
                    runtime.source_model,
                    runtime.target_model,
                    layer_index=H776_LAYER_INDEX,
                    input_bank=layer_bank,
                    boundary_charts=charts,
                    ridge=float(model_policy.ridge),
                    short_conv_gate_policy=None,
                    short_conv_v4_solver_policy=None,
                    candidate_replay_evaluator=None,
                    candidate_selector=None,
                    numerical_frontier_adapter=None,
                )
            resource_monitor.checkpoint("layer_2_compiled_and_validated")
            sequence_artifact = result.report["solver_evidence_policy"][
                "attention_candidate_evaluations"
            ][H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]["operator_artifact"]
            pcg_audit = collector.finalize(
                sequence_artifact=sequence_artifact,
            )
            result_audit = authenticate_h776_attention_result(
                result,
                input_bank_sha256=layer_bank.state_bank_sha256,
                pcg_audit=pcg_audit,
                pins=pins,
            )
            if (
                result.report.get("boundary_charts_sha256")
                != charts.layer_charts_sha256
            ):
                raise Lfm2CpcfV5RealAttentionProbeError(
                    "Layer result is not bound to the chart-transport mode."
                )
            provider.release_layer_payload(reference, layer_payload)
            layer_payload = None
            resource_monitor.checkpoint("layer_2_evidence_released")
        finally:
            if layer_payload is not None:
                provider.release_layer_payload(
                    catalog.dynamic_catalog.references[H776_LAYER_INDEX],
                    layer_payload,
                )
            if final_payload is not None:
                provider.release_final_norm_payload(final_payload)

    resource_report = resource_monitor.report()
    store.final_reauthenticate_checkpoint(checkpoint)
    store.final_reauthenticate_boundary(boundary)
    if _store_inventory(store.root) != store_before:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Checkpoint store changed; refusing a hidden k=3/publication path."
        )
    if model_fold._snapshot_model_auth(runtime.source_model) != source_before:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Source model changed during the layer-2 probe."
        )
    assert result is not None
    controller_report = (
        None
        if chart_transport_controller is None
        else chart_transport_controller.as_dict()
    )
    chart_transport_body = {
        "enabled": runtime.post_operator_chart_transport_shadow,
        "mode": (
            "fit_only_post_operator_orthogonal_chart_transport"
            if runtime.post_operator_chart_transport_shadow
            else "off"
        ),
        "source_layer_charts_sha256": source_charts_sha256,
        "compiler_input_layer_charts_sha256": (
            result.report["boundary_charts_sha256"]
        ),
        "controller_report_sha256": (
            None
            if controller_report is None
            else controller_report["report_sha256"]
        ),
        "transport_report_sha256": (
            None
            if controller_report is None
            else controller_report["transport_report_sha256"]
        ),
        "selection_used_by_controller_solver": (
            False if controller_report is not None else None
        ),
        "validation_rows_read_by_controller": (
            0 if controller_report is not None else None
        ),
        "default_off_semantics_preserved": (
            not runtime.post_operator_chart_transport_shadow
        ),
        "controller_report": controller_report,
    }
    chart_transport_binding = {
        **chart_transport_body,
        "binding_sha256": _sha256(chart_transport_body),
    }
    body = {
        "schema": LFM2_CPCF_V5_REAL_ATTENTION_PROBE_SCHEMA,
        "status": (
            "completed_positive_sequence_effective_selection"
            if result_audit["sequence_effective_won"]
            and result_audit[
                "sequence_effective_strictly_better_than_both_controls"
            ]
            else "completed_honest_negative_or_control_selected"
        ),
        "pins": pins.as_dict(),
        "runtime_compiler_implementation": implementation,
        "runtime_sequence_source_sha256": sequence_source_sha256,
        "cross_runtime_source_inventory": {
            "checkpoint_origin_inventory_sha256": (
                pins.expected_source_inventory_sha256
            ),
            "live_process_inventory_sha256": (
                source_before.full_inventory_sha256
            ),
            "origin_digest_reproduction_required": False,
            "reason": (
                "full inventory includes process-local object and storage "
                "identities; stable state bytes are pinned independently"
            ),
            "live_inventory_unchanged_before_after": True,
        },
        "target_geometry": geometry,
        "parameter_compaction": {
            "parent_unique_parameters": H776_PARENT_UNIQUE_PARAMETERS,
            "target_unique_parameters": H776_TARGET_UNIQUE_PARAMETERS,
            "compression_ratio": H776_PARAMETER_COMPRESSION_RATIO,
            "parent_inventory_sha256": (
                source_parameter_inventory["inventory_sha256"]
            ),
            "target_inventory_sha256": target_inventory["inventory_sha256"],
            "quantization_used": False,
            "pruning_used": False,
        },
        "checkpoint_audit": _plain(checkpoint_audit),
        "frontier_audit": _plain(frontier_audit),
        "phase_audit": _plain(phase_audit),
        "prefix_install_report": _plain(install_report),
        "layer_2_result": _plain(result_audit),
        "post_operator_chart_transport_shadow": (
            _plain(chart_transport_binding)
        ),
        "resource_report": _plain(resource_report),
        "stop_conditions": {
            "exactly_one_layer_compiled": True,
            "compiled_layer_indices": [H776_LAYER_INDEX],
            "next_layer_compilation_attempted": False,
            "k3_checkpoint_published": False,
            "child_checkpoint_saved": False,
            "production_publication_attempted": False,
            "ordinary_resume_gate_weakened": False,
            "parent_trace_recapture_used": False,
            "training_used": False,
            "optimizer_used": False,
            "backward_used": False,
            "pruning_used": False,
            "quantization_used": False,
            "automatic_retry_or_retuning_used": False,
        },
        "promotion_authorized": False,
        "accepted_child_artifact": False,
        "whole_model_intelligence_retention_proven": False,
        "non_promotable_probe_only": True,
    }
    return MappingProxyType({**body, "report_sha256": _sha256(body)})


__all__ = [
    "H776_ATTENTION_CANDIDATE_IDS",
    "H776_LAYER_INDEX",
    "H776_MAXIMUM_CUDA_BYTES_PER_DEVICE",
    "H776_MAXIMUM_PROCESS_RSS_BYTES",
    "H776_PARAMETER_COMPRESSION_RATIO",
    "H776_PHASE_COUNT",
    "H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID",
    "H776_TARGET_PLAN_FINGERPRINT",
    "H776_TARGET_TOPOLOGY_SHA256",
    "H776_TARGET_UNIQUE_PARAMETERS",
    "LFM2_CPCF_V5_REAL_ATTENTION_PCG_AUDIT_SCHEMA",
    "LFM2_CPCF_V5_REAL_ATTENTION_PROBE_SCHEMA",
    "Lfm2CpcfV5RealAttentionProbeError",
    "Lfm2CpcfV5RealAttentionProbePins",
    "Lfm2CpcfV5RealAttentionProbeRuntime",
    "Lfm2CpcfV5RealAttentionResourceMonitor",
    "Lfm2SequenceEffectivePcgCollector",
    "authenticate_h776_attention_result",
    "authenticate_h776_boundary_frontier",
    "authenticate_h776_k2_checkpoint",
    "authenticate_h776_phase_inventory",
    "execute_lfm2_cpcf_v5_real_attention_probe",
]
