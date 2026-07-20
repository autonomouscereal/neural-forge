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
LFM2_CPCF_V5_REAL_COMPOSED_ATTENTION_AUDIT_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-real-composed-attention-audit-v1"
)
LFM2_CPCF_V5_REAL_ENDPOINT_RESIDUAL_ATTENTION_AUDIT_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-real-endpoint-residual-attention-audit-v1"
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
H776_COMPOSED_ATTENTION_CANDIDATE_ID = "score_routed_residual_seeded_sequence_effective"
H776_COMPOSED_ATTENTION_CANDIDATE_IDS = (
    *H776_ATTENTION_CANDIDATE_IDS,
    H776_COMPOSED_ATTENTION_CANDIDATE_ID,
)
H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_ID = (
    "endpoint_residual_alternating_qkvo"
)
H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS = (
    *H776_COMPOSED_ATTENTION_CANDIDATE_IDS,
    H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_ID,
)
H776_ENDPOINT_WITHOUT_COMPOSED_ATTENTION_CANDIDATE_IDS = (
    *H776_ATTENTION_CANDIDATE_IDS,
    H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_ID,
)
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
H776_COMPOSED_UPSTREAM_FIT_CASES = 154
H776_COMPOSED_QK_FIT_CASES = 134
H776_COMPOSED_QK_FIT_CHECK_CASES = 20
H776_MAXIMUM_PROCESS_RSS_BYTES = 64 * 1024**3
H776_COMPOSED_MAXIMUM_PROCESS_RSS_BYTES = 192 * 1024**3
H776_MAXIMUM_CUDA_BYTES_PER_DEVICE = 20 * 1024**3
H776_COMPOSED_QK_ATTENTION_POLICY_SHA256 = (
    "a241c7a77990fbc45d20785fbe2c1897818642667cf7ff24b767db8ada1fb0bd"
)
H776_ENDPOINT_RESIDUAL_ATTENTION_POLICY_SHA256 = (
    "8b83650718ec0ce471a29ae67e28fec604f7f6a04e46dd03a12009ff43e09f87"
)
H776_COMPOSED_LOGICAL_CUDA_UUIDS = (
    "2ce6a95c-7b1f-644a-3808-d781732a0582",
    "4ead7f3a-cc79-fa5b-de42-ef024e27f561",
)
H776_COMPOSED_CUDA_DEVICE_NAME = "NVIDIA GeForce RTX 3090"
H776_PCG_MAXIMUM_ITERATIONS = 64
H776_PCG_RESTART_BLOCK_ITERATIONS = 64
H776_PCG_HARD_MAXIMUM_ITERATIONS = 512
H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS = 4096
H776_PCG_ROW_BLOCK_HARD_MAXIMUM_ITERATIONS = 4096
H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL = 1e-8
H776_HIDDEN_PARTIAL_TRACE_CHOLESKY_SCHEMA = (
    "neural-forge-lfm2-sequence-effective-hidden-partial-trace-cholesky-v1"
)
H776_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SCHEMA = (
    "neural-forge-lfm2-sequence-effective-hidden-partial-trace-pcg-execution-profile-v1"
)
H776_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256 = (
    "7f71ffe700643b6fc017c518dd2f853092917d288f8e70b4cb659e941aa5f13e"
)
H776_ROW_BLOCK_JACOBI_SCHEMA = (
    "neural-forge-lfm2-sequence-effective-row-block-jacobi-v1"
)

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
H776_CACHE_SHA256 = "76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95"
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
        raise Lfm2CpcfV5RealAttentionProbeError(f"{name} does not exist.") from exc
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
    expected_composed_qk_attention_policy_sha256: str = (
        H776_COMPOSED_QK_ATTENTION_POLICY_SHA256
    )
    expected_endpoint_residual_attention_policy_sha256: str = (
        H776_ENDPOINT_RESIDUAL_ATTENTION_POLICY_SHA256
    )
    expected_composed_logical_cuda_uuids: tuple[str, str] = (
        H776_COMPOSED_LOGICAL_CUDA_UUIDS
    )
    expected_composed_qk_attention_enabled: bool = False
    expected_endpoint_residual_attention_enabled: bool = False
    layer_index: int = H776_LAYER_INDEX
    expected_target_unique_parameters: int = H776_TARGET_UNIQUE_PARAMETERS
    maximum_process_rss_bytes: int = H776_MAXIMUM_PROCESS_RSS_BYTES
    maximum_cuda_bytes_per_device: int = H776_MAXIMUM_CUDA_BYTES_PER_DEVICE
    require_zero_swap: bool = True
    expected_attention_sequential_transport_enabled: bool = False

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
            "expected_composed_qk_attention_policy_sha256",
            "expected_endpoint_residual_attention_policy_sha256",
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
            "expected_sequence_solver_plan_sha256": (H776_SEQUENCE_SOLVER_PLAN_SHA256),
            "expected_composed_qk_attention_policy_sha256": (
                H776_COMPOSED_QK_ATTENTION_POLICY_SHA256
            ),
            "expected_endpoint_residual_attention_policy_sha256": (
                H776_ENDPOINT_RESIDUAL_ATTENTION_POLICY_SHA256
            ),
            "expected_composed_logical_cuda_uuids": (H776_COMPOSED_LOGICAL_CUDA_UUIDS),
            "layer_index": H776_LAYER_INDEX,
            "expected_target_unique_parameters": H776_TARGET_UNIQUE_PARAMETERS,
            "maximum_process_rss_bytes": (
                H776_COMPOSED_MAXIMUM_PROCESS_RSS_BYTES
                if (
                    self.expected_composed_qk_attention_enabled
                    or self.expected_endpoint_residual_attention_enabled
                )
                else H776_MAXIMUM_PROCESS_RSS_BYTES
            ),
            "maximum_cuda_bytes_per_device": (H776_MAXIMUM_CUDA_BYTES_PER_DEVICE),
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
        if (
            type(self.expected_attention_sequential_transport_enabled) is not bool
            or type(self.expected_composed_qk_attention_enabled) is not bool
            or type(self.expected_endpoint_residual_attention_enabled) is not bool
        ):
            raise TypeError("Expected attention diagnostic modes must be bool.")
        if (
            sum(
                (
                    self.expected_attention_sequential_transport_enabled,
                    self.expected_composed_qk_attention_enabled,
                    self.expected_endpoint_residual_attention_enabled,
                )
            )
            > 1
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Sequential transport, four-arm composed Q/K, and five-arm "
                "endpoint-residual attention are separate authenticated lanes."
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
            "expected_target_plan_fingerprint": (self.expected_target_plan_fingerprint),
            "expected_target_topology_sha256": (self.expected_target_topology_sha256),
            "expected_selective_catalog_sha256": (
                self.expected_selective_catalog_sha256
            ),
            "expected_selective_source_sha256": (self.expected_selective_source_sha256),
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
            "expected_source_inventory_sha256": (self.expected_source_inventory_sha256),
            "expected_blank_report_sha256": self.expected_blank_report_sha256,
            "expected_blank_state_sha256": self.expected_blank_state_sha256,
            "expected_full_policy_sha256": self.expected_full_policy_sha256,
            "expected_origin_compiler_sha256": (self.expected_origin_compiler_sha256),
            "expected_sequence_solver_plan_sha256": (
                self.expected_sequence_solver_plan_sha256
            ),
            "expected_composed_qk_attention_policy_sha256": (
                self.expected_composed_qk_attention_policy_sha256
            ),
            "expected_endpoint_residual_attention_policy_sha256": (
                self.expected_endpoint_residual_attention_policy_sha256
            ),
            "expected_composed_logical_cuda_uuids": list(
                self.expected_composed_logical_cuda_uuids
            ),
            "expected_composed_qk_attention_enabled": (
                self.expected_composed_qk_attention_enabled
            ),
            "expected_endpoint_residual_attention_enabled": (
                self.expected_endpoint_residual_attention_enabled
            ),
            "layer_index": self.layer_index,
            "expected_target_unique_parameters": (
                self.expected_target_unique_parameters
            ),
            "maximum_process_rss_bytes": self.maximum_process_rss_bytes,
            "maximum_cuda_bytes_per_device": (self.maximum_cuda_bytes_per_device),
            "require_zero_swap": self.require_zero_swap,
            "expected_attention_sequential_transport_enabled": (
                self.expected_attention_sequential_transport_enabled
            ),
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
    attention_sequential_transport_enabled: bool = False
    composed_qk_attention_enabled: bool = False
    endpoint_residual_attention_enabled: bool = False


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
        or receipts[-1].get("next_bank_sha256") != incoming.get("state_bank_sha256")
        or receipts[1].get("input_bank_sha256") != receipts[0].get("next_bank_sha256")
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Checkpoint is not the exact contiguous short-conv k=2 causal prefix."
        )
    for receipt in receipts:
        execution = receipt.get("execution")
        if (
            not isinstance(execution, dict)
            or execution.get("closed_form_compilers") is not True
            or any(
                execution.get(name) != 0
                for name in execution
                if name != "closed_form_compilers"
            )
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "A completed prefix receipt does not prove fold-only execution."
            )
    topology = completed.get("target_topology_snapshot")
    if (
        not isinstance(topology, dict)
        or topology.get("model_type") != "lfm2"
        or topology.get("layer_count") != H776_LAYER_COUNT
        or topology.get("topology_sha256") != pins.expected_target_topology_sha256
        or completed.get("target_topology_sha256")
        != pins.expected_target_topology_sha256
        or bindings.get("target_topology_sha256")
        != pins.expected_target_topology_sha256
        or bindings.get("full_policy_sha256") != pins.expected_full_policy_sha256
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
        or any(
            len(names) != H776_PHASES_PER_TRAJECTORY for names in trajectories.values()
        )
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


def _finite_nonnegative_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _authenticate_legacy_pcg_report(report: Mapping[str, Any]) -> None:
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
    iterations = report.get("iterations_executed")
    initial = report.get("initial_relative_normal_residual")
    final = report.get("final_relative_normal_residual")
    history = report.get("relative_normal_residual_history")
    if (
        set(report) != report_fields
        or report.get("solver") != "deterministic_jacobi_pcg_single_tensor"
        or report.get("maximum_iterations") != H776_PCG_MAXIMUM_ITERATIONS
        or isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not 0 <= iterations <= H776_PCG_MAXIMUM_ITERATIONS
        or report.get("required_relative_normal_residual")
        != H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
        or not _finite_nonnegative_number(initial)
        or not _finite_nonnegative_number(final)
        or float(final) > H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
        or not isinstance(history, list)
        or len(history) != iterations
        or any(not _finite_nonnegative_number(item) for item in history)
        or (iterations == 0 and (history or float(initial) != float(final)))
        or (iterations > 0 and float(history[-1]) != float(final))
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "A sequence-effective PCG solve missed its convergence gate."
        )
    _digest(report.get("solution_sha256"), "PCG solution SHA-256")


def _authenticate_restarted_pcg_report(report: Mapping[str, Any]) -> None:
    report_fields = {
        "role",
        "solver",
        "maximum_iterations",
        "original_lane_maximum_iterations",
        "original_lane_iterations_executed",
        "restart_block_iterations",
        "iterations_executed",
        "required_relative_normal_residual",
        "initial_relative_normal_residual",
        "original_lane_final_recurrence_relative_normal_residual",
        "final_recurrence_relative_normal_residual",
        "final_relative_normal_residual",
        "relative_normal_residual_history",
        "recurrence_history_sha256",
        "exact_relative_normal_residual_checkpoints",
        "exact_residual_checkpoints_sha256",
        "restart_schedule",
        "restart_schedule_sha256",
        "total_matvec_calls",
        "exact_final_residual_authenticated",
        "extended_after_original_lane_exact_authentication_miss",
        "solution_finite",
        "solution_sha256",
        "report_sha256",
    }
    iterations = report.get("iterations_executed")
    initial = report.get("initial_relative_normal_residual")
    original_final = report.get(
        "original_lane_final_recurrence_relative_normal_residual"
    )
    original_lane_iterations = report.get("original_lane_iterations_executed")
    recurrence_final = report.get("final_recurrence_relative_normal_residual")
    exact_final = report.get("final_relative_normal_residual")
    history = report.get("relative_normal_residual_history")
    checkpoints = report.get("exact_relative_normal_residual_checkpoints")
    schedule = report.get("restart_schedule")
    total_matvec_calls = report.get("total_matvec_calls")
    if (
        set(report) != report_fields
        or report.get("solver") != "deterministic_restarted_jacobi_pcg_single_tensor"
        or report.get("maximum_iterations") != H776_PCG_HARD_MAXIMUM_ITERATIONS
        or report.get("original_lane_maximum_iterations") != H776_PCG_MAXIMUM_ITERATIONS
        or isinstance(original_lane_iterations, bool)
        or not isinstance(original_lane_iterations, int)
        or not (1 <= original_lane_iterations <= H776_PCG_MAXIMUM_ITERATIONS)
        or report.get("restart_block_iterations") != H776_PCG_RESTART_BLOCK_ITERATIONS
        or isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not (1 <= iterations <= H776_PCG_HARD_MAXIMUM_ITERATIONS)
        or report.get("required_relative_normal_residual")
        != H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
        or not _finite_nonnegative_number(initial)
        or not _finite_nonnegative_number(original_final)
        or not _finite_nonnegative_number(recurrence_final)
        or not _finite_nonnegative_number(exact_final)
        or float(exact_final) > H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
        or not isinstance(history, list)
        or len(history) != iterations
        or any(not _finite_nonnegative_number(item) for item in history)
        or float(history[-1]) != float(recurrence_final)
        or report.get("recurrence_history_sha256") != _sha256(history)
        or not isinstance(checkpoints, list)
        or not checkpoints
        or report.get("exact_residual_checkpoints_sha256") != _sha256(checkpoints)
        or not isinstance(schedule, list)
        or report.get("restart_schedule_sha256") != _sha256(schedule)
        or isinstance(total_matvec_calls, bool)
        or not isinstance(total_matvec_calls, int)
        or total_matvec_calls != iterations + 2 + len(schedule)
        or report.get("exact_final_residual_authenticated") is not True
        or report.get("extended_after_original_lane_exact_authentication_miss")
        is not True
        or report.get("solution_finite") is not True
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "A restarted sequence-effective PCG solve lacks exact convergence evidence."
        )

    checkpoint_by_iteration: dict[int, float] = {}
    previous_iteration = -1
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "iteration",
            "relative_normal_residual",
            "finite",
        }:
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Restarted PCG exact residual checkpoints are malformed."
            )
        iteration = checkpoint.get("iteration")
        residual = checkpoint.get("relative_normal_residual")
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration <= previous_iteration
            or not (1 <= iteration <= iterations)
            or not _finite_nonnegative_number(residual)
            or checkpoint.get("finite") is not True
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Restarted PCG exact residual checkpoints are inconsistent."
            )
        checkpoint_by_iteration[iteration] = float(residual)
        previous_iteration = iteration
    if (
        checkpoints[0]["iteration"] != original_lane_iterations
        or float(history[original_lane_iterations - 1]) != float(original_final)
        or checkpoints[-1]["iteration"] != iterations
        or float(checkpoints[-1]["relative_normal_residual"]) != float(exact_final)
        or len(checkpoints) != len(schedule) + 1
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Restarted PCG exact residual inventory is incomplete."
        )

    expected_restart = original_lane_iterations
    schedule_fields = {
        "restart_iteration",
        "planned_stop_iteration",
        "iterations_executed",
        "ending_iteration",
        "attempted_matvec_calls",
        "termination",
        "ending_recurrence_relative_normal_residual",
        "ending_exact_relative_normal_residual",
        "ending_exact_residual_authenticated",
    }
    for block in schedule:
        if not isinstance(block, dict) or set(block) != schedule_fields:
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Restarted PCG schedule is malformed."
            )
        restart = block.get("restart_iteration")
        planned_stop = block.get("planned_stop_iteration")
        executed = block.get("iterations_executed")
        ending = block.get("ending_iteration")
        attempted_matvec_calls = block.get("attempted_matvec_calls")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (
                    restart,
                    planned_stop,
                    executed,
                    ending,
                    attempted_matvec_calls,
                )
            )
            or restart != expected_restart
            or planned_stop
            != min(
                restart + H776_PCG_RESTART_BLOCK_ITERATIONS,
                H776_PCG_HARD_MAXIMUM_ITERATIONS,
            )
            or not 0 < executed <= H776_PCG_RESTART_BLOCK_ITERATIONS
            or attempted_matvec_calls != executed + 1
            or ending != restart + executed
            or ending > planned_stop
            or ending not in checkpoint_by_iteration
            or not _finite_nonnegative_number(
                block.get("ending_recurrence_relative_normal_residual")
            )
            or float(block["ending_recurrence_relative_normal_residual"])
            != float(history[ending - 1])
            or not _finite_nonnegative_number(
                block.get("ending_exact_relative_normal_residual")
            )
            or float(block["ending_exact_relative_normal_residual"])
            != checkpoint_by_iteration[ending]
            or block.get("ending_exact_residual_authenticated") is not True
            or block.get("termination")
            not in {"block_boundary", "exact_tolerance_reached"}
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Restarted PCG schedule does not match its residual history."
            )
        expected_restart = ending
    if (
        (not schedule and iterations != original_lane_iterations)
        or (schedule and expected_restart != iterations)
        or (
            schedule
            and float(checkpoints[0]["relative_normal_residual"])
            <= H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
        )
        or (
            not schedule
            and float(checkpoints[0]["relative_normal_residual"])
            > H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
        )
        or (
            schedule
            and any(block["termination"] != "block_boundary" for block in schedule[:-1])
        )
        or (
            schedule
            and any(
                block["ending_iteration"] != block["planned_stop_iteration"]
                for block in schedule[:-1]
            )
        )
        or any(
            float(checkpoint["relative_normal_residual"])
            <= H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
            for checkpoint in checkpoints[:-1]
        )
        or (schedule and schedule[-1]["termination"] != "exact_tolerance_reached")
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Restarted PCG schedule does not reach its reported final iteration."
        )
    _digest(report.get("solution_sha256"), "PCG solution SHA-256")


def _authenticate_h776_row_block_jacobi_receipt(
    value: Any,
    *,
    role: str,
) -> None:
    receipt_fields = {
        "kind",
        "construction_report",
        "construction_report_sha256",
        "row_blocks_sha256",
        "factorization",
        "factor_sha256",
        "row_block_count",
        "row_block_size",
        "all_row_blocks_spd",
        "factorization_materialized_only_after_original_lane_exact_miss",
        "full_operator_materialized",
        "report_sha256",
    }
    construction_fields = {
        "schema",
        "role",
        "construction",
        "row_block_count",
        "row_block_size",
        "row_blocks_shape",
        "row_blocks_dtype",
        "row_blocks_sha256",
        "scalar_diagonal_sha256",
        "scalar_diagonal_maximum_absolute_error",
        "scalar_diagonal_match_tolerance",
        "row_blocks_maximum_asymmetry",
        "row_blocks_symmetry_tolerance",
        "source_term_count",
        "source_terms_sha256",
        "ridge",
        "full_operator_materialized",
        "report_sha256",
    }
    receipt = _self_sealed(
        value,
        seal_field="report_sha256",
        name="row-block Jacobi factorization receipt",
    )
    construction_value = receipt.get("construction_report")
    construction = _self_sealed(
        construction_value,
        seal_field="report_sha256",
        name="row-block Jacobi construction report",
    )
    diagonal_error = construction.get("scalar_diagonal_maximum_absolute_error")
    diagonal_tolerance = construction.get("scalar_diagonal_match_tolerance")
    asymmetry = construction.get("row_blocks_maximum_asymmetry")
    symmetry_tolerance = construction.get("row_blocks_symmetry_tolerance")
    ridge = construction.get("ridge")
    expected_source_terms = (
        H776_TARGET_QUERY_HEADS // H776_TARGET_KEY_VALUE_HEADS
    ) ** 2
    if (
        set(receipt) != receipt_fields
        or set(construction) != construction_fields
        or construction.get("schema") != H776_ROW_BLOCK_JACOBI_SCHEMA
        or construction.get("role") != role
        or construction.get("construction") != "exact_structured_operator_row_blocks"
        or construction.get("row_block_count") != H776_TARGET_HIDDEN_SIZE
        or construction.get("row_block_size") != H776_HEAD_DIM
        or construction.get("row_blocks_shape")
        != [H776_TARGET_HIDDEN_SIZE, H776_HEAD_DIM, H776_HEAD_DIM]
        or construction.get("row_blocks_dtype") != "torch.float64"
        or construction.get("source_term_count") != expected_source_terms
        or not _finite_nonnegative_number(diagonal_error)
        or not _finite_nonnegative_number(diagonal_tolerance)
        or float(diagonal_tolerance) <= 0.0
        or float(diagonal_error) > float(diagonal_tolerance)
        or not _finite_nonnegative_number(asymmetry)
        or not _finite_nonnegative_number(symmetry_tolerance)
        or float(symmetry_tolerance) <= 0.0
        or float(asymmetry) > float(symmetry_tolerance)
        or not _finite_nonnegative_number(ridge)
        or float(ridge) <= 0.0
        or construction.get("full_operator_materialized") is not False
        or receipt.get("kind") != "deterministic_exact_row_block_jacobi_cholesky"
        or receipt.get("construction_report_sha256") != construction["report_sha256"]
        or receipt.get("row_blocks_sha256") != construction.get("row_blocks_sha256")
        or receipt.get("factorization") != "batched_lower_cholesky_ex"
        or receipt.get("row_block_count") != H776_TARGET_HIDDEN_SIZE
        or receipt.get("row_block_size") != H776_HEAD_DIM
        or receipt.get("all_row_blocks_spd") is not True
        or receipt.get("factorization_materialized_only_after_original_lane_exact_miss")
        is not True
        or receipt.get("full_operator_materialized") is not False
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The continuous PCG row-block Jacobi receipt is not the exact "
            "authenticated H776 SPD preconditioner."
        )
    for field, name in (
        ("row_blocks_sha256", "row-block tensor SHA-256"),
        ("scalar_diagonal_sha256", "scalar diagonal SHA-256"),
        ("source_terms_sha256", "row-block source terms SHA-256"),
    ):
        _digest(construction.get(field), name)
    _digest(receipt.get("factor_sha256"), "row-block Cholesky factor SHA-256")


def _authenticate_continuous_row_block_pcg_report(
    report: Mapping[str, Any],
) -> None:
    report_fields = {
        "role",
        "solver",
        "maximum_iterations",
        "hard_cap_includes_original_lane_iterations",
        "original_lane_maximum_iterations",
        "original_lane_iterations_executed",
        "single_restart_after_original_lane_exact_authentication_miss",
        "periodic_restart_count",
        "iterations_executed",
        "continuous_row_block_lane_iterations_executed",
        "required_relative_normal_residual",
        "initial_relative_normal_residual",
        "original_lane_final_recurrence_relative_normal_residual",
        "final_recurrence_relative_normal_residual",
        "final_relative_normal_residual",
        "relative_normal_residual_history",
        "recurrence_history_sha256",
        "exact_relative_normal_residual_checkpoints",
        "exact_residual_checkpoints_sha256",
        "total_matvec_calls",
        "exact_final_residual_authenticated",
        "extended_after_original_lane_exact_authentication_miss",
        "exact_residual_used_to_initialize_continuous_lane_once",
        "exact_diagnostic_matvecs_modify_recurrence_state_or_direction",
        "continuous_row_block_lane",
        "restart_preconditioner",
        "solution_finite",
        "solution_sha256",
        "report_sha256",
    }
    lane_fields = {
        "start_iteration",
        "planned_stop_iteration",
        "iterations_executed",
        "attempted_matvec_calls",
        "exact_diagnostic_matvec_calls",
        "termination",
        "ending_iteration",
        "ending_recurrence_relative_normal_residual",
        "ending_exact_relative_normal_residual",
        "ending_exact_residual_authenticated",
    }
    preconditioner_fields = {
        "kind",
        "supplied",
        "authentication_attempted",
        "authenticated",
        "applied",
        "application_attempt_count",
        "application_count",
        "receipt",
        "authentication_failure_reason",
    }
    role = report.get("role")
    iterations = report.get("iterations_executed")
    original_iterations = report.get("original_lane_iterations_executed")
    block_iterations = report.get("continuous_row_block_lane_iterations_executed")
    initial = report.get("initial_relative_normal_residual")
    original_final = report.get(
        "original_lane_final_recurrence_relative_normal_residual"
    )
    recurrence_final = report.get("final_recurrence_relative_normal_residual")
    exact_final = report.get("final_relative_normal_residual")
    history = report.get("relative_normal_residual_history")
    checkpoints = report.get("exact_relative_normal_residual_checkpoints")
    total_matvec_calls = report.get("total_matvec_calls")
    lane = report.get("continuous_row_block_lane")
    preconditioner = report.get("restart_preconditioner")
    if (
        set(report) != report_fields
        or report.get("solver")
        != "deterministic_continuous_row_block_jacobi_pcg_single_tensor"
        or report.get("maximum_iterations")
        != H776_PCG_ROW_BLOCK_HARD_MAXIMUM_ITERATIONS
        or report.get("hard_cap_includes_original_lane_iterations") is not True
        or report.get("original_lane_maximum_iterations") != H776_PCG_MAXIMUM_ITERATIONS
        or original_iterations != H776_PCG_MAXIMUM_ITERATIONS
        or isinstance(original_iterations, bool)
        or report.get("single_restart_after_original_lane_exact_authentication_miss")
        is not True
        or report.get("periodic_restart_count") != 0
        or isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not (
            H776_PCG_MAXIMUM_ITERATIONS
            < iterations
            <= H776_PCG_ROW_BLOCK_HARD_MAXIMUM_ITERATIONS
        )
        or isinstance(block_iterations, bool)
        or not isinstance(block_iterations, int)
        or block_iterations != iterations - original_iterations
        or block_iterations <= 0
        or report.get("required_relative_normal_residual")
        != H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
        or not _finite_nonnegative_number(initial)
        or not _finite_nonnegative_number(original_final)
        or not _finite_nonnegative_number(recurrence_final)
        or not _finite_nonnegative_number(exact_final)
        or float(exact_final) > H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
        or not isinstance(history, list)
        or len(history) != iterations
        or any(not _finite_nonnegative_number(item) for item in history)
        or float(history[original_iterations - 1]) != float(original_final)
        or float(history[-1]) != float(recurrence_final)
        or report.get("recurrence_history_sha256") != _sha256(history)
        or not isinstance(checkpoints, list)
        or len(checkpoints) < 2
        or report.get("exact_residual_checkpoints_sha256") != _sha256(checkpoints)
        or isinstance(total_matvec_calls, bool)
        or not isinstance(total_matvec_calls, int)
        or report.get("exact_final_residual_authenticated") is not True
        or report.get("extended_after_original_lane_exact_authentication_miss")
        is not True
        or report.get("exact_residual_used_to_initialize_continuous_lane_once")
        is not True
        or report.get("exact_diagnostic_matvecs_modify_recurrence_state_or_direction")
        is not False
        or not isinstance(lane, dict)
        or set(lane) != lane_fields
        or not isinstance(preconditioner, dict)
        or set(preconditioner) != preconditioner_fields
        or report.get("solution_finite") is not True
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "A continuous row-block sequence-effective PCG solve lacks its "
            "exact H776 convergence evidence."
        )

    checkpoint_by_iteration: dict[int, float] = {}
    previous_iteration = -1
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "iteration",
            "relative_normal_residual",
            "finite",
        }:
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Continuous row-block PCG exact checkpoints are malformed."
            )
        iteration = checkpoint.get("iteration")
        residual = checkpoint.get("relative_normal_residual")
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration <= previous_iteration
            or not original_iterations <= iteration <= iterations
            or not _finite_nonnegative_number(residual)
            or checkpoint.get("finite") is not True
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Continuous row-block PCG exact checkpoints are inconsistent."
            )
        checkpoint_by_iteration[iteration] = float(residual)
        previous_iteration = iteration
    if (
        checkpoints[0]["iteration"] != original_iterations
        or float(checkpoints[0]["relative_normal_residual"])
        <= H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
        or checkpoints[-1]["iteration"] != iterations
        or float(checkpoints[-1]["relative_normal_residual"]) != float(exact_final)
        or any(
            float(checkpoint["relative_normal_residual"])
            <= H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
            for checkpoint in checkpoints[:-1]
        )
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Continuous row-block PCG checkpoints do not prove one scalar "
            "lane miss followed by one successful block lane."
        )

    exact_diagnostic_count = len(checkpoints) - 1
    lane_attempted_matvecs = block_iterations + exact_diagnostic_count
    termination = lane.get("termination")
    if (
        lane.get("start_iteration") != original_iterations
        or lane.get("planned_stop_iteration")
        != H776_PCG_ROW_BLOCK_HARD_MAXIMUM_ITERATIONS
        or lane.get("iterations_executed") != block_iterations
        or lane.get("attempted_matvec_calls") != lane_attempted_matvecs
        or lane.get("exact_diagnostic_matvec_calls") != exact_diagnostic_count
        or lane.get("ending_iteration") != iterations
        or not _finite_nonnegative_number(
            lane.get("ending_recurrence_relative_normal_residual")
        )
        or float(lane["ending_recurrence_relative_normal_residual"])
        != float(recurrence_final)
        or not _finite_nonnegative_number(
            lane.get("ending_exact_relative_normal_residual")
        )
        or float(lane["ending_exact_relative_normal_residual"]) != float(exact_final)
        or lane.get("ending_exact_residual_authenticated") is not True
        or termination
        not in {
            "exact_tolerance_reached",
            "exact_tolerance_reached_at_hard_cap",
        }
        or (
            termination == "exact_tolerance_reached"
            and (
                iterations >= H776_PCG_ROW_BLOCK_HARD_MAXIMUM_ITERATIONS
                or float(history[-1]) > H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
            )
        )
        or (
            termination == "exact_tolerance_reached_at_hard_cap"
            and iterations != H776_PCG_ROW_BLOCK_HARD_MAXIMUM_ITERATIONS
        )
        or any(
            float(history[checkpoint["iteration"] - 1])
            > H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
            for checkpoint in checkpoints[1:-1]
        )
        or total_matvec_calls != iterations + 1 + len(checkpoints)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The continuous row-block PCG lane counters or exact-residual "
            "gates are inconsistent."
        )

    if (
        preconditioner.get("kind") != "deterministic_exact_row_block_jacobi_cholesky"
        or preconditioner.get("supplied") is not True
        or preconditioner.get("authentication_attempted") is not True
        or preconditioner.get("authenticated") is not True
        or preconditioner.get("applied") is not True
        or preconditioner.get("application_attempt_count") != block_iterations
        or preconditioner.get("application_count") != block_iterations
        or preconditioner.get("authentication_failure_reason") is not None
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The continuous row-block PCG preconditioner lifecycle is inconsistent."
        )
    _authenticate_h776_row_block_jacobi_receipt(
        preconditioner.get("receipt"),
        role=str(role),
    )
    _digest(report.get("solution_sha256"), "PCG solution SHA-256")


def _authenticate_h776_hidden_partial_trace_receipt(
    value: Any,
    *,
    role: str,
) -> None:
    receipt_fields = {
        "kind",
        "construction_report",
        "construction_report_sha256",
        "partial_trace_sha256",
        "factorization",
        "factorization_info",
        "factor_sha256",
        "factor_shape",
        "factor_dtype",
        "factor_device",
        "hidden_size",
        "head_dim",
        "partial_trace_spd",
        "positive_finite_factor_diagonal",
        "inverse_orientation_probe_relative_residual",
        "inverse_orientation_probe_tolerance",
        "factorization_materialized_only_after_original_lane_exact_miss",
        "full_operator_materialized",
        "report_sha256",
    }
    construction_fields = {
        "schema",
        "role",
        "construction",
        "formula",
        "ridge",
        "hidden_size",
        "head_dim",
        "partial_trace_shape",
        "partial_trace_dtype",
        "partial_trace_device",
        "partial_trace_sha256",
        "partial_trace_tensor_bytes",
        "source_term_count",
        "source_terms",
        "source_terms_sha256",
        "raw_maximum_absolute_asymmetry",
        "symmetry_tolerance",
        "symmetric_within_numerical_tolerance",
        "symmetrization",
        "full_operator_materialized",
        "report_sha256",
    }
    receipt = _self_sealed(
        value,
        seal_field="report_sha256",
        name="hidden partial-trace factorization receipt",
    )
    construction_value = receipt.get("construction_report")
    construction = _self_sealed(
        construction_value,
        seal_field="report_sha256",
        name="hidden partial-trace construction report",
    )
    asymmetry = construction.get("raw_maximum_absolute_asymmetry")
    symmetry_tolerance = construction.get("symmetry_tolerance")
    ridge = construction.get("ridge")
    inverse_probe_residual = receipt.get("inverse_orientation_probe_relative_residual")
    inverse_probe_tolerance = receipt.get("inverse_orientation_probe_tolerance")
    source_terms = construction.get("source_terms")
    expected_source_terms = (
        H776_TARGET_QUERY_HEADS // H776_TARGET_KEY_VALUE_HEADS
    ) ** 2
    if (
        set(receipt) != receipt_fields
        or set(construction) != construction_fields
        or construction.get("schema") != H776_HIDDEN_PARTIAL_TRACE_CHOLESKY_SCHEMA
        or construction.get("role") != role
        or construction.get("construction") != "normalized_hidden_side_partial_trace"
        or construction.get("formula")
        != "P_H=ridge*I_H+sum_(h,o)(trace(C_o,h)/head_dim)*A_h,o"
        or construction.get("hidden_size") != H776_TARGET_HIDDEN_SIZE
        or construction.get("head_dim") != H776_HEAD_DIM
        or construction.get("partial_trace_shape")
        != [H776_TARGET_HIDDEN_SIZE, H776_TARGET_HIDDEN_SIZE]
        or construction.get("partial_trace_dtype") != "torch.float64"
        or not isinstance(construction.get("partial_trace_device"), str)
        or not construction["partial_trace_device"]
        or construction.get("partial_trace_tensor_bytes")
        != H776_TARGET_HIDDEN_SIZE * H776_TARGET_HIDDEN_SIZE * 8
        or construction.get("source_term_count") != expected_source_terms
        or not isinstance(source_terms, list)
        or len(source_terms) != expected_source_terms
        or construction.get("source_terms_sha256") != _sha256(source_terms)
        or not _finite_nonnegative_number(asymmetry)
        or not _finite_nonnegative_number(symmetry_tolerance)
        or float(symmetry_tolerance) <= 0.0
        or float(asymmetry) > float(symmetry_tolerance)
        or construction.get("symmetric_within_numerical_tolerance") is not True
        or construction.get("symmetrization") != "half_sum_with_transpose"
        or not _finite_nonnegative_number(ridge)
        or float(ridge) <= 0.0
        or construction.get("full_operator_materialized") is not False
        or receipt.get("kind") != "deterministic_hidden_partial_trace_cholesky"
        or receipt.get("construction_report_sha256") != construction["report_sha256"]
        or receipt.get("partial_trace_sha256")
        != construction.get("partial_trace_sha256")
        or receipt.get("factorization") != "lower_cholesky_ex"
        or receipt.get("factorization_info") != 0
        or receipt.get("factor_shape")
        != [H776_TARGET_HIDDEN_SIZE, H776_TARGET_HIDDEN_SIZE]
        or receipt.get("factor_dtype") != "torch.float64"
        or receipt.get("factor_device") != construction.get("partial_trace_device")
        or receipt.get("hidden_size") != H776_TARGET_HIDDEN_SIZE
        or receipt.get("head_dim") != H776_HEAD_DIM
        or receipt.get("partial_trace_spd") is not True
        or receipt.get("positive_finite_factor_diagonal") is not True
        or not _finite_nonnegative_number(inverse_probe_residual)
        or not _finite_nonnegative_number(inverse_probe_tolerance)
        or float(inverse_probe_tolerance) <= 0.0
        or float(inverse_probe_residual) > float(inverse_probe_tolerance)
        or receipt.get("factorization_materialized_only_after_original_lane_exact_miss")
        is not True
        or receipt.get("full_operator_materialized") is not False
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The continuous PCG hidden partial-trace receipt is not the exact "
            "authenticated H776 SPD preconditioner."
        )
    seen_pairs: set[tuple[int, int]] = set()
    for term in source_terms:
        if (
            not isinstance(term, dict)
            or set(term)
            != {
                "head",
                "other_head",
                "left_sha256",
                "coupling_sha256",
                "partial_trace_weight",
            }
            or isinstance(term.get("head"), bool)
            or not isinstance(term.get("head"), int)
            or isinstance(term.get("other_head"), bool)
            or not isinstance(term.get("other_head"), int)
            or not 0 <= term["head"] < H776_TARGET_QUERY_HEADS
            or not 0 <= term["other_head"] < H776_TARGET_QUERY_HEADS
            or (term["head"], term["other_head"]) in seen_pairs
            or not _finite_number(term.get("partial_trace_weight"))
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "The H776 hidden partial-trace source inventory is invalid."
            )
        seen_pairs.add((term["head"], term["other_head"]))
        _digest(term.get("left_sha256"), "partial-trace left tensor SHA-256")
        _digest(
            term.get("coupling_sha256"),
            "partial-trace coupling tensor SHA-256",
        )
    role_match = _STRUCTURED_VALUE_PCG_ROLE.fullmatch(role)
    if role_match is None:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The hidden partial-trace receipt role is not an H776 group solve."
        )
    group = int(role_match.group(2))
    first_head = group * (H776_TARGET_QUERY_HEADS // H776_TARGET_KEY_VALUE_HEADS)
    group_heads = (first_head, first_head + 1)
    expected_pairs = {
        (head, other_head) for head in group_heads for other_head in group_heads
    }
    if seen_pairs != expected_pairs:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The H776 hidden partial-trace source pairs differ from the "
            "immutable query-head group."
        )
    for field, name in (
        ("partial_trace_sha256", "hidden partial-trace tensor SHA-256"),
        ("source_terms_sha256", "hidden partial-trace source terms SHA-256"),
    ):
        _digest(construction.get(field), name)
    _digest(
        receipt.get("factor_sha256"),
        "hidden partial-trace Cholesky factor SHA-256",
    )


def _authenticate_h776_hidden_partial_trace_pcg_execution_profile(
    value: Any,
) -> None:
    profile_fields = {
        "schema",
        "original_lane_solver",
        "original_lane_maximum_iterations",
        "continuation_lane_solver",
        "continuation_preconditioner_schema",
        "continuation_preconditioner_formula",
        "single_restart_after_original_lane_exact_authentication_miss",
        "periodic_restart_count",
        "hard_maximum_iterations",
        "required_relative_normal_residual",
        "exact_residual_used_to_initialize_continuous_lane_once",
        "exact_diagnostic_matvecs_modify_recurrence_state_or_direction",
        "factorization_materialized_only_after_original_lane_exact_miss",
        "production_row_block_preconditioner_selected",
        "full_operator_materialized",
        "report_sha256",
    }
    profile = _self_sealed(
        value,
        seal_field="report_sha256",
        name="hidden partial-trace PCG execution profile",
    )
    if (
        set(profile) != profile_fields
        or profile.get("schema")
        != H776_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SCHEMA
        or profile.get("original_lane_solver")
        != "deterministic_jacobi_pcg_single_tensor"
        or profile.get("original_lane_maximum_iterations")
        != H776_PCG_MAXIMUM_ITERATIONS
        or profile.get("continuation_lane_solver")
        != ("deterministic_continuous_hidden_partial_trace_cholesky_pcg_single_tensor")
        or profile.get("continuation_preconditioner_schema")
        != H776_HIDDEN_PARTIAL_TRACE_CHOLESKY_SCHEMA
        or profile.get("continuation_preconditioner_formula")
        != "P_H=ridge*I_H+sum_(h,o)(trace(C_o,h)/head_dim)*A_h,o"
        or profile.get("single_restart_after_original_lane_exact_authentication_miss")
        is not True
        or profile.get("periodic_restart_count") != 0
        or profile.get("hard_maximum_iterations")
        != H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS
        or profile.get("required_relative_normal_residual")
        != H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
        or profile.get("exact_residual_used_to_initialize_continuous_lane_once")
        is not True
        or profile.get("exact_diagnostic_matvecs_modify_recurrence_state_or_direction")
        is not False
        or profile.get("factorization_materialized_only_after_original_lane_exact_miss")
        is not True
        or profile.get("production_row_block_preconditioner_selected") is not False
        or profile.get("full_operator_materialized") is not False
        or profile.get("report_sha256")
        != H776_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The hidden partial-trace PCG execution profile differs from "
            "the immutable H776 continuation pin."
        )


def _authenticate_continuous_hidden_partial_trace_pcg_report(
    report: Mapping[str, Any],
) -> None:
    report_fields = {
        "role",
        "solver",
        "maximum_iterations",
        "hard_cap_includes_original_lane_iterations",
        "original_lane_maximum_iterations",
        "original_lane_iterations_executed",
        "single_restart_after_original_lane_exact_authentication_miss",
        "periodic_restart_count",
        "iterations_executed",
        "continuous_hidden_partial_trace_lane_iterations_executed",
        "required_relative_normal_residual",
        "initial_relative_normal_residual",
        "original_lane_final_recurrence_relative_normal_residual",
        "final_recurrence_relative_normal_residual",
        "final_relative_normal_residual",
        "relative_normal_residual_history",
        "recurrence_history_sha256",
        "exact_relative_normal_residual_checkpoints",
        "exact_residual_checkpoints_sha256",
        "total_matvec_calls",
        "exact_final_residual_authenticated",
        "extended_after_original_lane_exact_authentication_miss",
        "exact_residual_used_to_initialize_continuous_lane_once",
        "exact_diagnostic_matvecs_modify_recurrence_state_or_direction",
        "execution_profile",
        "execution_profile_sha256",
        "continuous_hidden_partial_trace_lane",
        "restart_preconditioner",
        "solution_finite",
        "solution_sha256",
        "report_sha256",
    }
    lane_fields = {
        "start_iteration",
        "planned_stop_iteration",
        "iterations_executed",
        "attempted_matvec_calls",
        "exact_diagnostic_matvec_calls",
        "termination",
        "ending_iteration",
        "ending_recurrence_relative_normal_residual",
        "ending_exact_relative_normal_residual",
        "ending_exact_residual_authenticated",
    }
    preconditioner_fields = {
        "kind",
        "supplied",
        "authentication_attempted",
        "authenticated",
        "applied",
        "application_attempt_count",
        "application_count",
        "receipt",
        "authentication_failure_reason",
    }
    role = report.get("role")
    iterations = report.get("iterations_executed")
    original_iterations = report.get("original_lane_iterations_executed")
    partial_trace_iterations = report.get(
        "continuous_hidden_partial_trace_lane_iterations_executed"
    )
    initial = report.get("initial_relative_normal_residual")
    original_final = report.get(
        "original_lane_final_recurrence_relative_normal_residual"
    )
    recurrence_final = report.get("final_recurrence_relative_normal_residual")
    exact_final = report.get("final_relative_normal_residual")
    history = report.get("relative_normal_residual_history")
    checkpoints = report.get("exact_relative_normal_residual_checkpoints")
    total_matvec_calls = report.get("total_matvec_calls")
    lane = report.get("continuous_hidden_partial_trace_lane")
    preconditioner = report.get("restart_preconditioner")
    if (
        set(report) != report_fields
        or report.get("solver")
        != ("deterministic_continuous_hidden_partial_trace_cholesky_pcg_single_tensor")
        or report.get("maximum_iterations")
        != H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS
        or report.get("hard_cap_includes_original_lane_iterations") is not True
        or report.get("original_lane_maximum_iterations") != H776_PCG_MAXIMUM_ITERATIONS
        or original_iterations != H776_PCG_MAXIMUM_ITERATIONS
        or isinstance(original_iterations, bool)
        or report.get("single_restart_after_original_lane_exact_authentication_miss")
        is not True
        or report.get("periodic_restart_count") != 0
        or isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not (
            H776_PCG_MAXIMUM_ITERATIONS
            < iterations
            <= H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS
        )
        or isinstance(partial_trace_iterations, bool)
        or not isinstance(partial_trace_iterations, int)
        or partial_trace_iterations != iterations - original_iterations
        or partial_trace_iterations <= 0
        or report.get("required_relative_normal_residual")
        != H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
        or not _finite_nonnegative_number(initial)
        or not _finite_nonnegative_number(original_final)
        or not _finite_nonnegative_number(recurrence_final)
        or not _finite_nonnegative_number(exact_final)
        or float(exact_final) > H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
        or not isinstance(history, list)
        or len(history) != iterations
        or any(not _finite_nonnegative_number(item) for item in history)
        or float(history[original_iterations - 1]) != float(original_final)
        or float(history[-1]) != float(recurrence_final)
        or report.get("recurrence_history_sha256") != _sha256(history)
        or not isinstance(checkpoints, list)
        or len(checkpoints) < 2
        or report.get("exact_residual_checkpoints_sha256") != _sha256(checkpoints)
        or isinstance(total_matvec_calls, bool)
        or not isinstance(total_matvec_calls, int)
        or report.get("exact_final_residual_authenticated") is not True
        or report.get("extended_after_original_lane_exact_authentication_miss")
        is not True
        or report.get("exact_residual_used_to_initialize_continuous_lane_once")
        is not True
        or report.get("exact_diagnostic_matvecs_modify_recurrence_state_or_direction")
        is not False
        or report.get("execution_profile_sha256")
        != H776_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256
        or not isinstance(lane, dict)
        or set(lane) != lane_fields
        or not isinstance(preconditioner, dict)
        or set(preconditioner) != preconditioner_fields
        or report.get("solution_finite") is not True
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "A continuous hidden partial-trace PCG solve lacks its "
            "exact H776 convergence evidence."
        )
    _authenticate_h776_hidden_partial_trace_pcg_execution_profile(
        report.get("execution_profile")
    )

    checkpoint_by_iteration: dict[int, float] = {}
    previous_iteration = -1
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "iteration",
            "relative_normal_residual",
            "finite",
        }:
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Continuous hidden partial-trace PCG checkpoints are malformed."
            )
        iteration = checkpoint.get("iteration")
        residual = checkpoint.get("relative_normal_residual")
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration <= previous_iteration
            or not original_iterations <= iteration <= iterations
            or not _finite_nonnegative_number(residual)
            or checkpoint.get("finite") is not True
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Continuous hidden partial-trace PCG checkpoints are inconsistent."
            )
        checkpoint_by_iteration[iteration] = float(residual)
        previous_iteration = iteration
    if (
        checkpoints[0]["iteration"] != original_iterations
        or float(checkpoints[0]["relative_normal_residual"])
        <= H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
        or checkpoints[-1]["iteration"] != iterations
        or float(checkpoints[-1]["relative_normal_residual"]) != float(exact_final)
        or any(
            float(checkpoint["relative_normal_residual"])
            <= H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
            for checkpoint in checkpoints[:-1]
        )
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Continuous hidden partial-trace PCG checkpoints do not prove "
            "one scalar-lane miss followed by one successful continuous lane."
        )

    exact_diagnostic_count = len(checkpoints) - 1
    lane_attempted_matvecs = partial_trace_iterations + exact_diagnostic_count
    termination = lane.get("termination")
    if (
        lane.get("start_iteration") != original_iterations
        or lane.get("planned_stop_iteration")
        != H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS
        or lane.get("iterations_executed") != partial_trace_iterations
        or lane.get("attempted_matvec_calls") != lane_attempted_matvecs
        or lane.get("exact_diagnostic_matvec_calls") != exact_diagnostic_count
        or lane.get("ending_iteration") != iterations
        or not _finite_nonnegative_number(
            lane.get("ending_recurrence_relative_normal_residual")
        )
        or float(lane["ending_recurrence_relative_normal_residual"])
        != float(recurrence_final)
        or not _finite_nonnegative_number(
            lane.get("ending_exact_relative_normal_residual")
        )
        or float(lane["ending_exact_relative_normal_residual"]) != float(exact_final)
        or lane.get("ending_exact_residual_authenticated") is not True
        or termination
        not in {
            "exact_tolerance_reached",
            "exact_tolerance_reached_at_hard_cap",
        }
        or (
            termination == "exact_tolerance_reached"
            and (
                iterations >= H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS
                or float(history[-1]) > H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
            )
        )
        or (
            termination == "exact_tolerance_reached_at_hard_cap"
            and iterations != H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS
        )
        or any(
            float(history[checkpoint["iteration"] - 1])
            > H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
            for checkpoint in checkpoints[1:-1]
        )
        or total_matvec_calls != iterations + 1 + len(checkpoints)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The continuous hidden partial-trace PCG lane counters or "
            "exact-residual gates are inconsistent."
        )

    if (
        preconditioner.get("kind") != "deterministic_hidden_partial_trace_cholesky"
        or preconditioner.get("supplied") is not True
        or preconditioner.get("authentication_attempted") is not True
        or preconditioner.get("authenticated") is not True
        or preconditioner.get("applied") is not True
        or preconditioner.get("application_attempt_count") != partial_trace_iterations
        or preconditioner.get("application_count") != partial_trace_iterations
        or preconditioner.get("authentication_failure_reason") is not None
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The continuous hidden partial-trace PCG preconditioner lifecycle "
            "is inconsistent."
        )
    _authenticate_h776_hidden_partial_trace_receipt(
        preconditioner.get("receipt"),
        role=str(role),
    )
    _digest(report.get("solution_sha256"), "PCG solution SHA-256")


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
        if match is None:
            raise Lfm2CpcfV5RealAttentionProbeError(
                "A sequence-effective PCG solve missed its convergence gate."
            )
        solver = report.get("solver")
        if solver == "deterministic_jacobi_pcg_single_tensor":
            _authenticate_legacy_pcg_report(report)
        elif solver == "deterministic_restarted_jacobi_pcg_single_tensor":
            _authenticate_restarted_pcg_report(report)
        elif solver == ("deterministic_continuous_row_block_jacobi_pcg_single_tensor"):
            _authenticate_continuous_row_block_pcg_report(report)
        elif solver == (
            "deterministic_continuous_hidden_partial_trace_cholesky_pcg_single_tensor"
        ):
            _authenticate_continuous_hidden_partial_trace_pcg_report(report)
        else:
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Sequence-effective PCG telemetry names an unknown solver."
            )
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
            artifact.get("candidate_id") != H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID
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
            or frontier[-1].get("stage") != "fixed_producer_value_subspace_anchor"
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
            _digest(value, "sequence solve report SHA-256") for value in solve_hashes
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
            (f"sequence_effective_structured_value_iteration_{iteration}_group_{group}")
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
        restarted_event_count = sum(
            event["report"]["solver"]
            == "deterministic_restarted_jacobi_pcg_single_tensor"
            for event in self._events
        )
        row_block_event_count = sum(
            event["report"]["solver"]
            == "deterministic_continuous_row_block_jacobi_pcg_single_tensor"
            for event in self._events
        )
        hidden_partial_trace_event_count = sum(
            event["report"]["solver"]
            == (
                "deterministic_continuous_hidden_partial_trace_"
                "cholesky_pcg_single_tensor"
            )
            for event in self._events
        )
        maximum_authenticated_iterations = (
            H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS
            if hidden_partial_trace_event_count or row_block_event_count
            else (
                H776_PCG_HARD_MAXIMUM_ITERATIONS
                if restarted_event_count
                else H776_PCG_MAXIMUM_ITERATIONS
            )
        )
        if hidden_partial_trace_event_count:
            if (
                artifact.get("streamed_pcg_execution_profile_sha256")
                != H776_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256
            ):
                raise Lfm2CpcfV5RealAttentionProbeError(
                    "The sequence-effective artifact does not bind the "
                    "immutable H776 PCG continuation profile."
                )
            _authenticate_h776_hidden_partial_trace_pcg_execution_profile(
                artifact.get("streamed_pcg_execution_profile")
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
            "sequence_solve_report_inventory_sha256": _sha256(normalized_solve_hashes),
            "artifact_value_solve_report_sha256": artifact_value_hashes,
            "all_roles_unique": True,
            "all_event_report_hashes_unique": True,
            "all_observation_hashes_unique": True,
            "all_iterations_at_most": maximum_authenticated_iterations,
            "all_final_relative_normal_residuals_at_most": (
                H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
            ),
            "solver_results_or_arguments_modified": False,
        }
        if restarted_event_count:
            body = {
                **body,
                "legacy_solver_event_count": (
                    len(self._events)
                    - restarted_event_count
                    - row_block_event_count
                    - hidden_partial_trace_event_count
                ),
                "restarted_solver_event_count": restarted_event_count,
                "original_lane_maximum_iterations": (H776_PCG_MAXIMUM_ITERATIONS),
                "restart_block_iterations": (H776_PCG_RESTART_BLOCK_ITERATIONS),
                "hard_maximum_iterations": (H776_PCG_HARD_MAXIMUM_ITERATIONS),
                "exact_final_residual_gate": (
                    H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
                ),
                "all_extended_final_residuals_exactly_recomputed": True,
                "all_restart_schedules_authenticated": True,
                "all_matvec_counts_authenticated": True,
                "restarted_extended_solver_events_authenticated": True,
            }
        if row_block_event_count:
            body = {
                **body,
                "legacy_solver_event_count": (
                    len(self._events)
                    - restarted_event_count
                    - row_block_event_count
                    - hidden_partial_trace_event_count
                ),
                "restarted_solver_event_count": restarted_event_count,
                "continuous_row_block_solver_event_count": (row_block_event_count),
                "original_lane_maximum_iterations": (H776_PCG_MAXIMUM_ITERATIONS),
                "row_block_hard_maximum_iterations": (
                    H776_PCG_ROW_BLOCK_HARD_MAXIMUM_ITERATIONS
                ),
                "exact_final_residual_gate": (
                    H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
                ),
                "all_extended_final_residuals_exactly_recomputed": True,
                "all_matvec_counts_authenticated": True,
                "all_continuous_row_block_lanes_authenticated": True,
                "all_row_block_preconditioner_receipts_authenticated": True,
                "continuous_row_block_solver_events_authenticated": True,
            }
        if hidden_partial_trace_event_count:
            body = {
                **body,
                "legacy_solver_event_count": (
                    len(self._events)
                    - restarted_event_count
                    - row_block_event_count
                    - hidden_partial_trace_event_count
                ),
                "restarted_solver_event_count": restarted_event_count,
                "continuous_hidden_partial_trace_solver_event_count": (
                    hidden_partial_trace_event_count
                ),
                "original_lane_maximum_iterations": (H776_PCG_MAXIMUM_ITERATIONS),
                "hidden_partial_trace_hard_maximum_iterations": (
                    H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS
                ),
                "exact_final_residual_gate": (
                    H776_PCG_MAXIMUM_RELATIVE_NORMAL_RESIDUAL
                ),
                "all_extended_final_residuals_exactly_recomputed": True,
                "all_matvec_counts_authenticated": True,
                "all_continuous_hidden_partial_trace_lanes_authenticated": True,
                "all_hidden_partial_trace_preconditioner_receipts_authenticated": (
                    True
                ),
                "continuous_hidden_partial_trace_solver_events_authenticated": (True),
                "hidden_partial_trace_pcg_execution_profile_sha256": (
                    H776_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256
                ),
                "all_hidden_partial_trace_execution_profiles_authenticated": (True),
                "stale_row_block_event_count": row_block_event_count,
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
        maximum_process_rss_bytes: int = H776_MAXIMUM_PROCESS_RSS_BYTES,
        maximum_cuda_bytes_per_device: int = H776_MAXIMUM_CUDA_BYTES_PER_DEVICE,
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
        for name, value in (
            ("maximum_process_rss_bytes", maximum_process_rss_bytes),
            ("maximum_cuda_bytes_per_device", maximum_cuda_bytes_per_device),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise Lfm2CpcfV5RealAttentionProbeError(
                    f"{name} must be a positive integer."
                )
        self._maximum_process_rss_bytes = maximum_process_rss_bytes
        self._maximum_cuda_bytes_per_device = maximum_cuda_bytes_per_device
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._failure: BaseException | None = None

    @property
    def maximum_process_rss_bytes(self) -> int:
        return self._maximum_process_rss_bytes

    @property
    def maximum_cuda_bytes_per_device(self) -> int:
        return self._maximum_cuda_bytes_per_device

    def _sample(self, label: str) -> None:
        process = {str(k): int(v) for k, v in self._process_sampler().items()}
        cuda = {str(k): int(v) for k, v in self._cuda_sampler().items()}
        if (
            process["VmRSS"] > self._maximum_process_rss_bytes
            or process["VmHWM"] > self._maximum_process_rss_bytes
            or process["VmSwap"] != 0
            or any(
                value > self._maximum_cuda_bytes_per_device for value in cuda.values()
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
            "maximum_process_rss_bytes": self._maximum_process_rss_bytes,
            "maximum_cuda_bytes_per_device": (self._maximum_cuda_bytes_per_device),
            "peak_vm_rss_bytes": max(sample["process"]["VmRSS"] for sample in samples),
            "peak_vm_hwm_bytes": max(sample["process"]["VmHWM"] for sample in samples),
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
        or int(getattr(config, "num_attention_heads", -1)) != H776_TARGET_QUERY_HEADS
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


def _authenticate_attention_sequential_transport(
    *,
    report: Mapping[str, Any],
    policy: Mapping[str, Any],
    selection: Mapping[str, Any],
    evaluations: Mapping[str, Any],
    selected_id: Any,
) -> Mapping[str, Any]:
    transport_policy = _self_sealed(
        policy.get("attention_sequential_transport_policy"),
        seal_field="report_sha256",
        name="attention sequential-transport policy",
    )
    final_binding = _self_sealed(
        policy.get("attention_candidate_local_post_layer_transport"),
        seal_field="report_sha256",
        name="attention sequential-transport final binding",
    )
    authenticated_selection = _self_sealed(
        selection,
        seal_field="report_sha256",
        name="attention candidate selection",
    )
    selected_chart_sha = _digest(
        authenticated_selection.get("selected_effective_boundary_charts_sha256"),
        "selected effective boundary charts SHA-256",
    )
    selected_bundle_sha = _digest(
        authenticated_selection.get("selected_operator_transport_mlp_bundle_sha256"),
        "selected operator/transport/MLP bundle SHA-256",
    )
    candidate_chart_hashes = authenticated_selection.get(
        "candidate_effective_boundary_charts_sha256"
    )
    candidate_bundle_hashes = authenticated_selection.get(
        "candidate_operator_transport_mlp_bundle_sha256"
    )
    candidate_report_hashes = authenticated_selection.get("candidate_report_sha256")
    source_chart_sha = _digest(
        report.get("boundary_charts_sha256"),
        "source boundary charts SHA-256",
    )
    source_chart_payload_sha: str | None = None
    if (
        transport_policy.get("enabled") is not True
        or transport_policy.get("fit_split_only") is not True
        or transport_policy.get("selection_refits_or_mutates_transport") is not False
        or transport_policy.get("selection_selects_frozen_bundle") is not True
        or transport_policy.get("validation_argument_supported") is not False
        or authenticated_selection.get("candidate_local_post_layer_transport_enabled")
        is not True
        or authenticated_selection.get("operator_transport_mlp_bundle_frozen_together")
        is not True
        or authenticated_selection.get(
            "selection_metrics_use_candidate_effective_charts"
        )
        is not True
        or authenticated_selection.get("selection_refits_or_mutates_transport")
        is not False
        or authenticated_selection.get("selection_selects_frozen_bundle") is not True
        or authenticated_selection.get("candidate_local_mlp_preparation_count")
        != len(H776_ATTENTION_CANDIDATE_IDS)
        or authenticated_selection.get("cross_candidate_prepared_mlp_basis_reused")
        is not False
        or set(candidate_chart_hashes or {}) != set(H776_ATTENTION_CANDIDATE_IDS)
        or set(candidate_bundle_hashes or {}) != set(H776_ATTENTION_CANDIDATE_IDS)
        or set(candidate_report_hashes or {}) != set(H776_ATTENTION_CANDIDATE_IDS)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Attention sequential-transport selection policy is incomplete."
        )

    authenticated_evaluations: dict[str, dict[str, Any]] = {}
    for candidate_id in H776_ATTENTION_CANDIDATE_IDS:
        evaluation = _self_sealed(
            evaluations.get(candidate_id),
            seal_field="report_sha256",
            name=f"attention candidate {candidate_id!r}",
        )
        transport = _self_sealed(
            evaluation.get("sequential_boundary_transport"),
            seal_field="report_sha256",
            name=f"attention candidate {candidate_id!r} transport",
        )
        binding = _self_sealed(
            evaluation.get("sequential_transport_binding"),
            seal_field="report_sha256",
            name=f"attention candidate {candidate_id!r} transport binding",
        )
        chart_sha = _digest(
            evaluation.get("effective_boundary_charts_sha256"),
            f"attention candidate {candidate_id!r} chart SHA-256",
        )
        bundle_sha = _digest(
            evaluation.get("operator_transport_mlp_bundle_sha256"),
            f"attention candidate {candidate_id!r} bundle SHA-256",
        )
        binding_source_payload_sha = _digest(
            binding.get("source_boundary_charts_payload_sha256"),
            (f"attention candidate {candidate_id!r} source chart payload SHA-256"),
        )
        effective_payload_sha = _digest(
            evaluation.get("effective_boundary_charts_payload_sha256"),
            (f"attention candidate {candidate_id!r} effective chart payload SHA-256"),
        )
        if source_chart_payload_sha is None:
            source_chart_payload_sha = binding_source_payload_sha
        if (
            candidate_report_hashes[candidate_id] != evaluation["report_sha256"]
            or candidate_chart_hashes[candidate_id] != chart_sha
            or candidate_bundle_hashes[candidate_id] != bundle_sha
            or binding.get("candidate_id") != candidate_id
            or binding.get("effective_boundary_charts_sha256") != chart_sha
            or binding.get("effective_boundary_charts_payload_sha256")
            != effective_payload_sha
            or binding.get("source_boundary_charts_sha256") != source_chart_sha
            or binding_source_payload_sha != source_chart_payload_sha
            or binding.get("sequential_transport_report_sha256")
            != transport["report_sha256"]
            or binding.get("policy_report_sha256") != transport_policy["report_sha256"]
            or binding.get("fit_split_only") is not True
            or binding.get("selection_rows_read_by_transport") != 0
            or binding.get("selection_rows_read_before_transport_freeze") != 0
            or binding.get("validation_rows_read") != 0
            or binding.get("equal_total_mass_per_fit_phase") is not True
            or binding.get("one_closed_form_svd") is not True
            or binding.get("source_post_operator_chart_same_object") is not True
            or binding.get("source_post_layer_chart_unchanged") is not True
            or binding.get("transported_post_layer_chart_candidate_local") is not True
            or binding.get("automatic_retry_or_retuning_performed") is not False
            or evaluation.get("candidate_local_post_layer_transport_frozen_before_mlp")
            is not True
            or evaluation.get("candidate_local_mlp_basis_preparation_count") != 1
            or evaluation.get("cross_candidate_prepared_mlp_basis_reused") is not False
            or evaluation.get("selection_metrics_use_effective_boundary_charts")
            is not True
            or evaluation.get("operator_transport_and_mlp_ranked_as_one_bundle")
            is not True
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                f"Attention candidate {candidate_id!r} is not bound to one "
                "fit-only frozen transport bundle."
            )
        authenticated_evaluations[candidate_id] = evaluation

    if not isinstance(selected_id, str) or selected_id not in authenticated_evaluations:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Attention sequential-transport winner is invalid."
        )
    selected = authenticated_evaluations[selected_id]
    if (
        selected["effective_boundary_charts_sha256"] != selected_chart_sha
        or selected["operator_transport_mlp_bundle_sha256"] != selected_bundle_sha
        or authenticated_selection.get("selected_candidate_report_sha256")
        != selected["report_sha256"]
        or final_binding.get("policy") != transport_policy
        or final_binding.get("selected_candidate_id") != selected_id
        or final_binding.get("selected_candidate_report_sha256")
        != selected["report_sha256"]
        or final_binding.get("candidate_selection_report_sha256")
        != authenticated_selection["report_sha256"]
        or final_binding.get("operator_transport_mlp_bundle_sha256")
        != selected_bundle_sha
        or final_binding.get("effective_boundary_charts_sha256") != selected_chart_sha
        or final_binding.get("effective_boundary_charts_payload_sha256")
        != selected["effective_boundary_charts_payload_sha256"]
        or final_binding.get("source_boundary_charts_sha256") != source_chart_sha
        or final_binding.get("source_boundary_charts_payload_sha256")
        != source_chart_payload_sha
        or final_binding.get("sequential_transport_report_sha256")
        != selected["sequential_boundary_transport"]["report_sha256"]
        or final_binding.get("sequential_transport_binding_report_sha256")
        != selected["sequential_transport_binding"]["report_sha256"]
        or final_binding.get("final_endpoint_metrics_use_winning_effective_chart")
        is not True
        or final_binding.get(
            "validation_first_endpoint_replay_and_metric_after_bundle_freeze"
        )
        is not True
        or final_binding.get("validation_rows_used_by_transport_solver") != 0
        or final_binding.get("selection_refits_or_mutates_transport") is not False
        or final_binding.get("selection_selects_frozen_bundle") is not True
        or final_binding.get("operator_transport_mlp_bundle_installed_exactly")
        is not True
        or policy.get("attention_each_candidate_prepares_own_mlp_basis") is not True
        or policy.get("attention_cross_candidate_prepared_mlp_basis_reuse") is not False
        or policy.get("attention_final_endpoint_effective_boundary_charts_sha256")
        != selected_chart_sha
        or not isinstance(report.get("audit"), dict)
        or report["audit"].get(
            "source_boundary_charts_unchanged_after_candidate_transports"
        )
        is not True
        or report["audit"].get("winning_effective_boundary_chart_used_for_validation")
        is not True
        or not isinstance(report.get("actual_child_state_policy"), dict)
        or report["actual_child_state_policy"].get(
            "attention_operator_transport_mlp_frozen_together"
        )
        is not True
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Final attention state is not cross-bound to the selected frozen "
            "operator/transport/MLP bundle."
        )
    body = {
        "enabled": True,
        "policy_report_sha256": transport_policy["report_sha256"],
        "candidate_selection_report_sha256": (authenticated_selection["report_sha256"]),
        "selected_candidate_id": selected_id,
        "selected_candidate_report_sha256": selected["report_sha256"],
        "selected_effective_boundary_charts_sha256": selected_chart_sha,
        "selected_operator_transport_mlp_bundle_sha256": selected_bundle_sha,
        "source_boundary_charts_sha256": source_chart_sha,
        "source_boundary_charts_payload_sha256": source_chart_payload_sha,
        "candidate_effective_boundary_charts_sha256": dict(candidate_chart_hashes),
        "candidate_operator_transport_mlp_bundle_sha256": dict(candidate_bundle_hashes),
        "fit_only_candidate_local_transport_authenticated": True,
        "selection_selects_frozen_bundle_without_refit": True,
        "validation_rows_used_by_transport_solver": 0,
        "winning_chart_used_for_final_metrics": True,
    }
    return MappingProxyType({**body, "audit_sha256": _sha256(body)})


def authenticate_h776_attention_result(
    result: Any,
    *,
    input_bank_sha256: str,
    pcg_audit: Mapping[str, Any],
    pins: Lfm2CpcfV5RealAttentionProbePins,
    expected_attention_sequential_transport_enabled: bool = False,
) -> Mapping[str, Any]:
    from .hf_lfm2_sequence_effective_attention_fold import (
        LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_SCHEMA,
    )

    if type(expected_attention_sequential_transport_enabled) is not bool:
        raise TypeError("expected_attention_sequential_transport_enabled must be bool.")
    if (
        expected_attention_sequential_transport_enabled
        != pins.expected_attention_sequential_transport_enabled
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Attention sequential-transport authentication mode differs "
            "from its independent invocation pin."
        )
    authenticated_pcg = _self_sealed(
        pcg_audit,
        seal_field="audit_sha256",
        name="PCG audit",
    )
    if (
        authenticated_pcg.get("schema") != LFM2_CPCF_V5_REAL_ATTENTION_PCG_AUDIT_SCHEMA
        or authenticated_pcg.get("multi_rhs_solve_count") != 0
        or authenticated_pcg.get("single_tensor_solve_count") not in {5, 10}
        or authenticated_pcg.get("event_count")
        != authenticated_pcg.get("single_tensor_solve_count")
        or authenticated_pcg.get("all_roles_unique") is not True
        or authenticated_pcg.get("all_event_report_hashes_unique") is not True
        or authenticated_pcg.get("all_observation_hashes_unique") is not True
        or authenticated_pcg.get("all_iterations_at_most")
        not in {
            H776_PCG_MAXIMUM_ITERATIONS,
            H776_PCG_HARD_MAXIMUM_ITERATIONS,
            H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS,
        }
        or (
            authenticated_pcg.get("all_iterations_at_most")
            == H776_PCG_HARD_MAXIMUM_ITERATIONS
            and authenticated_pcg.get("restarted_extended_solver_events_authenticated")
            is not True
        )
        or (
            authenticated_pcg.get("all_iterations_at_most")
            == H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS
            and authenticated_pcg.get(
                "continuous_hidden_partial_trace_solver_events_authenticated"
            )
            is not True
        )
        or (
            authenticated_pcg.get("all_iterations_at_most")
            == H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS
            and authenticated_pcg.get(
                "all_continuous_hidden_partial_trace_lanes_authenticated"
            )
            is not True
        )
        or (
            authenticated_pcg.get("all_iterations_at_most")
            == H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS
            and authenticated_pcg.get(
                "all_hidden_partial_trace_preconditioner_receipts_authenticated"
            )
            is not True
        )
        or (
            authenticated_pcg.get("all_iterations_at_most")
            == H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS
            and authenticated_pcg.get(
                "all_hidden_partial_trace_execution_profiles_authenticated"
            )
            is not True
        )
        or (
            authenticated_pcg.get("all_iterations_at_most")
            == H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS
            and authenticated_pcg.get(
                "hidden_partial_trace_pcg_execution_profile_sha256"
            )
            != H776_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256
        )
        or (
            authenticated_pcg.get("all_iterations_at_most")
            == H776_PCG_HIDDEN_PARTIAL_TRACE_HARD_MAXIMUM_ITERATIONS
            and authenticated_pcg.get("stale_row_block_event_count") != 0
        )
        or authenticated_pcg.get("solver_results_or_arguments_modified") is not False
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
    if expected_attention_sequential_transport_enabled:
        sequential_transport_audit = _authenticate_attention_sequential_transport(
            report=report,
            policy=policy,
            selection=selection,
            evaluations=evaluations,
            selected_id=selected_id,
        )
    else:
        forbidden_transport_fields = {
            "attention_candidate_local_post_layer_transport",
            "attention_sequential_transport_policy",
            "attention_each_candidate_prepares_own_mlp_basis",
            "attention_cross_candidate_prepared_mlp_basis_reuse",
            "attention_final_endpoint_effective_boundary_charts_sha256",
        }
        if forbidden_transport_fields.intersection(policy) or any(
            {
                "sequential_boundary_transport",
                "sequential_transport_binding",
                "effective_boundary_charts_sha256",
                "operator_transport_mlp_bundle_sha256",
            }.intersection(evaluation)
            for evaluation in evaluations.values()
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Attention sequential transport executed while its pinned "
                "diagnostic mode was off."
            )
        sequential_transport_audit = None
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
    solver_plan = (
        None if not isinstance(artifact, dict) else artifact.get("solver_plan")
    )
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema") != LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_SCHEMA
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
        or solver_plan.get("report_sha256") != pins.expected_sequence_solver_plan_sha256
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
        or artifact.get("candidate_frontier_sha256") != _sha256(artifact_frontier)
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
            or binding.get("observation_sha256") != event.get("observation_sha256")
            or binding.get("pcg_report_sha256") != event.get("report_sha256")
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
        behavioral_keys[candidate_id] = tuple(float(value) for value in raw_key[:-1])
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
        "sequence_effective_artifact": _plain(artifact),
        "sequence_solver_plan": _plain(solver_plan),
        "pcg_convergence": authenticated_pcg,
        "in_memory_next_bank_created_by_layer_replay": True,
        "k3_checkpoint_published": False,
        "layer_3_compiled": False,
        "automatic_retry_or_retuning_performed": False,
    }
    if sequential_transport_audit is not None:
        body = {
            **body,
            "attention_sequential_transport": _plain(sequential_transport_audit),
        }
    return MappingProxyType({**body, "audit_sha256": _sha256(body)})


def _h776_composed_qk_attention_policy() -> Any:
    """Build the fixed two-3090 analytical policy for the real H776 lane."""

    from .hf_lfm2_cached_qk_score_residual_fold import (
        Lfm2QkScoreResidualFoldPolicy,
    )
    from .hf_lfm2_composed_qk_attention_fold import (
        Lfm2ComposedQkAttentionFoldPolicy,
    )
    from .hf_lfm2_qk_global_gauge_sync import (
        Lfm2QkGlobalGaugeSyncPolicy,
    )

    return Lfm2ComposedQkAttentionFoldPolicy(
        global_policy=Lfm2QkGlobalGaugeSyncPolicy(
            maximum_total_score_elements=134_217_728,
            maximum_total_factor_elements=67_108_864,
            maximum_total_output_elements=201_326_592,
            maximum_total_svd_workspace_elements=67_108_864,
        ),
        score_policy=Lfm2QkScoreResidualFoldPolicy(
            maximum_fit_centered_score_relative_rmse=0.10,
            maximum_fit_probability_relative_rmse=0.20,
            maximum_fit_js_divergence_mean=0.05,
            maximum_fit_js_divergence_worst=0.10,
            minimum_fit_top1_route_agreement=0.90,
            maximum_fit_check_centered_score_relative_rmse=0.20,
            maximum_fit_check_probability_relative_rmse=0.20,
            maximum_fit_check_js_divergence_mean=0.05,
            maximum_fit_check_js_divergence_worst=0.10,
            minimum_fit_check_top1_route_agreement=0.90,
            solver_device="cuda:1",
            compute_devices=("cuda:0", "cuda:1"),
        ),
    )


def _h776_endpoint_residual_attention_policy() -> Any:
    """Build the fixed fifth-arm endpoint policy for the real H776 lane."""

    from .hf_lfm2_cached_endpoint_qk_residual_fold import (
        Lfm2EndpointQkResidualFoldPolicy,
    )
    from .hf_lfm2_composed_qk_attention_fold import (
        Lfm2ComposedEndpointResidualAttentionFoldPolicy,
    )

    policy = Lfm2ComposedEndpointResidualAttentionFoldPolicy(
        maximum_outer_blocks=2,
        endpoint_policy=Lfm2EndpointQkResidualFoldPolicy(
            maximum_cg_iterations=64,
            cg_restart_iterations=64,
            cg_relative_tolerance=1e-2,
            solver_device="cuda:1",
            compute_devices=("cuda:0", "cuda:1"),
        ),
    )
    if (
        policy.authentication_sha256
        != H776_ENDPOINT_RESIDUAL_ATTENTION_POLICY_SHA256
        or policy.verify_authentication() is not True
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The fixed H776 endpoint-residual policy fingerprint changed."
        )
    return policy


def _authenticate_qk_routing_metrics(
    metrics: Any,
    *,
    policy: Mapping[str, Any],
    partition: str,
    expected_case_count: int,
    require_gate: bool = True,
) -> dict[str, Any]:
    if type(require_gate) is not bool:
        raise TypeError("require_gate must be bool.")
    if not isinstance(metrics, Mapping):
        raise Lfm2CpcfV5RealAttentionProbeError(
            f"Composed {partition} routing metrics are absent."
        )
    result = _plain(metrics)
    prefix = "fit" if partition == "fit" else "fit_check"
    ceilings = {
        "centered_score_relative_rmse": (
            f"maximum_{prefix}_centered_score_relative_rmse"
        ),
        "probability_relative_rmse": (f"maximum_{prefix}_probability_relative_rmse"),
        "js_divergence_mean": f"maximum_{prefix}_js_divergence_mean",
        "js_divergence_worst": f"maximum_{prefix}_js_divergence_worst",
    }
    for metric_name, policy_name in ceilings.items():
        value = result.get(metric_name)
        maximum = policy.get(policy_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or isinstance(maximum, bool)
            or not isinstance(maximum, (int, float))
            or (require_gate and float(value) > float(maximum))
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                f"Composed {partition} {metric_name} failed its fixed gate."
            )
    top1 = result.get("top1_route_agreement_fraction")
    minimum_top1 = policy.get(f"minimum_{prefix}_top1_route_agreement")
    if (
        isinstance(top1, bool)
        or not isinstance(top1, (int, float))
        or not math.isfinite(float(top1))
        or isinstance(minimum_top1, bool)
        or not isinstance(minimum_top1, (int, float))
        or (require_gate and float(top1) < float(minimum_top1))
        or result.get("maximum_causal_probability_leakage") != 0.0
        or result.get("case_count") != expected_case_count
        or not isinstance(result.get("eligible_query_rows"), int)
        or result["eligible_query_rows"] <= 0
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            f"Composed {partition} routing agreement or causal mask failed."
        )
    return result


def _authenticate_h776_composed_score_outcome(
    score: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    composed_candidate_eligible: bool,
) -> Mapping[str, Any]:
    """Authenticate accepted or truthful rejected score-stage split use."""

    if type(composed_candidate_eligible) is not bool:
        raise TypeError("composed_candidate_eligible must be bool.")
    operation = score.get("operation_audit")
    fit_check_evaluated = score.get("fit_check_evaluated")
    status = score.get("status")
    accepted_statuses = {
        "fit_check_accepted",
        "quality_gated_seed_no_op",
    }
    rejected_without_fit_check_statuses = {
        "pcg_nonconverged_seed_fallback",
        "pcg_unusable_seed_fallback",
        "poor_quality_seed_fallback",
        "unidentifiable_zero_bilinear_jacobian",
    }
    rejected_with_fit_check_statuses = {
        "fit_check_rejected_seed_fallback",
    }
    expected_fit_check_evaluated = (
        composed_candidate_eligible
        or status in rejected_with_fit_check_statuses
    )
    if (
        score.get("accepted") is not composed_candidate_eligible
        or type(score.get("changed")) is not bool
        or (
            composed_candidate_eligible
            and status not in accepted_statuses
        )
        or (
            not composed_candidate_eligible
            and status
            not in (
                rejected_without_fit_check_statuses
                | rejected_with_fit_check_statuses
            )
        )
        or (
            composed_candidate_eligible
            and score.get("changed")
            is not (status == "fit_check_accepted")
        )
        or (
            not composed_candidate_eligible
            and score.get("changed") is not False
        )
        or not isinstance(operation, Mapping)
        or operation.get("fit_cases_consumed") != H776_COMPOSED_QK_FIT_CASES
        or operation.get("selection_cases_consumed") != 0
        or operation.get("true_validation_cases_consumed") != 0
        or any(
            operation.get(name) != 0
            for name in (
                "backward_calls",
                "optimizer_steps",
                "training_steps",
                "model_forward_calls",
                "model_generate_calls",
                "parameters_pruned",
                "parameters_quantized",
                "target_mutations_during_compile",
            )
        )
        or type(fit_check_evaluated) is not bool
        or fit_check_evaluated is not expected_fit_check_evaluated
        or (
            fit_check_evaluated
            and (
                score.get("fit_check_triggered_refit") is not False
                or score.get("source_evidence_unchanged") is not True
            )
        )
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Composed score outcome crossed an evidence split or training gate."
        )
    fit_metrics = _authenticate_qk_routing_metrics(
        score.get("fit_metrics"),
        policy=policy,
        partition="fit",
        expected_case_count=H776_COMPOSED_QK_FIT_CASES,
        require_gate=composed_candidate_eligible,
    )
    fit_check_metrics = None
    if fit_check_evaluated:
        if (
            operation.get("fit_check_cases_consumed")
            != H776_COMPOSED_QK_FIT_CHECK_CASES
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Evaluated composed FIT-CHECK has the wrong case count."
            )
        fit_check_metrics = _authenticate_qk_routing_metrics(
            score.get("fit_check_metrics"),
            policy=policy,
            partition="fit_check",
            expected_case_count=H776_COMPOSED_QK_FIT_CHECK_CASES,
            require_gate=composed_candidate_eligible,
        )
    elif (
        score.get("fit_check_metrics") is not None
        or operation.get("fit_check_cases_consumed") != 0
        or score.get("seed_fit_check_metrics") is not None
        or score.get("proposed_fit_check_metrics") is not None
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Unevaluated rejected composed score outcome consumed FIT-CHECK."
        )
    body = {
        "status": status,
        "accepted": composed_candidate_eligible,
        "changed": score["changed"],
        "fit_check_evaluated": fit_check_evaluated,
        "fit_cases_consumed": operation["fit_cases_consumed"],
        "fit_check_cases_consumed": operation["fit_check_cases_consumed"],
        "fit_metrics": fit_metrics,
        "fit_check_metrics": fit_check_metrics,
        "fold_only": True,
    }
    return MappingProxyType({**body, "audit_sha256": _sha256(body)})


def _authenticate_h776_composed_device_execution(
    score: Mapping[str, Any],
    *,
    producer: Mapping[str, Any],
    global_sync: Mapping[str, Any],
    fit_trajectory_sha256s: Sequence[str],
    pins: Lfm2CpcfV5RealAttentionProbePins,
) -> Mapping[str, Any]:
    from .hf_lfm2_cached_qk_score_residual_fold import (
        LFM2_QK_LINEARIZATION_CACHE_RESOURCE_SCHEMA,
        LFM2_QK_OBSERVED_EXECUTION_RECEIPT_SCHEMA,
    )

    execution = score.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("solver_device") != "cuda:1"
        or execution.get("compute_devices") != ["cuda:0", "cuda:1"]
        or execution.get("case_assignment") != "authenticated_bank_order_round_robin"
        or execution.get("compute_device_concurrency") != "serial_round_robin"
        or execution.get("streamed_evidence") is not False
        or execution.get("fixed_linearization_cache") is not True
        or execution.get("linearization_cache_scope") != "one_gauss_newton_sweep"
        or execution.get("maximum_logically_resident_prepared_cases")
        != H776_COMPOSED_QK_FIT_CASES
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The score fold lacks its exact two-device execution contract."
        )
    sweeps = score.get("sweeps")
    if (
        not isinstance(sweeps, list)
        or len(sweeps) > 2
        or (score.get("changed") is True and not sweeps)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The score fold lacks its per-sweep fixed-linearization inventory."
        )
    cache_resource_receipts: list[Mapping[str, Any]] = []
    for sweep_index, sweep in enumerate(sweeps):
        if not isinstance(sweep, Mapping) or sweep.get("sweep") != sweep_index:
            raise Lfm2CpcfV5RealAttentionProbeError(
                "The score fold has a malformed linearization sweep."
            )
        resource = _self_sealed(
            sweep.get("linearization_cache_resource_receipt"),
            seal_field="resource_receipt_sha256",
            name=f"score fixed-linearization sweep {sweep_index}",
        )
        per_device_counts = resource.get("per_device_case_counts")
        per_device_bytes = resource.get("per_device_logical_tensor_bytes")
        if (
            resource.get("schema") != LFM2_QK_LINEARIZATION_CACHE_RESOURCE_SCHEMA
            or resource.get("cache_scope") != "one_gauss_newton_sweep"
            or resource.get("resident_storage") != "compute_device"
            or resource.get("source_fit_case_count") != H776_COMPOSED_QK_FIT_CASES
            or resource.get("resident_prepared_case_count")
            != H776_COMPOSED_QK_FIT_CASES
            or resource.get("linearization_replay_call_count")
            != H776_COMPOSED_QK_FIT_CASES
            or per_device_counts != {"cuda:0": 67, "cuda:1": 67}
            or not isinstance(per_device_bytes, dict)
            or set(per_device_bytes) != {"cuda:0", "cuda:1"}
            or any(
                type(value) is not int or value <= 0
                for value in per_device_bytes.values()
            )
            or resource.get("total_logical_tensor_bytes")
            != sum(per_device_bytes.values())
            or resource.get("maximum_device_logical_tensor_bytes")
            != max(per_device_bytes.values())
            or resource["maximum_device_logical_tensor_bytes"]
            > pins.maximum_cuda_bytes_per_device
            or resource.get("cache_reused_for_right_hand_side") is not True
            or resource.get("cache_reused_for_normal_operator_calls") is not True
            or resource.get("cache_invalidated_at_sweep_end") is not True
            or any(
                type(resource.get(name)) is not int or resource.get(name) != 0
                for name in (
                    "model_forward_calls",
                    "model_generate_calls",
                    "backward_calls",
                    "optimizer_steps",
                    "training_steps",
                )
            )
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "The score fold has an invalid fixed-linearization resource receipt."
            )
        cache_resource_receipts.append(resource)
    receipt = _self_sealed(
        execution.get("observed_execution_receipt"),
        seal_field="receipt_sha256",
        name="observed score execution",
    )
    observations = receipt.get("compute_observations")
    solver = receipt.get("solver_observation")
    distinct_case_receipts = receipt.get("distinct_case_receipts")
    expected_devices = ["cuda:0", "cuda:1"]
    expected_uuids = list(pins.expected_composed_logical_cuda_uuids)
    qk_fit_ordinals = producer.get("qk_fit_source_ordinals")
    trajectories = tuple(fit_trajectory_sha256s)
    if (
        not isinstance(qk_fit_ordinals, list)
        or len(qk_fit_ordinals) != H776_COMPOSED_QK_FIT_CASES
        or len(trajectories) != H776_COMPOSED_UPSTREAM_FIT_CASES
        or any(type(value) is not int for value in qk_fit_ordinals)
        or any(value < 0 or value >= len(trajectories) for value in qk_fit_ordinals)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Observed score execution lacks the exact Q/K-FIT assignment map."
        )
    expected_assignments = tuple(
        (ordinal, trajectories[ordinal]) for ordinal in qk_fit_ordinals[:2]
    )
    if (
        receipt.get("schema") != LFM2_QK_OBSERVED_EXECUTION_RECEIPT_SCHEMA
        or receipt.get("requested_solver_device") != "cuda:1"
        or receipt.get("requested_compute_devices") != expected_devices
        or receipt.get("requested_compute_device_count") != 2
        or receipt.get("observed_compute_device_count") != 2
        or receipt.get("every_requested_compute_device_observed") is not True
        or receipt.get("solver_device_observed") is not True
        or receipt.get("cached_evidence_analytical_tensor_operation_count") != 2
        or receipt.get("solver_tensor_operation_count") != 1
        or receipt.get("distinct_case_receipt_count") != 2
        or receipt.get("distinct_case_execution_receipt_count") != 2
        or receipt.get("per_device_case_counts") != {"cuda:0": 1, "cuda:1": 1}
        or not isinstance(observations, list)
        or len(observations) != 2
        or not isinstance(solver, dict)
        or not isinstance(distinct_case_receipts, list)
        or len(distinct_case_receipts) != 2
        or not isinstance(receipt.get("operation_audit"), dict)
        or any(
            type(value) is not int or value != 0
            for value in receipt["operation_audit"].values()
        )
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Observed score execution is incomplete or not fold-only."
        )
    for index, value in enumerate(distinct_case_receipts):
        _digest(value, f"observed distinct case receipt {index}")
    if len(set(distinct_case_receipts)) != 2:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Observed score execution did not use two distinct FIT cases."
        )
    authenticated_observations = []
    for index, observation in enumerate(observations):
        authenticated = _self_sealed(
            observation,
            seal_field="case_execution_receipt_sha256",
            name=f"observed score compute device {index}",
        )
        scalar = authenticated.get("analytical_scalar")
        if (
            authenticated.get("assignment_index") != index
            or authenticated.get("intended_logical_device") != expected_devices[index]
            or authenticated.get("observed_logical_device") != expected_devices[index]
            or authenticated.get("observed_device_type") != "cuda"
            or authenticated.get("observed_device_index") != index
            or authenticated.get("cuda_uuid") != expected_uuids[index]
            or authenticated.get("cuda_name") != H776_COMPOSED_CUDA_DEVICE_NAME
            or (
                authenticated.get("cached_case_ordinal"),
                authenticated.get("cached_trajectory_sha256"),
            )
            != expected_assignments[index]
            or authenticated.get("analytical_operation")
            != "masked_centered_score_energy_plus_valid_cell_count"
            or type(authenticated.get("analytical_tensor_element_count")) is not int
            or authenticated["analytical_tensor_element_count"] <= 0
            or authenticated.get("analytical_scalar_dtype") != "torch.float64"
            or isinstance(scalar, bool)
            or not isinstance(scalar, (int, float))
            or not math.isfinite(float(scalar))
            or float(scalar) <= 0.0
            or authenticated.get("forced_host_scalar_read") is not True
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                f"Logical CUDA device {index} lacks observed cached-evidence work."
            )
        for name in (
            "cached_evidence_sha256",
            "cached_content_sha256",
            "cached_transform_receipt_sha256",
            "cached_trajectory_sha256",
        ):
            _digest(authenticated.get(name), f"device {index} {name}")
        if type(authenticated.get("cached_case_ordinal")) is not int:
            raise Lfm2CpcfV5RealAttentionProbeError(
                f"Logical CUDA device {index} lacks a case ordinal."
            )
        authenticated_observations.append(authenticated)
    if [
        value["cached_evidence_sha256"] for value in authenticated_observations
    ] != distinct_case_receipts:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Observed device work differs from the sealed case inventory."
        )
    authenticated_solver = _self_sealed(
        solver,
        seal_field="solver_execution_receipt_sha256",
        name="observed score solver device",
    )
    solver_scalar = authenticated_solver.get("analytical_scalar")
    if (
        authenticated_solver.get("intended_logical_device") != "cuda:1"
        or authenticated_solver.get("observed_logical_device") != "cuda:1"
        or authenticated_solver.get("observed_device_type") != "cuda"
        or authenticated_solver.get("observed_device_index") != 1
        or authenticated_solver.get("cuda_uuid") != expected_uuids[1]
        or authenticated_solver.get("cuda_name") != H776_COMPOSED_CUDA_DEVICE_NAME
        or authenticated_solver.get("source_seed_sha256") != score.get("seed_sha256")
        or authenticated_solver.get("source_query_weight_sha256")
        != global_sync.get("query_weight_sha256")
        or authenticated_solver.get("source_key_weight_sha256")
        != global_sync.get("key_weight_sha256")
        or authenticated_solver.get("analytical_operation")
        != "dense_seed_qk_energy_plus_parameter_count"
        or type(authenticated_solver.get("analytical_tensor_element_count")) is not int
        or authenticated_solver["analytical_tensor_element_count"] <= 0
        or authenticated_solver.get("analytical_scalar_dtype") != "torch.float64"
        or isinstance(solver_scalar, bool)
        or not isinstance(solver_scalar, (int, float))
        or not math.isfinite(float(solver_scalar))
        or float(solver_scalar) <= 0.0
        or authenticated_solver.get("forced_host_scalar_read") is not True
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The logical CUDA solver device lacks observed dense Q/K work."
        )
    for name in (
        "source_seed_sha256",
        "source_query_weight_sha256",
        "source_key_weight_sha256",
    ):
        _digest(authenticated_solver.get(name), f"solver {name}")
    body = {
        "observed_execution_receipt_sha256": receipt["receipt_sha256"],
        "logical_compute_devices": expected_devices,
        "physical_cuda_uuids": expected_uuids,
        "cuda_device_name": H776_COMPOSED_CUDA_DEVICE_NAME,
        "distinct_cached_fit_cases_executed": 2,
        "solver_device": "cuda:1",
        "every_requested_compute_device_observed": True,
        "solver_device_observed": True,
        "fixed_linearization_cache_authenticated": bool(cache_resource_receipts),
        "linearization_cache_resource_receipt_sha256s": [
            value["resource_receipt_sha256"] for value in cache_resource_receipts
        ],
        "fold_only_execution_authenticated": True,
    }
    return MappingProxyType({**body, "audit_sha256": _sha256(body)})


def _authenticate_h776_endpoint_device_execution(
    proposal: Mapping[str, Any],
    *,
    expected_fit_trajectory_sha256s: Sequence[str],
    pins: Lfm2CpcfV5RealAttentionProbePins,
) -> Mapping[str, Any]:
    """Authenticate real endpoint JVP/VJP work across both pinned 3090s."""

    from .hf_lfm2_cached_endpoint_qk_residual_fold import (
        LFM2_ENDPOINT_QK_EXECUTION_SCHEMA,
    )

    expected_devices = ["cuda:0", "cuda:1"]
    expected_uuids = list(pins.expected_composed_logical_cuda_uuids)
    expected_transfer_execution = (
        "once_per_static_device_shard_serial_requested_device_order"
    )
    expected_fixed_cache_bytes = 4_470_528
    expected_static_shard_workspace_bytes = 11_919_360
    expected_solver_workspace_bytes = 47_677_440
    trajectories = tuple(expected_fit_trajectory_sha256s)
    if (
        len(trajectories) != H776_COMPOSED_QK_FIT_CASES
        or len(set(trajectories)) != len(trajectories)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint execution lacks the exact ordered H776 Q/K-FIT inventory."
        )
    for index, trajectory in enumerate(trajectories):
        _digest(trajectory, f"endpoint fit trajectory {index}")
    execution = _self_sealed(
        proposal.get("observed_execution_receipt"),
        seal_field="receipt_sha256",
        name="observed endpoint execution",
    )
    observations = execution.get("compute_observations")
    summaries = execution.get("compute_device_summaries")
    solver = execution.get("solver_observation")
    resource = execution.get("resource_observation")
    static_shards = _self_sealed(
        execution.get("static_device_shard_receipt"),
        seal_field="static_device_shards_sha256",
        name="endpoint static device shards",
    )
    numerical_sweep_count = execution.get("numerical_sweep_count")
    numerical_normal_calls = execution.get("numerical_normal_operator_calls")
    if (
        proposal.get("solver_device") != "cuda:1"
        or proposal.get("compute_devices") != expected_devices
        or proposal.get("case_assignment")
        != "authenticated_bank_order_round_robin"
        or proposal.get("fit_check_accessible") is not False
        or proposal.get("fit_check_cases_consumed") != 0
        or execution.get("schema") != LFM2_ENDPOINT_QK_EXECUTION_SCHEMA
        or execution.get("requested_solver_device") != "cuda:1"
        or execution.get("requested_compute_devices") != expected_devices
        or execution.get("requested_compute_device_count") != 2
        or execution.get("case_assignment")
        != "authenticated_bank_order_round_robin"
        or execution.get("compute_device_concurrency")
        != "serial_static_device_shards"
        or execution.get("qk_transfer_execution") != expected_transfer_execution
        or execution.get("every_requested_compute_device_observed") is not True
        or execution.get("distinct_cuda_uuid_count") != 2
        or type(numerical_sweep_count) is not int
        or numerical_sweep_count < 1
        or type(numerical_normal_calls) is not int
        or numerical_normal_calls < 1
        or execution.get("total_semantic_probe_calls") != len(trajectories)
        or not isinstance(observations, list)
        or len(observations) != len(trajectories)
        or not isinstance(summaries, list)
        or len(summaries) != 2
        or not isinstance(resource, dict)
        or resource.get("fit_case_count") != len(trajectories)
        or resource.get("fixed_tensor_cache_count") != 2
        or resource.get("shared_fixed_tensor_cache_bytes")
        != 2 * expected_fixed_cache_bytes
        or resource.get("conservative_static_shard_workspace_bytes_per_device")
        != expected_static_shard_workspace_bytes
        or resource.get("byte_accounting_scope")
        != "deterministic_declared_tensor_bytes_not_allocator_peak"
        or resource.get("production_allocator_peak_authority")
        != "outer_real_probe_continuous_cuda_peak_reserved_monitor"
        or resource.get("within_all_resource_ceilings") is not True
        or resource.get("every_compute_device_within_byte_ceiling") is not True
        or not isinstance(execution.get("operation_audit"), dict)
        or any(
            type(value) is not int or value != 0
            for value in execution["operation_audit"].values()
        )
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint execution is not the exact FIT-only two-3090 contract."
        )
    static_entries = static_shards.get("shards")
    if (
        static_shards.get("schema")
        != "neural-forge-lfm2-endpoint-static-device-shards-v1"
        or static_shards.get("assignment")
        != "authenticated_bank_order_round_robin_static_shards"
        or static_shards.get("device_order") != expected_devices
        or not isinstance(static_entries, list)
        or len(static_entries) != 2
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint execution lacks its authenticated static device shards."
        )
    authenticated_observations = []
    for index, (observation, trajectory) in enumerate(
        zip(observations, trajectories, strict=True)
    ):
        authenticated = _self_sealed(
            observation,
            seal_field="case_execution_receipt_sha256",
            name=f"observed endpoint FIT case {index}",
        )
        expected_device_index = index % 2
        expected_device = expected_devices[expected_device_index]
        scalar = authenticated.get("analytical_scalar")
        if (
            authenticated.get("assignment_index") != index
            or authenticated.get("intended_logical_device") != expected_device
            or authenticated.get("observed_logical_device") != expected_device
            or authenticated.get("observed_device_type") != "cuda"
            or authenticated.get("observed_device_index") != expected_device_index
            or authenticated.get("cuda_uuid") != expected_uuids[expected_device_index]
            or authenticated.get("cuda_name") != H776_COMPOSED_CUDA_DEVICE_NAME
            or authenticated.get("trajectory_sha256") != trajectory
            or authenticated.get("analytical_operation")
            != "exact_endpoint_jvp_then_vjp_normal_probe"
            or type(authenticated.get("exact_replay_calls")) is not int
            or authenticated["exact_replay_calls"] < 1
            or authenticated.get("endpoint_jvp_calls")
            != numerical_normal_calls + 1
            or authenticated.get("endpoint_vjp_calls")
            != numerical_normal_calls + numerical_sweep_count + 1
            or authenticated.get("normal_operator_calls")
            != numerical_normal_calls + 1
            or authenticated.get("semantic_probe_calls") != 1
            or type(authenticated.get("prepared_persistent_tensor_bytes")) is not int
            or authenticated["prepared_persistent_tensor_bytes"] <= 0
            or type(authenticated.get("attached_replay_tensor_bytes")) is not int
            or authenticated["attached_replay_tensor_bytes"] <= 0
            or type(authenticated.get("semantic_probe_tensor_bytes")) is not int
            or authenticated["semantic_probe_tensor_bytes"] <= 0
            or type(authenticated.get("analytical_tensor_element_count")) is not int
            or authenticated["analytical_tensor_element_count"] <= 0
            or isinstance(scalar, bool)
            or not isinstance(scalar, (int, float))
            or not math.isfinite(float(scalar))
            or float(scalar) <= 0.0
            or authenticated.get("forced_host_scalar_read") is not True
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                f"Endpoint FIT case {index} lacks real semantic work on "
                f"{expected_device}."
            )
        for name in (
            "evidence_sha256",
            "trajectory_sha256",
            "target_receipt_sha256",
            "semantic_delta_probability_sha256",
            "semantic_delta_endpoint_sha256",
            "semantic_query_gradient_sha256",
            "semantic_key_gradient_sha256",
        ):
            _digest(authenticated.get(name), f"endpoint case {index} {name}")
        authenticated_observations.append(authenticated)
    exact_replay_calls = authenticated_observations[0]["exact_replay_calls"]
    if any(
        observation["exact_replay_calls"] != exact_replay_calls
        for observation in authenticated_observations
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint FIT cases disagree on exact replay round count."
        )
    for device_index, summary in enumerate(summaries):
        expected_device = expected_devices[device_index]
        expected_indices = list(range(device_index, len(trajectories), 2))
        static_entry = static_entries[device_index]
        if not isinstance(static_entry, Mapping):
            raise Lfm2CpcfV5RealAttentionProbeError(
                f"Endpoint static shard {device_index} is malformed."
            )
        fixed_cache = _self_sealed(
            static_entry.get("fixed_cache_receipt"),
            seal_field="fixed_cache_receipt_sha256",
            name=f"endpoint {expected_device} fixed cache",
        )
        summary_fixed_cache = _self_sealed(
            summary.get("fixed_cache_receipt")
            if isinstance(summary, Mapping)
            else None,
            seal_field="fixed_cache_receipt_sha256",
            name=f"endpoint {expected_device} summary fixed cache",
        )
        matching_observations = [
            authenticated_observations[index] for index in expected_indices
        ]
        prepared_bytes = sum(
            int(value["prepared_persistent_tensor_bytes"])
            for value in matching_observations
        )
        replay_bytes = sum(
            int(value["attached_replay_tensor_bytes"])
            for value in matching_observations
        )
        maximum_probe_bytes = max(
            int(value["semantic_probe_tensor_bytes"])
            for value in matching_observations
        )
        expected_device_solver_workspace = (
            expected_solver_workspace_bytes if device_index == 1 else 0
        )
        expected_peak_bytes = (
            prepared_bytes
            + replay_bytes
            + maximum_probe_bytes
            + expected_fixed_cache_bytes
            + expected_static_shard_workspace_bytes
            + expected_device_solver_workspace
        )
        if (
            not isinstance(summary, dict)
            or static_entry.get("logical_device") != expected_device
            or static_entry.get("case_assignment_indices") != expected_indices
            or static_entry.get("case_count") != len(expected_indices)
            or static_entry.get("all_cases_share_fixed_cache_by_identity") is not True
            or fixed_cache != summary_fixed_cache
            or fixed_cache.get("schema")
            != "neural-forge-lfm2-endpoint-fixed-device-cache-v1"
            or fixed_cache.get("logical_device") != expected_device
            or fixed_cache.get("shared_fixed_tensor_bytes")
            != expected_fixed_cache_bytes
            or fixed_cache.get("fixed_cache_builds") != 1
            or fixed_cache.get("fixed_tensor_materialization")
            != "once_per_requested_compute_device"
            or summary.get("logical_device") != expected_device
            or summary.get("case_assignment_indices") != expected_indices
            or summary.get("case_count") != len(expected_indices)
            or summary.get("exact_replay_calls")
            != len(expected_indices) * exact_replay_calls
            or summary.get("endpoint_jvp_calls")
            != len(expected_indices) * (numerical_normal_calls + 1)
            or summary.get("endpoint_vjp_calls")
            != len(expected_indices)
            * (numerical_normal_calls + numerical_sweep_count + 1)
            or summary.get("normal_operator_calls")
            != len(expected_indices) * (numerical_normal_calls + 1)
            or summary.get("semantic_probe_calls") != len(expected_indices)
            or summary.get("cuda_uuid") != expected_uuids[device_index]
            or summary.get("cuda_name") != H776_COMPOSED_CUDA_DEVICE_NAME
            or summary.get("prepared_persistent_tensor_bytes") != prepared_bytes
            or summary.get("attached_replay_tensor_bytes") != replay_bytes
            or summary.get("shared_fixed_tensor_cache_bytes")
            != expected_fixed_cache_bytes
            or summary.get("replay_qk_shard_broadcasts") != exact_replay_calls
            or summary.get("normal_qk_direction_shard_broadcasts")
            != numerical_normal_calls
            or summary.get("rhs_gradient_shard_reductions")
            != numerical_sweep_count
            or summary.get("normal_gradient_shard_reductions")
            != numerical_normal_calls
            or summary.get("semantic_direction_shard_builds") != 1
            or summary.get("semantic_gradient_shard_reductions") != 1
            or summary.get("qk_transfer_execution") != expected_transfer_execution
            or summary.get("maximum_serial_semantic_probe_tensor_bytes")
            != maximum_probe_bytes
            or summary.get("conservative_static_shard_workspace_bytes")
            != expected_static_shard_workspace_bytes
            or summary.get("conservative_solver_workspace_bytes")
            != expected_device_solver_workspace
            or summary.get("conservative_peak_accounted_bytes")
            != expected_peak_bytes
            or summary.get("byte_accounting_scope")
            != "deterministic_declared_tensor_bytes_not_allocator_peak"
            or summary.get("maximum_bytes_per_compute_device")
            != pins.maximum_cuda_bytes_per_device
            or summary.get("within_compute_device_byte_ceiling") is not True
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                f"Endpoint execution lacks the exact {expected_device} summary."
            )
        for name in (
            "fixed_value_bfloat16_sha256",
            "fixed_output_float64_sha256",
            "query_rms_bfloat16_sha256",
            "key_rms_bfloat16_sha256",
            "query_rms_float32_sha256",
            "key_rms_float32_sha256",
        ):
            _digest(fixed_cache.get(name), f"{expected_device} fixed cache {name}")
    if (
        execution.get("total_exact_replay_calls")
        != sum(int(value["exact_replay_calls"]) for value in authenticated_observations)
        or execution.get("total_endpoint_jvp_calls")
        != sum(int(value["endpoint_jvp_calls"]) for value in authenticated_observations)
        or execution.get("total_endpoint_vjp_calls")
        != sum(int(value["endpoint_vjp_calls"]) for value in authenticated_observations)
        or execution.get("total_normal_operator_calls")
        != sum(
            int(value["normal_operator_calls"])
            for value in authenticated_observations
        )
        or resource.get("prepared_persistent_tensor_bytes")
        != sum(
            int(value["prepared_persistent_tensor_bytes"])
            for value in authenticated_observations
        )
        or resource.get("attached_replay_tensor_bytes")
        != sum(
            int(value["attached_replay_tensor_bytes"])
            for value in authenticated_observations
        )
        or resource.get("maximum_serial_semantic_probe_tensor_bytes")
        != max(
            int(value["semantic_probe_tensor_bytes"])
            for value in authenticated_observations
        )
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint aggregate execution/resource counters do not match "
            "the authenticated static shards."
        )
    authenticated_solver = _self_sealed(
        solver,
        seal_field="solver_execution_receipt_sha256",
        name="observed endpoint solver",
    )
    solver_scalar = authenticated_solver.get("analytical_scalar")
    if (
        authenticated_solver.get("intended_logical_device") != "cuda:1"
        or authenticated_solver.get("observed_logical_device") != "cuda:1"
        or authenticated_solver.get("observed_device_type") != "cuda"
        or authenticated_solver.get("observed_device_index") != 1
        or authenticated_solver.get("cuda_uuid") != expected_uuids[1]
        or authenticated_solver.get("cuda_name") != H776_COMPOSED_CUDA_DEVICE_NAME
        or authenticated_solver.get("analytical_operation")
        != "dense_seed_qk_plus_worker_normal_sum_energy_and_parameter_count"
        or isinstance(solver_scalar, bool)
        or not isinstance(solver_scalar, (int, float))
        or not math.isfinite(float(solver_scalar))
        or float(solver_scalar) <= 0.0
        or authenticated_solver.get("forced_host_scalar_read") is not True
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint solver lacks observed dense Q/K plus worker-normal work."
        )
    body = {
        "observed_execution_receipt_sha256": execution["receipt_sha256"],
        "logical_compute_devices": expected_devices,
        "physical_cuda_uuids": expected_uuids,
        "cuda_device_name": H776_COMPOSED_CUDA_DEVICE_NAME,
        "fit_case_count": len(trajectories),
        "per_device_case_counts": {
            "cuda:0": len(range(0, len(trajectories), 2)),
            "cuda:1": len(range(1, len(trajectories), 2)),
        },
        "semantic_probe_calls": len(trajectories),
        "solver_device": "cuda:1",
        "static_device_shards_sha256": static_shards[
            "static_device_shards_sha256"
        ],
        "fixed_tensor_cache_count": 2,
        "shared_fixed_tensor_cache_bytes": 2 * expected_fixed_cache_bytes,
        "qk_transfer_execution": expected_transfer_execution,
        "once_per_shard_transfer_counters_authenticated": True,
        "deterministic_logical_byte_accounting_authenticated": True,
        "allocator_peak_authority": (
            "outer_real_probe_continuous_cuda_peak_reserved_monitor"
        ),
        "fold_only_execution_authenticated": True,
    }
    return MappingProxyType({**body, "audit_sha256": _sha256(body)})


def authenticate_h776_composed_attention_result(
    result: Any,
    *,
    input_bank_sha256: str,
    expected_fit_bank_sha256: str,
    expected_fit_trajectory_sha256s: Sequence[str],
    pins: Lfm2CpcfV5RealAttentionProbePins,
) -> Mapping[str, Any]:
    """Authenticate the opt-in four-arm result without the legacy PCG observer."""

    from .hf_lfm2_composed_qk_attention_fold import (
        LFM2_COMPOSED_QK_PRODUCER_SCHEMA,
        LFM2_COMPOSED_QK_SEQUENCE_EFFECTIVE_SCHEMA,
        Lfm2CompiledComposedQkSequenceEffectiveAttentionFold,
    )
    from .hf_lfm2_cpcf_layer_compiler import (
        DIRECT_MONOLITHIC_ATTENTION_CANDIDATE_ID,
        RESIDUAL_ATTENTION_SPLIT_CANDIDATE_ID,
        Lfm2CompiledCpcfResidualAttentionSplitFold,
        _tensor_bundle_sha256,
    )
    from .hf_lfm2_sequence_effective_attention_fold import (
        Lfm2CompiledFixedQkSequenceEffectiveAttentionFold,
    )
    from .hf_lfm2_structural_residual_attention_fold import (
        Lfm2CompiledStructuralAttentionFold,
        _state_sha256 as _structural_attention_state_sha256,
    )

    if pins.expected_composed_qk_attention_enabled is not True:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Composed attention authentication was not independently pinned."
        )
    input_bank_sha = _digest(input_bank_sha256, "input_bank_sha256")
    fit_bank_sha = _digest(
        expected_fit_bank_sha256,
        "expected_fit_bank_sha256",
    )
    fit_trajectory_sha256s = tuple(expected_fit_trajectory_sha256s)
    if len(fit_trajectory_sha256s) != H776_COMPOSED_UPSTREAM_FIT_CASES or len(
        set(fit_trajectory_sha256s)
    ) != len(fit_trajectory_sha256s):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The independent H776 FIT trajectory inventory is incomplete."
        )
    for index, value in enumerate(fit_trajectory_sha256s):
        _digest(value, f"expected_fit_trajectory_sha256s[{index}]")
    report = _self_sealed(
        getattr(result, "report", None),
        seal_field="result_sha256",
        name="composed layer result",
    )
    policy = report.get("solver_evidence_policy")
    if not isinstance(policy, dict):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Composed layer result lacks solver evidence policy."
        )
    selection = _self_sealed(
        policy.get("attention_candidate_selection"),
        seal_field="report_sha256",
        name="composed attention candidate selection",
    )
    evaluations = policy.get("attention_candidate_evaluations")
    if not isinstance(evaluations, dict):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Composed layer result lacks four candidate evaluations."
        )
    candidate_ids = tuple(selection.get("candidate_ids", ()))
    selected_id = selection.get("selected_candidate_id")
    if (
        report.get("layer_index") != H776_LAYER_INDEX
        or report.get("operator_kind") != "full_attention"
        or report.get("input_bank_sha256") != input_bank_sha
        or report.get("split_phase_counts") != dict(H776_SPLIT_PHASE_COUNTS)
        or policy.get("operator_case_count") != H776_COMPOSED_UPSTREAM_FIT_CASES
        or candidate_ids != H776_COMPOSED_ATTENTION_CANDIDATE_IDS
        or set(evaluations) != set(H776_COMPOSED_ATTENTION_CANDIDATE_IDS)
        or selection.get("candidate_count") != 4
        or selection.get("selection_split_only") is not True
        or selection.get("winner_frozen_before_validation") is not True
        or selection.get("validation_phase_count_consumed_before_freeze") != 0
        or policy.get("attention_candidate_inventory_mode")
        != "opt_in_authenticated_four_arm"
        or policy.get("attention_each_candidate_has_own_fit_only_mlp") is not True
        or policy.get("attention_winner_frozen_before_validation") is not True
        or policy.get("attention_validation_replays_before_winner_freeze") != 0
        or {
            "attention_candidate_local_post_layer_transport",
            "attention_sequential_transport_policy",
            "attention_each_candidate_prepares_own_mlp_basis",
            "attention_cross_candidate_prepared_mlp_basis_reuse",
            "attention_final_endpoint_effective_boundary_charts_sha256",
        }.intersection(policy)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Composed tournament, split isolation, or layer identity changed."
        )

    authenticated_evaluations: dict[str, dict[str, Any]] = {}
    selection_keys: dict[str, tuple[Any, ...]] = {}
    behavioral_keys: dict[str, tuple[float, ...]] = {}
    candidate_report_sha256 = selection.get("candidate_report_sha256")
    if not isinstance(candidate_report_sha256, dict) or set(
        candidate_report_sha256
    ) != set(H776_COMPOSED_ATTENTION_CANDIDATE_IDS):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Composed selection lacks the exact candidate-report inventory."
        )
    for candidate_id in H776_COMPOSED_ATTENTION_CANDIDATE_IDS:
        evaluation = _self_sealed(
            evaluations.get(candidate_id),
            seal_field="report_sha256",
            name=f"composed attention candidate {candidate_id!r}",
        )
        raw_key = evaluation.get("selection_key")
        if (
            evaluation.get("candidate_id") != candidate_id
            or evaluation.get("operator_compiled_from_splits") != ["fit"]
            or evaluation.get("mlp_compiled_from_splits") != ["fit"]
            or evaluation.get("candidate_evaluated_on_splits") != ["selection"]
            or evaluation.get("validation_phase_count_consumed") != 0
            or evaluation.get("training_steps") != 0
            or evaluation.get("parameters_pruned") != 0
            or evaluation.get("parameters_quantized") != 0
            or candidate_report_sha256.get(candidate_id)
            != evaluation.get("report_sha256")
            or {
                "sequential_boundary_transport",
                "sequential_transport_binding",
                "effective_boundary_charts_sha256",
                "operator_transport_mlp_bundle_sha256",
            }.intersection(evaluation)
            or not isinstance(raw_key, list)
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
                f"Composed candidate {candidate_id!r} crossed a split, "
                "mutation, or behavioral-ranking gate."
            )
        authenticated_evaluations[candidate_id] = evaluation
        selection_keys[candidate_id] = tuple(raw_key)
        behavioral_keys[candidate_id] = tuple(float(value) for value in raw_key[:-1])
    expected_selected_id = min(
        H776_COMPOSED_ATTENTION_CANDIDATE_IDS,
        key=lambda candidate_id: selection_keys[candidate_id],
    )
    if (
        selected_id != expected_selected_id
        or selection.get("selected_candidate_report_sha256")
        != candidate_report_sha256[selected_id]
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Composed winner does not reproduce from the sealed ranking keys."
        )

    composed_evaluation = authenticated_evaluations[
        H776_COMPOSED_ATTENTION_CANDIDATE_ID
    ]
    outer = _self_sealed(
        composed_evaluation.get("operator_artifact"),
        seal_field="artifact_sha256",
        name="outer composed attention artifact",
    )
    producer = _self_sealed(
        policy.get("composed_qk_producer_artifact"),
        seal_field="artifact_sha256",
        name="composed Q/K producer artifact",
    )
    global_sync = _self_sealed(
        policy.get("composed_qk_global_sync"),
        seal_field="report_sha256",
        name="composed global Q/K synchronization",
    )
    score = _self_sealed(
        policy.get("composed_qk_score_result"),
        seal_field="report_sha256",
        name="composed score-residual result",
    )
    composed_policy = policy.get("composed_qk_attention_policy")
    if not isinstance(composed_policy, dict):
        raise Lfm2CpcfV5RealAttentionProbeError("Composed Q/K policy is absent.")
    expected_composed_policy = _plain(_h776_composed_qk_attention_policy().as_dict())
    expected_policy_sha = _digest(
        expected_composed_policy.get("authentication_sha256"),
        "expected composed Q/K policy authentication",
    )
    composed_policy_body = dict(composed_policy)
    composed_policy_sha = _digest(
        composed_policy_body.pop("authentication_sha256", None),
        "composed Q/K policy authentication",
    )
    score_policy = composed_policy.get("score_policy")
    global_policy = composed_policy.get("global_policy")
    if (
        _sha256(composed_policy_body) != composed_policy_sha
        or expected_policy_sha != H776_COMPOSED_QK_ATTENTION_POLICY_SHA256
        or expected_policy_sha != pins.expected_composed_qk_attention_policy_sha256
        or composed_policy_sha != expected_policy_sha
        or composed_policy != expected_composed_policy
        or not isinstance(score_policy, dict)
        or not isinstance(global_policy, dict)
        or score_policy.get("solver_device") != "cuda:1"
        or score_policy.get("compute_devices") != ["cuda:0", "cuda:1"]
        or composed_policy.get("training_allowed") is not False
        or producer.get("policy") != expected_composed_policy
        or producer.get("policy_sha256") != expected_policy_sha
        or score.get("policy") != score_policy
        or global_sync.get("policy") != global_policy
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Composed policy authentication or two-3090 routing changed."
        )
    producer_seal = producer["artifact_sha256"]
    global_seal = global_sync["report_sha256"]
    score_seal = score["report_sha256"]
    outer_operation = outer.get("operation_audit")
    producer_operation = producer.get("operation_audit")
    score_operation = score.get("operation_audit")
    expected_trajectory_bundle_sha = _sha256(list(fit_trajectory_sha256s))
    qk_fit_ordinals = producer.get("qk_fit_source_ordinals")
    qk_fit_check_ordinals = producer.get("qk_fit_check_source_ordinals")
    if (
        outer.get("schema") != LFM2_COMPOSED_QK_SEQUENCE_EFFECTIVE_SCHEMA
        or producer.get("schema") != LFM2_COMPOSED_QK_PRODUCER_SCHEMA
        or outer.get("candidate_id") != H776_COMPOSED_ATTENTION_CANDIDATE_ID
        or producer.get("candidate_id") != H776_COMPOSED_ATTENTION_CANDIDATE_ID
        or producer.get("source_split_sha256") != fit_bank_sha
        or producer.get("upstream_fit_case_count") != H776_COMPOSED_UPSTREAM_FIT_CASES
        or producer.get("upstream_trajectory_sha256s") != list(fit_trajectory_sha256s)
        or producer.get("upstream_trajectory_bundle_sha256")
        != expected_trajectory_bundle_sha
        or outer.get("upstream_trajectory_sha256s") != list(fit_trajectory_sha256s)
        or outer.get("upstream_trajectory_bundle_sha256")
        != expected_trajectory_bundle_sha
        or outer.get("upstream_fit_case_bundle_sha256")
        != producer.get("upstream_fit_case_bundle_sha256")
        or not isinstance(qk_fit_ordinals, list)
        or not isinstance(qk_fit_check_ordinals, list)
        or len(qk_fit_ordinals) != H776_COMPOSED_QK_FIT_CASES
        or len(qk_fit_check_ordinals) != H776_COMPOSED_QK_FIT_CHECK_CASES
        or any(type(value) is not int for value in qk_fit_ordinals)
        or any(type(value) is not int for value in qk_fit_check_ordinals)
        or set(qk_fit_ordinals) & set(qk_fit_check_ordinals)
        or set(qk_fit_ordinals) | set(qk_fit_check_ordinals)
        != set(range(H776_COMPOSED_UPSTREAM_FIT_CASES))
        or outer.get("composed_qk_artifact_sha256") != producer_seal
        or policy.get("composed_qk_producer_artifact_sha256") != producer_seal
        or producer.get("global_sync_report_sha256") != global_seal
        or producer.get("score_result_report_sha256") != score_seal
        or composed_evaluation.get("query_key_score_state_sha256")
        != outer.get("fixed_query_key_sha256")
        or composed_evaluation.get("operator_artifact_sha256") != _sha256(outer)
        or score.get("accepted") is not True
        or score.get("status") not in {"fit_check_accepted", "quality_gated_seed_no_op"}
        or score.get("fit_check_evaluated") is not True
        or score.get("fit_check_triggered_refit") is not False
        or score.get("source_evidence_unchanged") is not True
        or not isinstance(score_operation, dict)
        or score_operation.get("fit_cases_consumed") != H776_COMPOSED_QK_FIT_CASES
        or score_operation.get("fit_check_cases_consumed")
        != H776_COMPOSED_QK_FIT_CHECK_CASES
        or not isinstance(outer_operation, dict)
        or any(
            type(value) is not int or value != 0 for value in outer_operation.values()
        )
        or not isinstance(producer_operation, dict)
        or any(
            type(value) is not int or value != 0
            for value in producer_operation.values()
        )
        or any(
            type(score_operation.get(name)) is not int or score_operation.get(name) != 0
            for name in (
                "selection_cases_consumed",
                "true_validation_cases_consumed",
                "backward_calls",
                "optimizer_steps",
                "training_steps",
                "model_forward_calls",
                "model_generate_calls",
                "parameters_pruned",
                "parameters_quantized",
                "target_mutations_during_compile",
            )
        )
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Composed artifacts are not cross-bound, accepted, or fold-only."
        )
    fit_metrics = _authenticate_qk_routing_metrics(
        score.get("fit_metrics"),
        policy=score_policy,
        partition="fit",
        expected_case_count=H776_COMPOSED_QK_FIT_CASES,
    )
    fit_check_metrics = _authenticate_qk_routing_metrics(
        score.get("fit_check_metrics"),
        policy=score_policy,
        partition="fit_check",
        expected_case_count=H776_COMPOSED_QK_FIT_CHECK_CASES,
    )
    device_execution = _authenticate_h776_composed_device_execution(
        score,
        producer=producer,
        global_sync=global_sync,
        fit_trajectory_sha256s=fit_trajectory_sha256s,
        pins=pins,
    )
    resource_usage = global_sync.get("resource_usage")
    resource_limits = (
        ("total_rows", "maximum_total_rows"),
        ("total_score_elements", "maximum_total_score_elements"),
        ("total_factor_elements", "maximum_total_factor_elements"),
        ("total_output_elements", "maximum_total_output_elements"),
        (
            "total_svd_workspace_elements",
            "maximum_total_svd_workspace_elements",
        ),
    )
    if not isinstance(resource_usage, dict):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Composed global synchronization exceeded its authenticated budget."
        )
    for metric, limit in resource_limits:
        observed = resource_usage.get(metric)
        maximum = global_policy.get(limit)
        if (
            type(observed) is not int
            or observed < 0
            or type(maximum) is not int
            or maximum <= 0
            or observed > maximum
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Composed global synchronization exceeded its authenticated "
                f"budget for {metric}."
            )
    selected_type_by_id = {
        DIRECT_MONOLITHIC_ATTENTION_CANDIDATE_ID: (Lfm2CompiledStructuralAttentionFold),
        RESIDUAL_ATTENTION_SPLIT_CANDIDATE_ID: (
            Lfm2CompiledCpcfResidualAttentionSplitFold
        ),
        H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID: (
            Lfm2CompiledFixedQkSequenceEffectiveAttentionFold
        ),
        H776_COMPOSED_ATTENTION_CANDIDATE_ID: (
            Lfm2CompiledComposedQkSequenceEffectiveAttentionFold
        ),
    }
    selected_artifact = getattr(result, "operator_artifact", None)
    expected_selected_type = selected_type_by_id[selected_id]
    expected_selected_report = authenticated_evaluations[selected_id].get(
        "operator_artifact"
    )
    if (
        not isinstance(selected_artifact, expected_selected_type)
        or not callable(getattr(selected_artifact, "as_dict", None))
        or not callable(getattr(selected_artifact, "state_dict", None))
        or _plain(selected_artifact.as_dict()) != expected_selected_report
        or report.get("operator_artifact") != expected_selected_report
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The live selected attention artifact is absent, has the wrong "
            "typed arm, or differs from the frozen winner."
        )
    if isinstance(selected_artifact, Lfm2CompiledStructuralAttentionFold):
        live_state_sha = _structural_attention_state_sha256(
            selected_artifact.state_dict()
        )
        if (
            live_state_sha != selected_artifact.compiled_state_sha256
            or live_state_sha != expected_selected_report.get("compiled_state_sha256")
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "The live direct attention winner tensors differ from their "
                "compiled-state receipt."
            )
    elif isinstance(
        selected_artifact,
        Lfm2CompiledCpcfResidualAttentionSplitFold,
    ):
        live_state_sha = _tensor_bundle_sha256(selected_artifact.state_dict())
        if (
            live_state_sha != selected_artifact.compiled_state_sha256
            or live_state_sha != expected_selected_report.get("compiled_state_sha256")
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                "The live residual-attention winner tensors differ from their "
                "compiled-state receipt."
            )
    if (
        isinstance(
            selected_artifact,
            Lfm2CompiledFixedQkSequenceEffectiveAttentionFold,
        )
        and selected_artifact.verify_authentication() is not True
    ) or (
        isinstance(
            selected_artifact,
            Lfm2CompiledComposedQkSequenceEffectiveAttentionFold,
        )
        and selected_artifact.verify_authentication()
        != selected_artifact.artifact_sha256
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The live selected sequence artifact failed typed authentication."
        )
    next_bank = getattr(result, "next_bank", None)
    if (
        next_bank is None
        or getattr(next_bank, "layer_index", None) != H776_LAYER_INDEX + 1
        or report.get("next_bank_sha256")
        != getattr(next_bank, "state_bank_sha256", None)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Composed layer result did not create the layer-3 input bank."
        )
    composed_won = selected_id == H776_COMPOSED_ATTENTION_CANDIDATE_ID
    composed_strictly_better = all(
        behavioral_keys[H776_COMPOSED_ATTENTION_CANDIDATE_ID]
        < behavioral_keys[candidate_id]
        for candidate_id in H776_ATTENTION_CANDIDATE_IDS
    )
    body = {
        "schema": LFM2_CPCF_V5_REAL_COMPOSED_ATTENTION_AUDIT_SCHEMA,
        "layer_result_sha256": report["result_sha256"],
        "input_bank_sha256": input_bank_sha,
        "fit_bank_sha256": fit_bank_sha,
        "fit_trajectory_sha256s": list(fit_trajectory_sha256s),
        "fit_trajectory_bundle_sha256": expected_trajectory_bundle_sha,
        "next_bank_sha256": report["next_bank_sha256"],
        "candidate_ids": list(candidate_ids),
        "selected_candidate_id": selected_id,
        "selection_keys": {
            candidate_id: list(selection_keys[candidate_id])
            for candidate_id in candidate_ids
        },
        "behavioral_selection_keys": {
            candidate_id: list(behavioral_keys[candidate_id])
            for candidate_id in candidate_ids
        },
        "composed_qk_won": composed_won,
        "composed_qk_strictly_better_than_all_legacy_arms": (composed_strictly_better),
        "scientific_outcome": (
            "positive_composed_qk_selection"
            if composed_won and composed_strictly_better
            else "honest_negative_or_legacy_arm_selected"
        ),
        "full_validation_metrics": _plain(report["split_metrics"]["validation"]),
        "composed_outer_artifact": outer,
        "composed_producer_artifact": producer,
        "global_sync": global_sync,
        "score_result": score,
        "composed_qk_attention_policy_sha256": expected_policy_sha,
        "fit_routing_metrics": fit_metrics,
        "fit_check_routing_metrics": fit_check_metrics,
        "score_device_execution": _plain(device_execution),
        "two_3090_score_execution_authenticated": True,
        "selected_live_operator_artifact_bound": True,
        "in_memory_next_bank_created_by_layer_replay": True,
        "k3_checkpoint_published": False,
        "layer_3_compiled": False,
        "automatic_retry_or_retuning_performed": False,
    }
    return MappingProxyType({**body, "audit_sha256": _sha256(body)})


def authenticate_h776_endpoint_residual_attention_result(
    result: Any,
    *,
    input_bank_sha256: str,
    expected_fit_bank_sha256: str,
    expected_fit_trajectory_sha256s: Sequence[str],
    pins: Lfm2CpcfV5RealAttentionProbePins,
) -> Mapping[str, Any]:
    """Authenticate the real independently gated endpoint H776 transaction."""

    from .hf_lfm2_composed_qk_attention_fold import (
        ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID,
        LFM2_COMPOSED_ENDPOINT_RESIDUAL_BLOCK_SCHEMA,
        LFM2_COMPOSED_ENDPOINT_RESIDUAL_SCHEMA,
        LFM2_COMPOSED_QK_PRODUCER_SCHEMA,
        LFM2_COMPOSED_QK_STAGE_ONE_ATTEMPT_SCHEMA,
        LFM2_ENDPOINT_QK_START_SCHEMA,
        Lfm2CompiledComposedEndpointResidualAttentionFold,
        Lfm2CompiledComposedQkSequenceEffectiveAttentionFold,
    )
    from .hf_lfm2_cpcf_layer_compiler import (
        DIRECT_MONOLITHIC_ATTENTION_CANDIDATE_ID,
        RESIDUAL_ATTENTION_SPLIT_CANDIDATE_ID,
        Lfm2CompiledCpcfResidualAttentionSplitFold,
        _tensor_bundle_sha256,
    )
    from .hf_lfm2_sequence_effective_attention_fold import (
        Lfm2CompiledFixedQkSequenceEffectiveAttentionFold,
    )
    from .hf_lfm2_structural_residual_attention_fold import (
        Lfm2CompiledStructuralAttentionFold,
        _state_sha256 as _structural_attention_state_sha256,
    )

    if (
        pins.expected_endpoint_residual_attention_enabled is not True
        or pins.expected_composed_qk_attention_enabled is not False
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint-residual authentication was not independently pinned."
        )
    input_bank_sha = _digest(input_bank_sha256, "input_bank_sha256")
    fit_bank_sha = _digest(expected_fit_bank_sha256, "expected_fit_bank_sha256")
    full_fit_trajectories = tuple(expected_fit_trajectory_sha256s)
    if (
        len(full_fit_trajectories) != H776_COMPOSED_UPSTREAM_FIT_CASES
        or len(set(full_fit_trajectories)) != len(full_fit_trajectories)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "The endpoint lane lacks the exact independent H776 FIT inventory."
        )
    for index, trajectory in enumerate(full_fit_trajectories):
        _digest(trajectory, f"endpoint upstream FIT trajectory {index}")

    report = _self_sealed(
        getattr(result, "report", None),
        seal_field="result_sha256",
        name="endpoint-residual layer result",
    )
    policy = report.get("solver_evidence_policy")
    if not isinstance(policy, dict):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint-residual layer result lacks solver evidence."
        )
    selection = _self_sealed(
        policy.get("attention_candidate_selection"),
        seal_field="report_sha256",
        name="endpoint-residual attention candidate selection",
    )
    evaluations = policy.get("attention_candidate_evaluations")
    candidate_ids = tuple(selection.get("candidate_ids", ()))
    candidate_inventory_is_five_arm = (
        candidate_ids == H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS
    )
    candidate_inventory_is_endpoint_without_composed = (
        candidate_ids == H776_ENDPOINT_WITHOUT_COMPOSED_ATTENTION_CANDIDATE_IDS
    )
    composed_candidate_eligible = candidate_inventory_is_five_arm
    expected_inventory_mode = (
        "opt_in_authenticated_five_arm"
        if composed_candidate_eligible
        else (
            "opt_in_authenticated_endpoint_four_arm_without_rejected_"
            "composed_candidate"
        )
    )
    selected_id = selection.get("selected_candidate_id")
    if (
        report.get("layer_index") != H776_LAYER_INDEX
        or report.get("operator_kind") != "full_attention"
        or report.get("input_bank_sha256") != input_bank_sha
        or report.get("split_phase_counts") != dict(H776_SPLIT_PHASE_COUNTS)
        or policy.get("operator_case_count") != H776_COMPOSED_UPSTREAM_FIT_CASES
        or not (
            candidate_inventory_is_five_arm
            or candidate_inventory_is_endpoint_without_composed
        )
        or not isinstance(evaluations, dict)
        or set(evaluations) != set(candidate_ids)
        or selection.get("candidate_count") != len(candidate_ids)
        or selection.get("candidate_local_mlp_preparation_count")
        != len(candidate_ids)
        or selection.get("cross_candidate_prepared_mlp_basis_reused") is not False
        or selection.get("endpoint_residual_candidate_fit_only_mlp_evaluated")
        is not True
        or selection.get("selection_split_only") is not True
        or selection.get("winner_frozen_before_validation") is not True
        or selection.get("validation_phase_count_consumed_before_freeze") != 0
        or policy.get("attention_candidate_inventory_mode")
        != expected_inventory_mode
        or policy.get("attention_each_candidate_has_own_fit_only_mlp") is not True
        or policy.get("attention_each_candidate_prepares_own_mlp_basis") is not True
        or policy.get("attention_cross_candidate_prepared_mlp_basis_reuse")
        is not False
        or policy.get("endpoint_residual_fit_only_downstream_mlp_evaluation")
        is not True
        or policy.get("endpoint_residual_fit_check_final_one_shot_only")
        is not True
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint-residual dynamic tournament or split isolation changed."
        )

    candidate_report_sha256 = selection.get("candidate_report_sha256")
    if (
        not isinstance(candidate_report_sha256, dict)
        or set(candidate_report_sha256) != set(candidate_ids)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint selection lacks its exact dynamic candidate receipts."
        )
    authenticated_evaluations: dict[str, dict[str, Any]] = {}
    selection_keys: dict[str, tuple[Any, ...]] = {}
    behavioral_keys: dict[str, tuple[float, ...]] = {}
    for candidate_id in candidate_ids:
        evaluation = _self_sealed(
            evaluations.get(candidate_id),
            seal_field="report_sha256",
            name=f"endpoint attention candidate {candidate_id!r}",
        )
        raw_key = evaluation.get("selection_key")
        if (
            evaluation.get("candidate_id") != candidate_id
            or evaluation.get("operator_compiled_from_splits") != ["fit"]
            or evaluation.get("mlp_compiled_from_splits") != ["fit"]
            or evaluation.get("candidate_evaluated_on_splits") != ["selection"]
            or evaluation.get("validation_phase_count_consumed") != 0
            or evaluation.get("training_steps") != 0
            or evaluation.get("parameters_pruned") != 0
            or evaluation.get("parameters_quantized") != 0
            or candidate_report_sha256.get(candidate_id)
            != evaluation.get("report_sha256")
            or not isinstance(raw_key, list)
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
                f"Endpoint candidate {candidate_id!r} crossed a split or "
                "fold-only gate."
            )
        authenticated_evaluations[candidate_id] = evaluation
        selection_keys[candidate_id] = tuple(raw_key)
        behavioral_keys[candidate_id] = tuple(float(value) for value in raw_key[:-1])
    expected_selected_id = min(
        candidate_ids,
        key=lambda candidate_id: selection_keys[candidate_id],
    )
    if (
        selected_id != expected_selected_id
        or selection.get("selected_candidate_report_sha256")
        != candidate_report_sha256[selected_id]
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint dynamic-inventory winner does not reproduce from its "
            "ranking keys."
        )

    expected_composed_policy = _plain(_h776_composed_qk_attention_policy().as_dict())
    expected_endpoint_policy = _plain(
        _h776_endpoint_residual_attention_policy().as_dict()
    )
    composed_policy = policy.get("composed_qk_attention_policy")
    endpoint_policy = policy.get("endpoint_residual_attention_policy")
    if (
        composed_policy != expected_composed_policy
        or endpoint_policy != expected_endpoint_policy
        or composed_policy.get("authentication_sha256")
        != pins.expected_composed_qk_attention_policy_sha256
        or endpoint_policy.get("authentication_sha256")
        != pins.expected_endpoint_residual_attention_policy_sha256
        or endpoint_policy.get("allow_legacy_recomputed_parent_probabilities")
        is not False
        or endpoint_policy.get("training_allowed") is not False
        or endpoint_policy.get("endpoint_policy", {}).get("solver_device")
        != "cuda:1"
        or endpoint_policy.get("endpoint_policy", {}).get("compute_devices")
        != ["cuda:0", "cuda:1"]
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint or prerequisite composed policy authentication changed."
        )
    stage_one_attempt = _self_sealed(
        policy.get("composed_qk_stage_one_attempt"),
        seal_field="attempt_sha256",
        name="endpoint prerequisite stage-one attempt",
    )
    endpoint_qk_start = _self_sealed(
        policy.get("endpoint_qk_start_artifact"),
        seal_field="artifact_sha256",
        name="selected endpoint Q/K start",
    )
    expected_start_source = (
        "accepted_score_residual_corrected_dense_qkv"
        if composed_candidate_eligible
        else "structural_global_gauge_dense_qkv"
    )
    expected_start_schema = (
        LFM2_COMPOSED_QK_PRODUCER_SCHEMA
        if composed_candidate_eligible
        else LFM2_ENDPOINT_QK_START_SCHEMA
    )
    if (
        stage_one_attempt.get("schema")
        != LFM2_COMPOSED_QK_STAGE_ONE_ATTEMPT_SCHEMA
        or stage_one_attempt.get("composed_candidate_eligible")
        is not composed_candidate_eligible
        or stage_one_attempt.get("score_result_accepted")
        is not composed_candidate_eligible
        or stage_one_attempt.get("failed_composed_candidate_omitted")
        is composed_candidate_eligible
        or stage_one_attempt.get("endpoint_start_issued_independently") is not True
        or stage_one_attempt.get("endpoint_start_source") != expected_start_source
        or stage_one_attempt.get("endpoint_start_artifact_sha256")
        != endpoint_qk_start["artifact_sha256"]
        or endpoint_qk_start.get("schema") != expected_start_schema
        or policy.get("endpoint_qk_start_source") != expected_start_source
        or policy.get("composed_qk_candidate_eligible")
        is not composed_candidate_eligible
        or (
            policy.get("composed_qk_producer_artifact") is None
        )
        is composed_candidate_eligible
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint stage-one outcome relabeled a rejected composed arm or "
            "changed its selected dense QKV start."
        )

    endpoint_evaluation = authenticated_evaluations[
        ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID
    ]
    endpoint = _self_sealed(
        policy.get("endpoint_residual_attention_artifact"),
        seal_field="artifact_sha256",
        name="endpoint-residual operator artifact",
    )
    endpoint_from_evaluation = _self_sealed(
        endpoint_evaluation.get("operator_artifact"),
        seal_field="artifact_sha256",
        name="endpoint-residual evaluated operator artifact",
    )
    source_inventory = _self_sealed(
        endpoint.get("source_inventory"),
        seal_field="inventory_sha256",
        name="actual source dense Q/K/V/O inventory",
    )
    tensor_receipts = source_inventory.get("tensor_receipts")
    if not isinstance(tensor_receipts, dict) or set(tensor_receipts) != {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
    }:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint source inventory is not actual dense Q/K/V/O."
        )
    source_logical_parameters = 0
    for name, tensor_receipt in tensor_receipts.items():
        if (
            not isinstance(tensor_receipt, dict)
            or not isinstance(tensor_receipt.get("shape"), list)
            or len(tensor_receipt["shape"]) != 2
            or any(type(value) is not int or value <= 0 for value in tensor_receipt["shape"])
            or tensor_receipt.get("dtype") != "torch.bfloat16"
            or type(tensor_receipt.get("logical_parameters")) is not int
            or tensor_receipt["logical_parameters"]
            != math.prod(tensor_receipt["shape"])
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                f"Endpoint source tensor receipt {name!r} is invalid."
            )
        _digest(tensor_receipt.get("tensor_sha256"), f"{name} tensor_sha256")
        _digest(
            tensor_receipt.get("structural_tensor_sha256"),
            f"{name} structural_tensor_sha256",
        )
        source_logical_parameters += tensor_receipt["logical_parameters"]
    endpoint_operation = endpoint.get("operation_audit")
    if (
        endpoint.get("schema") != LFM2_COMPOSED_ENDPOINT_RESIDUAL_SCHEMA
        or endpoint.get("candidate_id")
        != H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_ID
        or endpoint.get("accepted") is not True
        or endpoint.get("policy") != expected_endpoint_policy
        or endpoint.get("policy_sha256")
        != pins.expected_endpoint_residual_attention_policy_sha256
        or endpoint_from_evaluation != endpoint
        or endpoint_evaluation.get("operator_artifact_sha256") != _sha256(endpoint)
        or policy.get("endpoint_residual_attention_artifact_sha256")
        != endpoint["artifact_sha256"]
        or source_inventory.get("count_derived_from_actual_dense_tensors")
        is not True
        or source_inventory.get("logical_parameters") != source_logical_parameters
        or endpoint.get("source_inventory_sha256")
        != source_inventory["inventory_sha256"]
        or endpoint.get("source_structural_state_sha256")
        != source_inventory.get("structural_source_state_sha256")
        or endpoint.get("source_logical_parameters") != source_logical_parameters
        or type(endpoint.get("target_logical_parameters")) is not int
        or endpoint["target_logical_parameters"] <= 0
        or endpoint["target_logical_parameters"] >= source_logical_parameters
        or not math.isclose(
            float(endpoint.get("parameter_compaction_ratio", math.nan)),
            source_logical_parameters / endpoint["target_logical_parameters"],
            rel_tol=1e-15,
            abs_tol=0.0,
        )
        or endpoint.get("ordinary_dense_target_qkvo") is not True
        or endpoint.get("fit_check_endpoint_and_route_nonregressing") is not True
        or endpoint.get(
            "fit_check_first_metric_evaluation_after_all_outer_blocks"
        )
        is not True
        or endpoint.get("fit_check_triggered_refit") is not False
        or endpoint.get("fit_check_role")
        != "reused_development_acceptance_gate_not_unbiased_validation"
        or endpoint.get("unbiased_generalization_claimed") is not False
        or endpoint.get("fresh_whole_child_intelligence_validation_required")
        is not True
        or endpoint.get("proof_scope")
        != "source_bound_local_attention_operator_fold_not_whole_child_intelligence"
        or endpoint.get("direct_parent_probability_barycenter_target") is not True
        or endpoint.get("direct_parent_attention_endpoint_target") is not True
        or endpoint.get("parent_probability_evidence_modes")
        != ["authenticated_cached_parent_attention_probabilities"]
        or endpoint.get("authenticated_cached_parent_probability_case_count")
        != H776_COMPOSED_UPSTREAM_FIT_CASES
        or endpoint.get("legacy_recomputed_parent_probability_case_count") != 0
        or endpoint.get("all_parent_probability_targets_from_authenticated_cache")
        is not True
        or endpoint.get("legacy_parent_probability_recomputation_allowed")
        is not False
        or not isinstance(endpoint_operation, dict)
        or any(
            type(value) is not int or value != 0
            for value in endpoint_operation.values()
        )
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint artifact is not source-bound, cache-bound, compact, or "
            "fold-only."
        )

    composed_producer = None
    if composed_candidate_eligible:
        composed_producer = _self_sealed(
            policy.get("composed_qk_producer_artifact"),
            seal_field="artifact_sha256",
            name="endpoint prerequisite composed producer",
        )
        if composed_producer != endpoint_qk_start:
            raise Lfm2CpcfV5RealAttentionProbeError(
                "Accepted composed producer differs from the selected endpoint "
                "start."
            )
    elif policy.get("composed_qk_producer_artifact") is not None:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Rejected composed candidate was retained in the tournament."
        )
    stage_one_producer = endpoint_qk_start
    global_sync = _self_sealed(
        policy.get("composed_qk_global_sync"),
        seal_field="report_sha256",
        name="endpoint prerequisite global synchronization",
    )
    score = _self_sealed(
        policy.get("composed_qk_score_result"),
        seal_field="report_sha256",
        name="endpoint prerequisite score residual",
    )
    stage_one_operation = stage_one_attempt.get("operation_audit")
    start_operation = endpoint_qk_start.get("operation_audit")
    if composed_candidate_eligible:
        start_weight_receipt_names = (
            "corrected_query_weight_sha256",
            "corrected_key_weight_sha256",
            "corrected_value_weight_sha256",
        )
        start_specific_lineage_is_valid = (
            endpoint_qk_start.get("candidate_id")
            == H776_COMPOSED_ATTENTION_CANDIDATE_ID
            and endpoint_qk_start.get("corrected_query_weight_sha256")
            == score.get("query_weight_sha256")
            and endpoint_qk_start.get("corrected_key_weight_sha256")
            == score.get("key_weight_sha256")
            and isinstance(
                endpoint_qk_start.get("corrected_value_weight_sha256"),
                str,
            )
        )
    else:
        start_weight_receipt_names = (
            "starting_producer_sha256",
            "starting_query_weight_sha256",
            "starting_key_weight_sha256",
            "starting_value_weight_sha256",
        )
        start_specific_lineage_is_valid = (
            endpoint_qk_start.get("starting_query_weight_sha256")
            == global_sync.get("query_weight_sha256")
            and endpoint_qk_start.get("starting_key_weight_sha256")
            == global_sync.get("key_weight_sha256")
            and isinstance(
                endpoint_qk_start.get("starting_value_weight_sha256"),
                str,
            )
            and isinstance(endpoint_qk_start.get("starting_producer_sha256"), str)
            and endpoint_qk_start.get("score_result_accepted") is False
            and endpoint_qk_start.get("score_residual_fourth_arm_eligible")
            is False
            and endpoint_qk_start.get("starting_state_source")
            == "fit_structural_groups_plus_global_gauge_qk_plus_fit_only_v"
            and stage_one_attempt.get(
                "global_gauge_fallback_start_artifact_sha256"
            )
            == endpoint_qk_start["artifact_sha256"]
        )
    for receipt_name in start_weight_receipt_names:
        _digest(
            endpoint_qk_start.get(receipt_name),
            f"endpoint Q/K start {receipt_name}",
        )
    qk_fit_ordinals = stage_one_producer.get("qk_fit_source_ordinals")
    qk_fit_check_ordinals = stage_one_producer.get("qk_fit_check_source_ordinals")
    endpoint_start_link_is_valid = (
        endpoint.get("composed_qk_artifact_sha256")
        == stage_one_producer["artifact_sha256"]
        if composed_candidate_eligible
        else (
            endpoint.get("composed_qk_artifact_sha256") is None
            and endpoint.get("endpoint_qk_start_artifact_sha256")
            == stage_one_producer["artifact_sha256"]
            and endpoint.get("endpoint_qk_start_artifact_schema")
            == LFM2_ENDPOINT_QK_START_SCHEMA
            and endpoint.get("endpoint_qk_start_source")
            == "structural_global_gauge_dense_qkv"
            and endpoint.get("composed_score_residual_candidate_eligible") is False
            and endpoint.get("composed_score_residual_result_sha256")
            == score["report_sha256"]
        )
    )
    if (
        stage_one_producer.get("schema") != expected_start_schema
        or stage_one_producer.get("policy") != expected_composed_policy
        or stage_one_producer.get("policy_sha256")
        != expected_composed_policy["authentication_sha256"]
        or stage_one_producer.get("source_split_sha256") != fit_bank_sha
        or stage_one_producer.get("upstream_fit_case_count")
        != H776_COMPOSED_UPSTREAM_FIT_CASES
        or stage_one_producer.get("upstream_trajectory_sha256s")
        != list(full_fit_trajectories)
        or not isinstance(qk_fit_ordinals, list)
        or not isinstance(qk_fit_check_ordinals, list)
        or len(qk_fit_ordinals) != H776_COMPOSED_QK_FIT_CASES
        or len(qk_fit_check_ordinals) != H776_COMPOSED_QK_FIT_CHECK_CASES
        or any(type(value) is not int for value in qk_fit_ordinals)
        or any(type(value) is not int for value in qk_fit_check_ordinals)
        or set(qk_fit_ordinals) & set(qk_fit_check_ordinals)
        or set(qk_fit_ordinals) | set(qk_fit_check_ordinals)
        != set(range(H776_COMPOSED_UPSTREAM_FIT_CASES))
        or not endpoint_start_link_is_valid
        or not start_specific_lineage_is_valid
        or stage_one_producer.get("global_sync_report_sha256")
        != global_sync["report_sha256"]
        or score.get("accepted") is not composed_candidate_eligible
        or (
            not composed_candidate_eligible
            and stage_one_producer.get("score_result_accepted") is not False
        )
        or stage_one_producer.get("score_result_report_sha256")
        != score["report_sha256"]
        or stage_one_attempt.get("score_result_report_sha256")
        != score["report_sha256"]
        or stage_one_attempt.get("score_result_status") != score.get("status")
        or stage_one_attempt.get("composed_candidate_artifact_sha256")
        != (
            stage_one_producer["artifact_sha256"]
            if composed_candidate_eligible
            else None
        )
        or score.get("policy") != expected_composed_policy["score_policy"]
        or global_sync.get("policy") != expected_composed_policy["global_policy"]
        or not isinstance(stage_one_operation, Mapping)
        or any(
            type(value) is not int or value != 0
            for value in stage_one_operation.values()
        )
        or not isinstance(start_operation, Mapping)
        or any(
            type(value) is not int or value != 0
            for value in start_operation.values()
        )
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint artifact lost its exact admitted-or-fallback Q/K start "
            "lineage."
        )
    fit_trajectories = tuple(full_fit_trajectories[index] for index in qk_fit_ordinals)
    fit_check_trajectories = tuple(
        full_fit_trajectories[index] for index in qk_fit_check_ordinals
    )
    if (
        endpoint.get("fit_trajectory_sha256s") != list(fit_trajectories)
        or endpoint.get("fit_check_trajectory_sha256s")
        != list(fit_check_trajectories)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint final evaluation does not use the exact composed "
            "Q/K FIT and FIT-CHECK partitions."
        )
    score_outcome_audit = _authenticate_h776_composed_score_outcome(
        score,
        policy=expected_composed_policy["score_policy"],
        composed_candidate_eligible=composed_candidate_eligible,
    )
    composed_device_execution = _authenticate_h776_composed_device_execution(
        score,
        producer=stage_one_producer,
        global_sync=global_sync,
        fit_trajectory_sha256s=full_fit_trajectories,
        pins=pins,
    )

    block_reports = policy.get("endpoint_residual_block_reports")
    proposal_reports = policy.get("endpoint_residual_qk_fit_proposal_reports")
    outer_block_count = endpoint.get("outer_block_count")
    output_prior = policy.get("endpoint_residual_output_prior_receipt")
    if (
        type(outer_block_count) is not int
        or outer_block_count not in {1, 2}
        or not isinstance(block_reports, list)
        or len(block_reports) != outer_block_count
        or not isinstance(proposal_reports, list)
        or len(proposal_reports) != outer_block_count
        or not isinstance(output_prior, dict)
        or output_prior.get("construction")
        != "zero_from_target_shape_and_dtype_metadata_only"
        or output_prior.get("blank_target_values_read") is not False
        or policy.get("endpoint_residual_blank_target_values_read") is not False
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint blocks or blank-independent output prior are incomplete."
        )
    _digest(output_prior.get("tensor_sha256"), "endpoint output prior tensor")
    block_shas = []
    proposal_shas = []
    device_execution_audits = []
    for block_index, (block, proposal) in enumerate(
        zip(block_reports, proposal_reports, strict=True)
    ):
        authenticated_block = _self_sealed(
            block,
            seal_field="block_sha256",
            name=f"endpoint outer block {block_index}",
        )
        authenticated_proposal = _self_sealed(
            proposal,
            seal_field="report_sha256",
            name=f"endpoint Q/K FIT proposal {block_index}",
        )
        block_operation = authenticated_block.get("operation_audit")
        proposal_operation = authenticated_proposal.get("operation_audit")
        sweeps = authenticated_proposal.get("sweeps")
        if (
            authenticated_block.get("schema")
            != LFM2_COMPOSED_ENDPOINT_RESIDUAL_BLOCK_SCHEMA
            or authenticated_block.get("block_index") != block_index
            or authenticated_block.get("fit_trajectory_sha256s")
            != list(fit_trajectories)
            or authenticated_block.get("qk_fit_proposal_sha256")
            != authenticated_proposal["report_sha256"]
            or authenticated_block.get("fit_check_accessible") is not False
            or authenticated_block.get("fit_check_cases_consumed") != 0
            or authenticated_proposal.get("policy")
            != expected_endpoint_policy["endpoint_policy"]
            or authenticated_proposal.get("fit_check_accessible") is not False
            or authenticated_proposal.get("fit_check_cases_consumed") != 0
            or authenticated_proposal.get("changed") is not True
            or not isinstance(sweeps, list)
            or not sweeps
            or len(sweeps)
            > expected_endpoint_policy["endpoint_policy"]["maximum_sweeps"]
            or not isinstance(block_operation, dict)
            or any(
                type(value) is not int or value != 0
                for value in block_operation.values()
            )
            or not isinstance(proposal_operation, dict)
            or proposal_operation.get("fit_cases_consumed")
            != H776_COMPOSED_QK_FIT_CASES
            or proposal_operation.get("fit_check_cases_consumed") != 0
            or any(
                type(proposal_operation.get(name)) is not int
                or proposal_operation.get(name) != 0
                for name in (
                    "model_forward_calls",
                    "model_generate_calls",
                    "backward_calls",
                    "optimizer_steps",
                    "training_steps",
                    "parameters_pruned",
                    "parameters_quantized",
                    "blank_target_value_reads",
                )
            )
        ):
            raise Lfm2CpcfV5RealAttentionProbeError(
                f"Endpoint outer block {block_index} is not FIT-only or "
                "authenticated."
            )
        positive_frontier_step_observed = False
        for sweep_index, sweep in enumerate(sweeps):
            solve = sweep.get("solve_report") if isinstance(sweep, dict) else None
            if (
                not isinstance(sweep, dict)
                or sweep.get("sweep") != sweep_index
                or not isinstance(solve, dict)
                or set(solve)
                != {
                    "converged",
                    "usable_direction",
                    "usable_direction_rule",
                    "direction_is_finite",
                    "direction_has_nonzero_entry",
                    "iterations",
                    "relative_normal_residual",
                    "termination",
                    "normal_operator_calls",
                }
                or solve.get("usable_direction_rule")
                != (
                    "finite_nonzero_solution_and_relative_residual_or_"
                    "maximum_iterations"
                )
                or any(
                    type(solve.get(name)) is not bool
                    for name in (
                        "converged",
                        "usable_direction",
                        "direction_is_finite",
                        "direction_has_nonzero_entry",
                    )
                )
                or type(solve.get("iterations")) is not int
                or not 0
                <= solve["iterations"]
                <= expected_endpoint_policy["endpoint_policy"][
                    "maximum_cg_iterations"
                ]
                or type(solve.get("normal_operator_calls")) is not int
                or solve["normal_operator_calls"] < 0
            ):
                raise Lfm2CpcfV5RealAttentionProbeError(
                    f"Endpoint block {block_index} sweep {sweep_index} has "
                    "invalid usable-direction evidence."
                )
            expected_usable = bool(
                solve["direction_is_finite"]
                and solve["direction_has_nonzero_entry"]
                and solve.get("termination")
                in {"relative_normal_residual", "maximum_iterations"}
            )
            if (
                solve["usable_direction"] is not expected_usable
                or (
                    solve.get("termination") == "maximum_iterations"
                    and (
                        solve["converged"] is not False
                        or solve["iterations"]
                        != expected_endpoint_policy["endpoint_policy"][
                            "maximum_cg_iterations"
                        ]
                    )
                )
                or (
                    solve.get("termination") == "relative_normal_residual"
                    and solve["converged"] is not True
                )
            ):
                raise Lfm2CpcfV5RealAttentionProbeError(
                    f"Endpoint block {block_index} sweep {sweep_index} "
                    "misstates its usable truncated-CG direction."
                )
            selected_alpha = sweep.get("selected_alpha")
            if (
                expected_usable
                and isinstance(selected_alpha, (int, float))
                and not isinstance(selected_alpha, bool)
                and math.isfinite(float(selected_alpha))
                and float(selected_alpha) > 0.0
            ):
                positive_frontier_step_observed = True
        if not positive_frontier_step_observed:
            raise Lfm2CpcfV5RealAttentionProbeError(
                f"Endpoint block {block_index} changed state without one "
                "authenticated positive exact-BF16 frontier step."
            )
        block_shas.append(authenticated_block["block_sha256"])
        proposal_shas.append(authenticated_proposal["report_sha256"])
        device_execution_audits.append(
            _plain(
                _authenticate_h776_endpoint_device_execution(
                    authenticated_proposal,
                    expected_fit_trajectory_sha256s=fit_trajectories,
                    pins=pins,
                )
            )
        )
    if (
        block_shas != endpoint.get("block_sha256s")
        or block_shas
        != [block["block_sha256"] for block in block_reports]
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint outer-block lineage changed."
        )

    baseline_fit = endpoint.get("baseline_fit_metrics")
    final_fit = endpoint.get("final_fit_metrics")
    baseline_check = endpoint.get("baseline_fit_check_metrics")
    final_check = endpoint.get("final_fit_check_metrics")
    if (
        not all(
            isinstance(value, dict)
            for value in (baseline_fit, final_fit, baseline_check, final_check)
        )
        or final_fit.get("endpoint_relative_rmse", math.inf)
        >= baseline_fit.get("endpoint_relative_rmse", -math.inf)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint final state did not strictly improve FIT endpoint error."
        )

    selected_type_by_id = {
        DIRECT_MONOLITHIC_ATTENTION_CANDIDATE_ID: (
            Lfm2CompiledStructuralAttentionFold
        ),
        RESIDUAL_ATTENTION_SPLIT_CANDIDATE_ID: (
            Lfm2CompiledCpcfResidualAttentionSplitFold
        ),
        H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID: (
            Lfm2CompiledFixedQkSequenceEffectiveAttentionFold
        ),
        H776_COMPOSED_ATTENTION_CANDIDATE_ID: (
            Lfm2CompiledComposedQkSequenceEffectiveAttentionFold
        ),
        H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_ID: (
            Lfm2CompiledComposedEndpointResidualAttentionFold
        ),
    }
    selected_artifact = getattr(result, "operator_artifact", None)
    selected_report = authenticated_evaluations[selected_id].get("operator_artifact")
    if (
        not isinstance(selected_artifact, selected_type_by_id[selected_id])
        or not callable(getattr(selected_artifact, "as_dict", None))
        or not callable(getattr(selected_artifact, "state_dict", None))
        or _plain(selected_artifact.as_dict()) != selected_report
        or report.get("operator_artifact") != selected_report
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint tournament winner is not the exact typed frozen artifact."
        )
    if isinstance(selected_artifact, Lfm2CompiledStructuralAttentionFold):
        state_sha = _structural_attention_state_sha256(
            selected_artifact.state_dict()
        )
        authenticated = state_sha == selected_artifact.compiled_state_sha256
    elif isinstance(selected_artifact, Lfm2CompiledCpcfResidualAttentionSplitFold):
        state_sha = _tensor_bundle_sha256(selected_artifact.state_dict())
        authenticated = state_sha == selected_artifact.compiled_state_sha256
    elif isinstance(
        selected_artifact,
        Lfm2CompiledFixedQkSequenceEffectiveAttentionFold,
    ):
        authenticated = selected_artifact.verify_authentication() is True
    else:
        authenticated = (
            selected_artifact.verify_authentication()
            == selected_artifact.artifact_sha256
        )
    if not authenticated:
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint tournament winner failed live typed authentication."
        )
    next_bank = getattr(result, "next_bank", None)
    if (
        next_bank is None
        or getattr(next_bank, "layer_index", None) != H776_LAYER_INDEX + 1
        or report.get("next_bank_sha256")
        != getattr(next_bank, "state_bank_sha256", None)
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Endpoint layer transaction did not create its in-memory k=3 bank."
        )
    endpoint_won = (
        selected_id == H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_ID
    )
    prior_candidate_ids = tuple(
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id != H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_ID
    )
    endpoint_strictly_better = all(
        behavioral_keys[H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_ID]
        < behavioral_keys[candidate_id]
        for candidate_id in prior_candidate_ids
    )
    body = {
        "schema": LFM2_CPCF_V5_REAL_ENDPOINT_RESIDUAL_ATTENTION_AUDIT_SCHEMA,
        "layer_result_sha256": report["result_sha256"],
        "input_bank_sha256": input_bank_sha,
        "fit_bank_sha256": fit_bank_sha,
        "fit_trajectory_sha256s": list(full_fit_trajectories),
        "next_bank_sha256": report["next_bank_sha256"],
        "candidate_ids": list(candidate_ids),
        "selected_candidate_id": selected_id,
        "selection_keys": {
            candidate_id: list(selection_keys[candidate_id])
            for candidate_id in candidate_ids
        },
        "endpoint_residual_won": endpoint_won,
        "endpoint_residual_strictly_better_than_all_prior_arms": (
            endpoint_strictly_better
        ),
        "scientific_outcome": (
            "positive_endpoint_residual_selection"
            if endpoint_won and endpoint_strictly_better
            else "honest_negative_or_prior_arm_selected"
        ),
        "full_validation_metrics": _plain(report["split_metrics"]["validation"]),
        "endpoint_artifact": endpoint,
        "source_dense_qkvo_inventory": source_inventory,
        "endpoint_residual_attention_policy_sha256": (
            pins.expected_endpoint_residual_attention_policy_sha256
        ),
        "composed_prerequisite_device_execution": _plain(
            composed_device_execution
        ),
        "composed_score_outcome": _plain(score_outcome_audit),
        "endpoint_block_device_execution": device_execution_audits,
        "all_parent_probability_targets_from_authenticated_cache": True,
        "blank_target_values_read": False,
        "candidate_local_fit_only_mlp_count": len(candidate_ids),
        "composed_score_residual_candidate_eligible": (
            composed_candidate_eligible
        ),
        "endpoint_qk_start_source": expected_start_source,
        "fit_check_final_one_shot_only": True,
        "continuous_cuda_allocator_peak_monitor_is_authority": True,
        "selected_live_operator_artifact_bound": True,
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
        raise TypeError("post_operator_chart_transport_shadow must be bool.")
    if type(runtime.attention_sequential_transport_enabled) is not bool:
        raise TypeError("attention_sequential_transport_enabled must be bool.")
    if type(runtime.composed_qk_attention_enabled) is not bool:
        raise TypeError("composed_qk_attention_enabled must be bool.")
    if type(runtime.endpoint_residual_attention_enabled) is not bool:
        raise TypeError("endpoint_residual_attention_enabled must be bool.")
    if (
        sum(
            (
                runtime.attention_sequential_transport_enabled,
                runtime.composed_qk_attention_enabled,
                runtime.endpoint_residual_attention_enabled,
            )
        )
        > 1
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Sequential transport, four-arm composed Q/K, and five-arm "
            "endpoint-residual attention cannot share one diagnostic run."
        )
    if (
        runtime.attention_sequential_transport_enabled
        != pins.expected_attention_sequential_transport_enabled
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Runtime attention sequential-transport mode differs from its "
            "independent invocation pin."
        )
    if (
        runtime.composed_qk_attention_enabled
        != pins.expected_composed_qk_attention_enabled
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Runtime composed Q/K mode differs from its independent invocation pin."
        )
    if (
        runtime.endpoint_residual_attention_enabled
        != pins.expected_endpoint_residual_attention_enabled
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Runtime endpoint-residual mode differs from its independent "
            "invocation pin."
        )
    if (
        resource_monitor.maximum_process_rss_bytes != pins.maximum_process_rss_bytes
        or resource_monitor.maximum_cuda_bytes_per_device
        != pins.maximum_cuda_bytes_per_device
    ):
        raise Lfm2CpcfV5RealAttentionProbeError(
            "Resource monitor caps differ from the independently pinned lane."
        )

    from . import hf_lfm2_cpcf_model_fold as model_fold
    from . import hf_lfm2_cpcf_v5_model_compiler as v5
    from . import hf_lfm2_sequence_effective_attention_fold as sequence_fold
    from .hf_lfm2_cpcf_layer_compiler import (
        Lfm2CpcfAttentionSequentialTransportPolicy,
        _attention_trajectory_sha256s,
        compile_and_install_lfm2_cpcf_layer,
    )
    from .hf_lfm2_cpcf_model_compiler import (
        rebind_lfm2_cpcf_phase_state_bank,
        select_lfm2_cpcf_phase_state_bank,
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
    target_inventory = model_fold.independent_parameter_inventory(runtime.target_model)
    if (
        runtime.target_plan_fingerprint != pins.expected_target_plan_fingerprint
        or blank_report["report_sha256"] != pins.expected_blank_report_sha256
        or target_before.state_fingerprint != pins.expected_blank_state_sha256
        or source_before.state_fingerprint != pins.expected_source_state_sha256
        or int(source_parameter_inventory["unique_parameters"])
        != H776_PARENT_UNIQUE_PARAMETERS
        or int(target_inventory["unique_parameters"]) != H776_TARGET_UNIQUE_PARAMETERS
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
                    enabled=(runtime.post_operator_chart_transport_shadow),
                )
            )
            if chart_transport_controller is not None:
                charts = chart_transport_controller.boundary_charts
                resource_monitor.checkpoint(
                    "layer_2_post_operator_chart_transport_frozen"
                )
            resource_monitor.checkpoint("layer_2_evidence_and_charts_ready")
            composed_fit_bank = (
                select_lfm2_cpcf_phase_state_bank(
                    layer_bank,
                    splits=("fit",),
                )
                if (
                    runtime.composed_qk_attention_enabled
                    or runtime.endpoint_residual_attention_enabled
                )
                else None
            )
            composed_fit_trajectory_sha256s = (
                _attention_trajectory_sha256s(
                    composed_fit_bank,
                    required_split="fit",
                )
                if composed_fit_bank is not None
                else ()
            )
            composed_qk_policy = (
                _h776_composed_qk_attention_policy()
                if (
                    runtime.composed_qk_attention_enabled
                    or runtime.endpoint_residual_attention_enabled
                )
                else None
            )
            endpoint_residual_policy = (
                _h776_endpoint_residual_attention_policy()
                if runtime.endpoint_residual_attention_enabled
                else None
            )

            def compile_layer() -> Any:
                """Execute the one authenticated layer transaction exactly once."""

                return compile_and_install_lfm2_cpcf_layer(
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
                    attention_sequential_transport_policy=(
                        Lfm2CpcfAttentionSequentialTransportPolicy(enabled=True)
                        if runtime.attention_sequential_transport_enabled
                        else None
                    ),
                    composed_qk_attention_policy=composed_qk_policy,
                    endpoint_residual_attention_policy=(
                        endpoint_residual_policy
                    ),
                )

            if (
                runtime.composed_qk_attention_enabled
                or runtime.endpoint_residual_attention_enabled
            ):
                # The composed lane contains both the legacy sequence arm and a
                # second seeded sequence arm; the endpoint lane adds a fifth
                # artifact.  The old single-arm PCG observer intentionally
                # cannot authenticate either expanded inventory.
                result = compile_layer()
            else:
                collector = Lfm2SequenceEffectivePcgCollector()
                with sequence_fold.observe_lfm2_sequence_effective_pcg(collector):
                    result = compile_layer()
            resource_monitor.checkpoint("layer_2_compiled_and_validated")
            if runtime.endpoint_residual_attention_enabled:
                result_audit = (
                    authenticate_h776_endpoint_residual_attention_result(
                        result,
                        input_bank_sha256=layer_bank.state_bank_sha256,
                        expected_fit_bank_sha256=(
                            composed_fit_bank.state_bank_sha256
                        ),
                        expected_fit_trajectory_sha256s=(
                            composed_fit_trajectory_sha256s
                        ),
                        pins=pins,
                    )
                )
            elif runtime.composed_qk_attention_enabled:
                result_audit = authenticate_h776_composed_attention_result(
                    result,
                    input_bank_sha256=layer_bank.state_bank_sha256,
                    expected_fit_bank_sha256=(composed_fit_bank.state_bank_sha256),
                    expected_fit_trajectory_sha256s=(composed_fit_trajectory_sha256s),
                    pins=pins,
                )
            else:
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
                    expected_attention_sequential_transport_enabled=(
                        pins.expected_attention_sequential_transport_enabled
                    ),
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
        "compiler_input_layer_charts_sha256": (result.report["boundary_charts_sha256"]),
        "controller_report_sha256": (
            None if controller_report is None else controller_report["report_sha256"]
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
    sequential_transport_body = {
        "enabled": runtime.attention_sequential_transport_enabled,
        "mode": (
            "candidate_local_fit_only_post_layer_transport"
            if runtime.attention_sequential_transport_enabled
            else "off"
        ),
        "runtime_matches_independent_invocation_pin": True,
        "layer_call_only_diagnostic": True,
        "authenticated_prefix_full_policy_sha256": policy_sha256,
        "authenticated_prefix_full_policy_unchanged": (
            policy_sha256 == pins.expected_full_policy_sha256
        ),
        "prefix_runtime_policy_mutated": False,
    }
    sequential_transport_binding = {
        **sequential_transport_body,
        "binding_sha256": _sha256(sequential_transport_body),
    }
    if runtime.endpoint_residual_attention_enabled:
        positive_attention_outcome = (
            result_audit["endpoint_residual_won"]
            and result_audit[
                "endpoint_residual_strictly_better_than_all_prior_arms"
            ]
        )
        completed_status = (
            "completed_positive_endpoint_residual_selection"
            if positive_attention_outcome
            else "completed_honest_negative_or_prior_arm_selected"
        )
    elif runtime.composed_qk_attention_enabled:
        positive_attention_outcome = (
            result_audit["composed_qk_won"]
            and result_audit["composed_qk_strictly_better_than_all_legacy_arms"]
        )
        completed_status = (
            "completed_positive_composed_qk_selection"
            if positive_attention_outcome
            else "completed_honest_negative_or_legacy_arm_selected"
        )
    else:
        positive_attention_outcome = (
            result_audit["sequence_effective_won"]
            and result_audit["sequence_effective_strictly_better_than_both_controls"]
        )
        completed_status = (
            "completed_positive_sequence_effective_selection"
            if positive_attention_outcome
            else "completed_honest_negative_or_control_selected"
        )
    resolved_attention_candidate_ids = (
        tuple(result_audit["candidate_ids"])
        if runtime.endpoint_residual_attention_enabled
        else (
            H776_COMPOSED_ATTENTION_CANDIDATE_IDS
            if runtime.composed_qk_attention_enabled
            else H776_ATTENTION_CANDIDATE_IDS
        )
    )
    body = {
        "schema": LFM2_CPCF_V5_REAL_ATTENTION_PROBE_SCHEMA,
        "status": completed_status,
        "pins": pins.as_dict(),
        "runtime_compiler_implementation": implementation,
        "runtime_sequence_source_sha256": sequence_source_sha256,
        "cross_runtime_source_inventory": {
            "checkpoint_origin_inventory_sha256": (
                pins.expected_source_inventory_sha256
            ),
            "live_process_inventory_sha256": (source_before.full_inventory_sha256),
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
            "parent_inventory_sha256": (source_parameter_inventory["inventory_sha256"]),
            "target_inventory_sha256": target_inventory["inventory_sha256"],
            "quantization_used": False,
            "pruning_used": False,
        },
        "checkpoint_audit": _plain(checkpoint_audit),
        "frontier_audit": _plain(frontier_audit),
        "phase_audit": _plain(phase_audit),
        "prefix_install_report": _plain(install_report),
        "layer_2_result": _plain(result_audit),
        "post_operator_chart_transport_shadow": (_plain(chart_transport_binding)),
        "attention_sequential_transport": _plain(sequential_transport_binding),
        "composed_qk_attention": {
            "enabled": runtime.composed_qk_attention_enabled,
            "enabled_as_endpoint_prerequisite": (
                runtime.endpoint_residual_attention_enabled
            ),
            "runtime_matches_independent_invocation_pin": True,
            "candidate_inventory": list(
                resolved_attention_candidate_ids
            ),
            "legacy_single_artifact_pcg_observer_used": (
                not (
                    runtime.composed_qk_attention_enabled
                    or runtime.endpoint_residual_attention_enabled
                )
            ),
            "positive_attention_outcome": positive_attention_outcome,
        },
        "endpoint_residual_attention": {
            "enabled": runtime.endpoint_residual_attention_enabled,
            "runtime_matches_independent_invocation_pin": True,
            "composed_qk_prerequisite_enabled_in_layer_call": (
                runtime.endpoint_residual_attention_enabled
            ),
            "candidate_inventory": list(
                resolved_attention_candidate_ids
            ),
            "candidate_local_fit_only_mlp_count": (
                len(resolved_attention_candidate_ids)
                if runtime.endpoint_residual_attention_enabled
                else None
            ),
            "fit_check_final_one_shot_only": (
                True if runtime.endpoint_residual_attention_enabled else None
            ),
            "blank_target_values_read": (
                False if runtime.endpoint_residual_attention_enabled else None
            ),
            "continuous_cuda_allocator_peak_monitor_authoritative": (
                True if runtime.endpoint_residual_attention_enabled else None
            ),
            "positive_attention_outcome": positive_attention_outcome,
        },
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
    "H776_COMPOSED_ATTENTION_CANDIDATE_ID",
    "H776_COMPOSED_ATTENTION_CANDIDATE_IDS",
    "H776_COMPOSED_CUDA_DEVICE_NAME",
    "H776_COMPOSED_LOGICAL_CUDA_UUIDS",
    "H776_COMPOSED_MAXIMUM_PROCESS_RSS_BYTES",
    "H776_COMPOSED_QK_ATTENTION_POLICY_SHA256",
    "H776_COMPOSED_QK_FIT_CASES",
    "H776_COMPOSED_QK_FIT_CHECK_CASES",
    "H776_COMPOSED_UPSTREAM_FIT_CASES",
    "H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_ID",
    "H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS",
    "H776_ENDPOINT_RESIDUAL_ATTENTION_POLICY_SHA256",
    "H776_LAYER_INDEX",
    "H776_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256",
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
    "LFM2_CPCF_V5_REAL_COMPOSED_ATTENTION_AUDIT_SCHEMA",
    "LFM2_CPCF_V5_REAL_ENDPOINT_RESIDUAL_ATTENTION_AUDIT_SCHEMA",
    "Lfm2CpcfV5RealAttentionProbeError",
    "Lfm2CpcfV5RealAttentionProbePins",
    "Lfm2CpcfV5RealAttentionProbeRuntime",
    "Lfm2CpcfV5RealAttentionResourceMonitor",
    "Lfm2SequenceEffectivePcgCollector",
    "authenticate_h776_attention_result",
    "authenticate_h776_composed_attention_result",
    "authenticate_h776_endpoint_residual_attention_result",
    "authenticate_h776_boundary_frontier",
    "authenticate_h776_k2_checkpoint",
    "authenticate_h776_phase_inventory",
    "execute_lfm2_cpcf_v5_real_attention_probe",
]
