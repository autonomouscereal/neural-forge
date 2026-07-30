from __future__ import annotations

import torch

from experiments import publish_lfm2_base_dual_route_bridge as publisher
from experiments import run_lfm2_carrier_reclosure as carrier_runner


def test_base_publisher_uses_the_existing_carrier_counted_abi() -> None:
    assert publisher.STANDALONE_BRIDGE_SCHEMA == carrier_runner.STANDALONE_BRIDGE_SCHEMA
    assert publisher.PUBLISHED_TENSOR_KEYS == carrier_runner.PUBLISHED_TENSOR_KEYS

    input_coefficients = torch.tensor([1.0, -0.25, 0.5], dtype=torch.float32)
    output_coefficients = torch.tensor([1.0, 0.75, -0.5], dtype=torch.float32)
    assert publisher._materialization_tensor_sha256(
        input_coefficients
    ) == carrier_runner._materialization_tensor_sha256(input_coefficients)
    assert publisher._materialization_tensor_sha256(
        output_coefficients
    ) == carrier_runner._materialization_tensor_sha256(output_coefficients)


def test_zero_operation_audit_excludes_training_quantization_and_pruning() -> None:
    assert publisher._ZERO_OPERATION_AUDIT == {
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameter_pruning_calls": 0,
        "parameter_quantization_calls": 0,
        "parameter_sparsification_calls": 0,
        "parameter_sharing_calls": 0,
        "runtime_sidecar_parameters": 0,
    }
