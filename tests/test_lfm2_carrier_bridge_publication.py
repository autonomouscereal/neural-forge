from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_artifact import _artifact_tree
from neural_forge.hf_lfm2_semantic_step_token_fold import (
    seal_lfm2_semantic_step_span_evidence,
)
from neural_forge.lfm2_carrier_bridge_publication import (
    PACKAGED_ROUTE_DIRECTORY,
    PACKAGED_ROUTE_TENSOR_FILENAME,
    STANDALONE_BRIDGE_SCHEMA,
    Lfm2CarrierBridgePublicationError,
    bind_packaged_lfm2_carrier_routes,
    carrier_materialization_tensor_sha256,
    package_lfm2_carrier_routes_with_child,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _UntiedChild(nn.Module):
    def __init__(self, rows: int = 2, hidden_size: int = 3) -> None:
        super().__init__()
        self.input_table = nn.Embedding(rows, hidden_size)
        self.output_table = nn.Linear(hidden_size, rows, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.input_table

    def get_output_embeddings(self) -> nn.Module:
        return self.output_table


def _csr(
    rows: list[tuple[list[int], list[float]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = [0]
    target_ids: list[int] = []
    coefficients: list[float] = []
    for ids, values in rows:
        target_ids.extend(ids)
        coefficients.extend(values)
        offsets.append(len(target_ids))
    return (
        torch.tensor(offsets, dtype=torch.int64),
        torch.tensor(target_ids, dtype=torch.int64),
        torch.tensor(coefficients, dtype=torch.float32),
    )


def _source_publication(root: Path) -> tuple[Path, str]:
    publication = root / "publication"
    publication.mkdir()
    input_code = _csr(
        [
            ([0], [1.0]),
            ([1], [1.0]),
            ([0, 1], [0.4, 0.6]),
        ]
    )
    output_code = _csr(
        [
            ([0], [1.0]),
            ([1], [1.0]),
            ([0, 1], [0.25, 0.75]),
        ]
    )
    tensors = {
        # Deliberately a different width from the child.  Packaging must treat
        # this table as authenticated source provenance, never as child state.
        "target_table": torch.arange(10, dtype=torch.float32).reshape(2, 5),
        "target_to_donor": torch.tensor([0, 1], dtype=torch.int64),
        "input_offsets": input_code[0],
        "input_target_ids": input_code[1],
        "input_coefficients": input_code[2],
        "output_offsets": output_code[0],
        "output_target_ids": output_code[1],
        "output_coefficients": output_code[2],
    }
    tensor_path = publication / "bridge.safetensors"
    save_file(tensors, str(tensor_path))
    coefficient_count = int(input_code[2].numel() + output_code[2].numel())
    materialization = {
        "input_coefficients_sha256": carrier_materialization_tensor_sha256(
            input_code[2]
        ),
        "output_coefficients_sha256": carrier_materialization_tensor_sha256(
            output_code[2]
        ),
        "continuous_fold_coefficient_count": coefficient_count,
    }
    operation_audit = {
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
    body = {
        "schema": STANDALONE_BRIDGE_SCHEMA,
        "status": "complete",
        "bridge_tensor_file": tensor_path.name,
        "bridge_tensor_file_sha256": _file_sha256(tensor_path),
        "runtime_requires_parent": False,
        "runtime_requires_external_sidecar": False,
        "all_continuous_bridge_state_is_in_bridge_tensor_file": True,
        "integer_csr_routes_are_structural_metadata": True,
        "continuous_semantic_coefficient_count": coefficient_count,
        "physical_target_table_parameter_count": int(tensors["target_table"].numel()),
        "target_table_dtype": str(tensors["target_table"].dtype),
        "operation_audit": operation_audit,
        "materialization_receipt": materialization,
    }
    receipt = {**body, "artifact_sha256": canonical_sha256(body)}
    (publication / "bridge-receipt.json").write_text(
        json.dumps(receipt, allow_nan=False, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return publication, str(receipt["artifact_sha256"])


def _package(
    tmp_path: Path,
) -> tuple[_UntiedChild, object, Path]:
    publication, artifact_sha256 = _source_publication(tmp_path)
    child_directory = tmp_path / "child"
    child_directory.mkdir()
    child = _UntiedChild()
    spans = seal_lfm2_semantic_step_span_evidence(
        [[0], [1], [0, 1]],
        target_row_count=2,
        lineage={"test_fixture": True},
    )
    packaged = package_lfm2_carrier_routes_with_child(
        publication,
        child_directory,
        child,
        expected_artifact_sha256=artifact_sha256,
        span_evidence=spans,
        expected_target_rows=2,
        expected_hidden_size=3,
    )
    return child, packaged, child_directory


@pytest.mark.parametrize("table_role", ["input", "output"])
def test_bind_rejects_live_child_table_mutation(
    tmp_path: Path,
    table_role: str,
) -> None:
    child, packaged, _child_directory = _package(tmp_path)
    table = (
        child.get_input_embeddings().weight
        if table_role == "input"
        else child.get_output_embeddings().weight
    )
    with torch.no_grad():
        table[0, 0].add_(1.0)

    with pytest.raises(
        Lfm2CarrierBridgePublicationError,
        match="Fresh child input/readout tables differ",
    ):
        bind_packaged_lfm2_carrier_routes(
            child,
            packaged,
            expected_target_rows=2,
            expected_hidden_size=3,
        )


@pytest.mark.parametrize("route_role", ["input", "output"])
def test_bind_rejects_mutated_loaded_route_coefficients(
    tmp_path: Path,
    route_role: str,
) -> None:
    child, packaged, _child_directory = _package(tmp_path)
    coefficients = packaged.tensors[f"{route_role}_coefficients"]
    # Index two belongs to the first non-anchor route, so the mutation remains
    # finite and structurally valid.  Integrity, not CSR validation, must catch it.
    coefficients[2].add_(0.125)

    with pytest.raises(
        Lfm2CarrierBridgePublicationError,
        match="changed between authentication and live binding",
    ):
        bind_packaged_lfm2_carrier_routes(
            child,
            packaged,
            expected_target_rows=2,
            expected_hidden_size=3,
        )


def test_outer_recursive_inventory_accepts_package_without_source_table(
    tmp_path: Path,
) -> None:
    _child, _packaged, child_directory = _package(tmp_path)
    package = child_directory / PACKAGED_ROUTE_DIRECTORY
    tensors = load_file(
        str(package / PACKAGED_ROUTE_TENSOR_FILENAME),
        device="cpu",
    )

    assert "target_table" not in tensors
    assert tuple(tensors["input_coefficients"].shape) == (4,)
    assert tuple(tensors["output_coefficients"].shape) == (4,)
    tree = _artifact_tree(child_directory)
    assert {row["path"] for row in tree} == {
        f"{PACKAGED_ROUTE_DIRECTORY}/carrier-routes-receipt.json",
        f"{PACKAGED_ROUTE_DIRECTORY}/{PACKAGED_ROUTE_TENSOR_FILENAME}",
    }
