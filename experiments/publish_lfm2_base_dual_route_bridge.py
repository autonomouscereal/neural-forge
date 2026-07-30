"""Publish the proven unchanged-table A9 C_in + A11 C_out bridge.

This is an offline packaging step, not a solve.  It authenticates the raw A9
semantic codebook, the independent A11 output refit, and the original folded
target table, then emits the same standalone two-route ABI used by carrier
publications.  No carrier row is replanted and no route is reclosed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_functional_output_refit import (
    authenticate_lfm2_functional_output_refit_result,
)
from neural_forge.hf_lfm2_sparse_semantic_codebook import (
    load_lfm2_sparse_semantic_codebook,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_lfm2_carrier_reclosure import (  # noqa: E402
    STANDALONE_BRIDGE_SCHEMA,
    carrier_bridge_allocations_from_publication,
)


COMPLETION_SCHEMA = "neural-forge-lfm2-base-dual-route-publication-v1"
MATERIALIZATION_SCHEMA = "neural-forge-lfm2-base-dual-route-materialization-v1"
PUBLISHED_TENSOR_KEYS = (
    "target_table",
    "target_to_donor",
    "input_offsets",
    "input_target_ids",
    "input_coefficients",
    "output_offsets",
    "output_target_ids",
    "output_coefficients",
)
_MATERIALIZATION_TENSOR_HASH_DOMAIN = (
    b"neural-forge-lfm2-semantic-carrier-tensor-v1\0"
)
_ZERO_OPERATION_AUDIT = {
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


class BaseDualRoutePublicationError(RuntimeError):
    """The unchanged-table base bridge cannot be authenticated or published."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise BaseDualRoutePublicationError(f"{name} must be a SHA-256 digest.")
    return value.lower()


def _file(path: str | os.PathLike[str], name: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file() or value.is_symlink():
        raise BaseDualRoutePublicationError(f"{name} is not an ordinary file.")
    return value


def _directory(path: str | os.PathLike[str], name: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_dir() or value.is_symlink():
        raise BaseDualRoutePublicationError(f"{name} is not an ordinary directory.")
    return value


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaseDualRoutePublicationError(f"Cannot read {name}.") from exc
    if not isinstance(value, dict):
        raise BaseDualRoutePublicationError(f"{name} must be an object.")
    return value


def _load_source_table(
    path: Path, *, expected_file_sha256: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if _file_sha256(path) != _digest(expected_file_sha256, "target source file"):
        raise BaseDualRoutePublicationError("Target source file hash differs.")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(PUBLISHED_TENSOR_KEYS):
            raise BaseDualRoutePublicationError(
                "Target source must be the exact authenticated pre-replant bridge."
            )
        table = handle.get_tensor("target_table").contiguous()
        mapping = handle.get_tensor("target_to_donor").contiguous()
    if (
        table.ndim != 2
        or not table.is_floating_point()
        or table.is_complex()
        or mapping.dtype != torch.int64
        or mapping.ndim != 1
        or int(mapping.numel()) != int(table.shape[0])
    ):
        raise BaseDualRoutePublicationError("Target source geometry is invalid.")
    return table, mapping


def _materialization_tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach()
    digest = hashlib.sha256()
    digest.update(_MATERIALIZATION_TENSOR_HASH_DOMAIN)
    digest.update(str(tensor.dtype).removeprefix("torch.").encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        tensor.to(device="cpu")
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return digest.hexdigest()


def _load_a11_coefficients(
    path: Path, *, expected_file_sha256: str
) -> torch.Tensor:
    if _file_sha256(path) != _digest(expected_file_sha256, "A11 coefficient file"):
        raise BaseDualRoutePublicationError("A11 coefficient file hash differs.")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"output_semantic_code_coefficients"}:
            raise BaseDualRoutePublicationError("A11 tensor inventory differs.")
        return handle.get_tensor("output_semantic_code_coefficients").contiguous()


def _publish_directory(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise BaseDualRoutePublicationError("Output directory already exists.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(source, destination)


def publish(args: argparse.Namespace) -> dict[str, Any]:
    codebook = load_lfm2_sparse_semantic_codebook(
        _directory(args.a9_codebook_directory, "A9 codebook"),
        expected_artifact_sha256=_digest(args.expected_a9_artifact_sha256, "A9 pin"),
    )
    target_source = _file(args.target_source_safetensors, "target source")
    target_table, target_to_donor = _load_source_table(
        target_source,
        expected_file_sha256=args.expected_target_source_file_sha256,
    )
    if not torch.equal(target_to_donor, codebook.target_to_donor):
        raise BaseDualRoutePublicationError(
            "Unchanged target mapping differs from the authenticated A9 codebook."
        )
    if tuple(target_table.shape) != (
        int(codebook.receipt["target_row_count"]),
        int(codebook.receipt["hidden_size"]),
    ):
        raise BaseDualRoutePublicationError(
            "Unchanged target table geometry differs from A9."
        )

    a11_path = _file(args.a11_coefficients_safetensors, "A11 coefficients")
    a11_coefficients = _load_a11_coefficients(
        a11_path,
        expected_file_sha256=args.expected_a11_coefficients_file_sha256,
    )
    a11 = authenticate_lfm2_functional_output_refit_result(
        output_semantic_code_coefficients=a11_coefficients,
        receipt=_json(_file(args.a11_receipt_json, "A11 receipt"), "A11 receipt"),
        expected_cache_key=_digest(args.expected_a11_cache_key, "A11 cache key"),
    )
    if a11.result_sha256 != _digest(args.expected_a11_result_sha256, "A11 result"):
        raise BaseDualRoutePublicationError("A11 result differs from its pin.")
    if a11.receipt.get("input_codebook_artifact_sha256") != codebook.artifact_sha256:
        raise BaseDualRoutePublicationError("A11 was fit against a different A9 codebook.")
    if (
        a11_coefficients.dtype != torch.float32
        or tuple(a11_coefficients.shape)
        != tuple(codebook.semantic_code_coefficients.shape)
    ):
        raise BaseDualRoutePublicationError("A11 does not use the A9 route layout.")
    direct_positions = codebook.semantic_code_offsets.index_select(
        0, codebook.target_to_donor
    )
    if not torch.equal(
        a11_coefficients.index_select(0, direct_positions),
        codebook.semantic_code_coefficients.index_select(0, direct_positions),
    ):
        raise BaseDualRoutePublicationError("A11 changed an A9 direct anchor.")

    tensors = {
        "target_table": target_table,
        "target_to_donor": codebook.target_to_donor,
        "input_offsets": codebook.semantic_code_offsets,
        "input_target_ids": codebook.semantic_code_target_ids,
        "input_coefficients": codebook.semantic_code_coefficients,
        "output_offsets": codebook.semantic_code_offsets.clone(),
        "output_target_ids": codebook.semantic_code_target_ids.clone(),
        "output_coefficients": a11_coefficients,
    }
    input_sha = _materialization_tensor_sha256(tensors["input_coefficients"])
    output_sha = _materialization_tensor_sha256(tensors["output_coefficients"])
    target_sha = _materialization_tensor_sha256(tensors["target_table"])
    mapping_sha = _materialization_tensor_sha256(tensors["target_to_donor"])
    coefficient_count = int(tensors["input_coefficients"].numel()) + int(
        tensors["output_coefficients"].numel()
    )
    materialization = {
        "schema": MATERIALIZATION_SCHEMA,
        "status": "complete-unchanged-table-no-replant-no-reclosure",
        "target_table_sha256": target_sha,
        "target_to_donor_sha256": mapping_sha,
        "input_coefficients_sha256": input_sha,
        "output_coefficients_sha256": output_sha,
        "continuous_fold_coefficient_count": coefficient_count,
        "carrier_replant_count": 0,
        "route_reclosure_count": 0,
        "base_target_table_preserved_bit_exact": True,
        "raw_a9_input_route_preserved_bit_exact": True,
        "authenticated_a11_output_route_preserved_bit_exact": True,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    materialization = {
        **materialization,
        "artifact_sha256": canonical_sha256(materialization),
    }
    output = Path(args.output_directory).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        tensor_path = temporary / "bridge.safetensors"
        save_file(
            {name: tensor.detach().to(device="cpu").contiguous() for name, tensor in tensors.items()},
            str(tensor_path),
        )
        tensor_file_sha = _file_sha256(tensor_path)
        body = {
            "schema": STANDALONE_BRIDGE_SCHEMA,
            "status": "complete",
            "package_role": "proven-coherent-unchanged-table-a9-c-in-a11-c-out",
            "bridge_tensor_file": "bridge.safetensors",
            "bridge_tensor_file_sha256": tensor_file_sha,
            "source_a9_codebook_artifact_sha256": codebook.artifact_sha256,
            "source_a11_result_sha256": a11.result_sha256,
            "source_a11_coefficient_file_sha256": _file_sha256(a11_path),
            "source_target_table_file_sha256": _file_sha256(target_source),
            "materialization_artifact_sha256": materialization["artifact_sha256"],
            "materialization_receipt": materialization,
            "target_table_dtype": str(target_table.dtype),
            "target_table_dtype_preserved": True,
            "base_target_table_preserved_bit_exact": True,
            "input_and_output_routes_are_independent": True,
            "carrier_replant_count": 0,
            "route_reclosure_count": 0,
            "complete_reverse_incidence_closure_materialized": True,
            "reverse_incidence_closure_is_vacuous_without_replants": True,
            "runtime_requires_parent": False,
            "runtime_requires_external_sidecar": False,
            "all_continuous_bridge_state_is_in_bridge_tensor_file": True,
            "integer_csr_routes_are_structural_metadata": True,
            "continuous_semantic_coefficient_count": coefficient_count,
            "physical_target_table_parameter_count": int(target_table.numel()),
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }
        receipt = {**body, "artifact_sha256": canonical_sha256(body)}
        receipt_path = temporary / "bridge-receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _publish_directory(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    allocations = carrier_bridge_allocations_from_publication(
        output, expected_artifact_sha256=receipt["artifact_sha256"]
    )
    fresh_receipt = _json(output / "bridge-receipt.json", "fresh bridge receipt")
    fresh_body = dict(fresh_receipt)
    fresh_artifact = fresh_body.pop("artifact_sha256", None)
    if (
        fresh_artifact != canonical_sha256(fresh_body)
        or fresh_artifact != receipt["artifact_sha256"]
        or fresh_body.get("schema") != STANDALONE_BRIDGE_SCHEMA
        or _file_sha256(output / "bridge.safetensors")
        != fresh_body.get("bridge_tensor_file_sha256")
    ):
        raise BaseDualRoutePublicationError(
            "Fresh source bridge receipt or tensor binding differs."
        )
    completion_body = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete-fresh-reload-authenticated",
        "artifact_sha256": fresh_artifact,
        "bridge_tensor_file_sha256": fresh_receipt["bridge_tensor_file_sha256"],
        "bridge_receipt_file_sha256": _file_sha256(output / "bridge-receipt.json"),
        "input_route": allocations[0].as_dict(),
        "output_route": allocations[1].as_dict(),
        "input_coefficient_count": allocations[0].parameter_elements,
        "output_coefficient_count": allocations[1].parameter_elements,
        "continuous_semantic_coefficient_count": sum(
            allocation.parameter_elements for allocation in allocations
        ),
        "fresh_reload_authenticated": True,
        "base_target_table_preserved_bit_exact": True,
        "carrier_replant_count": 0,
        "route_reclosure_count": 0,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    completion = {
        **completion_body,
        "completion_sha256": canonical_sha256(completion_body),
    }
    completion_path = output / "complete.json"
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a9-codebook-directory", type=Path, required=True)
    parser.add_argument("--expected-a9-artifact-sha256", required=True)
    parser.add_argument("--a11-coefficients-safetensors", type=Path, required=True)
    parser.add_argument("--expected-a11-coefficients-file-sha256", required=True)
    parser.add_argument("--a11-receipt-json", type=Path, required=True)
    parser.add_argument("--expected-a11-cache-key", required=True)
    parser.add_argument("--expected-a11-result-sha256", required=True)
    parser.add_argument("--target-source-safetensors", type=Path, required=True)
    parser.add_argument("--expected-target-source-file-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main() -> None:
    result = publish(build_parser().parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
