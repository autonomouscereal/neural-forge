from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

import neural_forge.hf_lfm2_semantic_foundation_statistics as stats_module


PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")
BRIDGE_JSON = Path(
    "/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/"
    "v4096_modelcap_v2/neural_forge_strict_tokenizer_bridge.json"
)
FOUNDATION_RECEIPT = Path(
    "/home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_20260721/"
    "97ae4b9/a0003/foundation-receipt.json"
)
STATISTICS = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_semantic_foundation_statistics_20260721/02b5c6e/a0002/"
    "statistics-store/"
    "42083aed1e86665fa99a53e023dfbd613608ac60f86cd7c85c4bb1b92908cd6a"
)
EXPECTED_STATISTICS_SHA256 = (
    "42083aed1e86665fa99a53e023dfbd613608ac60f86cd7c85c4bb1b92908cd6a"
)
TENSOR_KEY = "model.embed_tokens.weight"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_table() -> torch.Tensor:
    bridge = json.loads(BRIDGE_JSON.read_text(encoding="utf-8"))
    mapping = bridge["transport_profile"]["target_to_donor_row_indices"]
    selected = None
    for shard in sorted(PARENT.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if TENSOR_KEY not in handle.keys():
                continue
            if selected is not None:
                raise RuntimeError("tied table occurs in more than one shard")
            selected = handle.get_tensor(TENSOR_KEY).detach().cpu().contiguous().clone()
    if selected is None:
        raise RuntimeError("parent tied table not found")
    rows = selected.index_select(0, torch.tensor(mapping, dtype=torch.long))
    return rows.detach().to(dtype=torch.bfloat16).contiguous().clone()


def differences(left: Any, right: Any, path: str = "", limit: int = 80) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def walk(a: Any, b: Any, current: str) -> None:
        if len(result) >= limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                child = f"{current}.{key}" if current else str(key)
                if key not in a:
                    result.append({"path": child, "left": "<missing>", "right": b[key]})
                elif key not in b:
                    result.append({"path": child, "left": a[key], "right": "<missing>"})
                else:
                    walk(a[key], b[key], child)
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                result.append({"path": current + ".length", "left": len(a), "right": len(b)})
            for index, (avalue, bvalue) in enumerate(zip(a, b, strict=False)):
                walk(avalue, bvalue, f"{current}[{index}]")
            return
        if a != b:
            result.append({"path": current, "left": a, "right": b})

    walk(left, right, path)
    return result


def main() -> None:
    receipt = json.loads(FOUNDATION_RECEIPT.read_text(encoding="utf-8"))
    receipt_file_sha256 = file_sha256(FOUNDATION_RECEIPT)
    artifact = stats_module._open_lfm2_semantic_foundation_interface_statistics(
        STATISTICS,
        expected_artifact_sha256=EXPECTED_STATISTICS_SHA256,
        verify_runtime=False,
        allow_legacy_implementation_for_frontier_migration=True,
    )
    bindings, policy, legacy_anchor = stats_module._foundation_bindings(
        receipt,
        foundation_receipt_file_sha256=receipt_file_sha256,
    )
    statistics = stats_module._reauthenticate_base_statistics(artifact)
    splits = stats_module._reauthenticate_artifact_splits(artifact)
    legacy_receipt = stats_module._verify_sealed(
        receipt,
        field="receipt_sha256",
        name="legacy foundation receipt",
    )
    legacy_solve = stats_module._verify_sealed(
        legacy_receipt.get("interface_solve"),
        field="result_sha256",
        name="legacy foundation interface solve",
    )
    legacy_base = stats_module._verify_sealed(
        legacy_solve.get("base_interface_report"),
        field="report_sha256",
        name="legacy base interface report",
    )
    legacy_active = stats_module._verify_sealed(
        legacy_solve.get("active_interface_report"),
        field="report_sha256",
        name="legacy active interface report",
    )
    table = load_source_table()
    base = stats_module.fit_tied_causal_interface_from_statistics(table, statistics)
    anchor = stats_module.fit_tied_causal_interface_anchor_residual(
        base,
        table,
        selection_split=splits["selection"],
        fit_split=splits["fit"],
        validation_split=splits["validation"],
        **{
            name: policy[name]
            for name in sorted(stats_module._ANCHOR_CALL_FIELDS)
        },
    )
    new_report = dict(anchor.report)
    legacy_projection = stats_module._legacy_anchor_report_from_frontier(new_report)
    observed = {
        "frontier_present": anchor.frontier is not None,
        "base_report_equal": dict(base.report) == legacy_base,
        "legacy_projection_equal": legacy_projection == legacy_anchor,
        "legacy_active_equal": legacy_active == legacy_anchor,
        "source_sha_equal": stats_module._foundation_tensor_sha256(table)
        == legacy_solve.get("source_tied_table_sha256"),
        "active_target_sha_equal": stats_module._foundation_tensor_sha256(
            anchor.interface.target_weight
        )
        == legacy_solve.get("active_target_weight_sha256"),
        "producer_sha_equal": stats_module._foundation_tensor_sha256(
            anchor.interface.producer_encoder
        )
        == legacy_solve.get("producer_encoder_sha256"),
    }
    report = {
        "runtime": {
            "torch": torch.__version__,
            "torch_file": torch.__file__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "artifact_bindings_equal": dict(artifact.bindings) == bindings,
        "artifact_policy_equal": dict(artifact.anchor_policy) == policy,
        "predicates": observed,
        "hashes": {
            "receipt_file_observed": receipt_file_sha256,
            "source_expected": legacy_solve.get("source_tied_table_sha256"),
            "source_observed": stats_module._foundation_tensor_sha256(table),
            "base_report_expected": legacy_base.get("report_sha256"),
            "base_report_observed": base.report.get("report_sha256"),
            "projection_expected": legacy_anchor.get("report_sha256"),
            "projection_observed": legacy_projection.get("report_sha256"),
            "active_target_expected": legacy_solve.get("active_target_weight_sha256"),
            "active_target_observed": stats_module._foundation_tensor_sha256(
                anchor.interface.target_weight
            ),
            "producer_expected": legacy_solve.get("producer_encoder_sha256"),
            "producer_observed": stats_module._foundation_tensor_sha256(
                anchor.interface.producer_encoder
            ),
        },
        "base_report_differences_observed_vs_expected": differences(
            dict(base.report), legacy_base
        ),
        "legacy_projection_differences_observed_vs_expected": differences(
            legacy_projection, legacy_anchor
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
