#!/usr/bin/env python3
"""Real-artifact, pre-body hybrid-interface authentication preflight.

This intentionally enters the production semantic-postfold loader and stops at
the injected semantic layer-stack compiler boundary.  It does not execute any
of the fourteen layer operators/MLPs, the fused terminal, generation, training,
or checkpoint/capsule publication.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
for _path in (_REPOSITORY_ROOT, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from experiments.run_lfm2_residual_attention_fold import (  # noqa: E402
    load_local_tokenizer,
)
from neural_forge.hf_lfm2_semantic_postfold import (  # noqa: E402
    TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
    run_lfm2_semantic_postfold,
)
from neural_forge.hf_tied_residual_congruence import (  # noqa: E402
    authenticate_tied_residual_congruence_interface,
)


_CODE_COMMIT = "02cec5c82b6820a77e273af38b78cc0b0880285a"
_THREAD_SEQUENCE = (16, 1, 8, 16)

_INPUT_CHECKPOINT = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_semantic_foundation_20260721/97ae4b9/a0003/checkpoint"
)
_INPUT_ARTIFACT_SHA256 = (
    "517515e3525c107db180953dcde6c1f5dc2ab6af3bc0f50486e442d6bf69899f"
)
_TRACE_CACHE = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_complete_causal_evidence_20260720_a0003/parent/"
    "dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b"
)
_CACHE_KEY = "dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b"
_BRIDGE = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2"
)
_BRIDGE_SHA256 = (
    "5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384"
)
_DONOR_TOKENIZER = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_donor_tokenizer_packages_20260719/"
    "eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46"
)
_PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")
_FOUNDATION_RECEIPT = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_semantic_foundation_20260721/97ae4b9/a0003/foundation-receipt.json"
)
_FOUNDATION_RECEIPT_SHA256 = (
    "7bffddca0fc90e026a62d625cff9f14b436955dbc799f9d8cdb804dc54e07afd"
)
_STATISTICS = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_semantic_foundation_statistics_20260721/02b5c6e/a0002/"
    "statistics-store/"
    "42083aed1e86665fa99a53e023dfbd613608ac60f86cd7c85c4bb1b92908cd6a"
)
_STATISTICS_SHA256 = (
    "42083aed1e86665fa99a53e023dfbd613608ac60f86cd7c85c4bb1b92908cd6a"
)
_COMPANION = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_interface_frontier_companion_20260722/48e7913/a0001/"
    "frontier-companion.json"
)
_COMPANION_FILE_SHA256 = (
    "703d7b458f4fcbd346521d0ecfb41ffaf8e262a99cfe480adcf9b3954c4efda9"
)
_COMPANION_SHA256 = (
    "7fa07c2c6e834f35313ef36b5a2a1d0876d8bc20be72e843a488a7ad5f6a1046"
)


class _PreflightSucceeded(RuntimeError):
    def __init__(self, document: Mapping[str, Any]) -> None:
        super().__init__("real-artifact hybrid authentication preflight succeeded")
        self.document = dict(document)


def _forbidden_attention_installer(**_kwargs: Any) -> Mapping[str, Any]:
    raise AssertionError("preflight entered fourteen-layer attention work")


def _preflight_layer_stack_compiler(**kwargs: Any) -> Any:
    import torch

    interface = kwargs.get("interface")
    mapped_parent_tied_table = kwargs.get("mapped_parent_tied_table")
    anchor_transport_witness = kwargs.get("anchor_transport_witness")
    target_model = kwargs.get("target_model")
    if (
        interface is None
        or mapped_parent_tied_table is None
        or anchor_transport_witness is None
        or target_model is None
    ):
        raise AssertionError(
            "production compiler boundary omitted the hybrid algebraic inputs"
        )
    if interface.report.get("mode") != TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE:
        raise AssertionError("production compiler boundary is not hybrid mode")

    input_weight = getattr(target_model.get_input_embeddings(), "weight", None)
    output_weight = getattr(target_model.get_output_embeddings(), "weight", None)
    if (
        not isinstance(input_weight, torch.nn.Parameter)
        or output_weight is not input_weight
        or not torch.equal(
            input_weight.detach().to(device="cpu"),
            interface.target_weight.to(device="cpu", dtype=input_weight.dtype),
        )
    ):
        raise AssertionError("live tied child differs from the hybrid interface")

    entry_threads = int(torch.get_num_threads())
    authentications: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    try:
        for requested_threads in _THREAD_SEQUENCE:
            torch.set_num_threads(requested_threads)
            authentication = dict(
                authenticate_tied_residual_congruence_interface(
                    interface,
                    mapped_parent_tied_table=mapped_parent_tied_table,
                    anchor_transport_witness=anchor_transport_witness,
                )
            )
            observed_threads = int(torch.get_num_threads())
            operation_audit = authentication.get("operation_audit")
            if (
                observed_threads != requested_threads
                or authentication.get("mode")
                != TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE
                or authentication.get("mapped_parent_tied_table_supplied") is not True
                or authentication.get("strong_base_recomputed_for_anchor_residual")
                is not True
                or authentication.get("anchor_source_bound_polar_authenticated")
                is not True
                or authentication.get("anchor_residual_formula_recomputed_exact")
                is not True
                or authentication.get("strong_base_plus_anchor_residual_claimed")
                is not True
                or authentication.get(
                    "raw_linalg_diagnostics_in_authentication_identity"
                )
                is not False
                or not isinstance(operation_audit, Mapping)
                or any(value != 0 for value in operation_audit.values())
            ):
                raise AssertionError(
                    "hybrid authentication lost its exact polar/formula contract"
                )
            authentications.append(authentication)
            records.append(
                {
                    "requested_threads": requested_threads,
                    "threads_restored_after_authentication": observed_threads,
                    "authentication_sha256": authentication["result_sha256"],
                    "polar_authenticated": True,
                    "exact_anchor_formula_authenticated": True,
                }
            )
    finally:
        torch.set_num_threads(entry_threads)

    reference = authentications[0]
    if any(value != reference for value in authentications[1:]):
        raise AssertionError(
            "hybrid authentication document changed across Torch thread states"
        )
    if len({value["result_sha256"] for value in authentications}) != 1:
        raise AssertionError(
            "hybrid authentication SHA changed across Torch thread states"
        )

    raise _PreflightSucceeded(
        {
            "schema": "neural-forge-lfm2-real-artifact-hybrid-thread-preflight-v1",
            "code_commit": _CODE_COMMIT,
            "anchor_candidate_index": 7,
            "body_recipe": "fit_guided",
            "thread_sequence": list(_THREAD_SEQUENCE),
            "entry_threads_restored": int(torch.get_num_threads()),
            "authentication_sha256": reference["result_sha256"],
            "interface_report_sha256": reference["interface_report_sha256"],
            "polar_certificate_sha256": reference[
                "anchor_source_bound_polar_certificate_sha256"
            ],
            "authentication_documents_identical": True,
            "authentication_sha256_identical": True,
            "source_bound_polar_authenticated": True,
            "exact_anchor_formula_authenticated": True,
            "fourteen_layer_work_entered": False,
            "fused_terminal_entered": False,
            "child_or_parent_forward_calls": 0,
            "training_steps": 0,
            "thread_records": records,
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-checkpoint",
        default="/tmp/neural-forge-lfm2-hybrid-thread-preflight-02cec5c",
        help="Must not exist; success leaves it absent.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output_checkpoint).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"preflight output path already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("preflight output parent must be a real directory")

    # This is the same direct-local tokenizer loader used by the production CLI;
    # it never calls from_pretrained or the Hub.
    donor_tokenizer = load_local_tokenizer(_DONOR_TOKENIZER)
    target_tokenizer = load_local_tokenizer(_BRIDGE)
    success: _PreflightSucceeded | None = None
    try:
        try:
            run_lfm2_semantic_postfold(
                input_checkpoint_directory=_INPUT_CHECKPOINT,
                expected_input_artifact_sha256=_INPUT_ARTIFACT_SHA256,
                output_checkpoint_directory=output,
                cache_directory=_TRACE_CACHE,
                expected_cache_key=_CACHE_KEY,
                bridge_directory=_BRIDGE,
                expected_bridge_artifact_sha256=_BRIDGE_SHA256,
                donor_tokenizer=donor_tokenizer,
                target_tokenizer=target_tokenizer,
                parent_model_directory=_PARENT,
                compiler_diagnostic_json=_FOUNDATION_RECEIPT,
                expected_compiler_diagnostic_sha256=_FOUNDATION_RECEIPT_SHA256,
                device=args.device,
                dtype="bfloat16",
                fold_compute_device=args.device,
                body_recipe="fit_guided",
                tied_interface_mode=TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
                interface_statistics_directory=_STATISTICS,
                expected_interface_statistics_sha256=_STATISTICS_SHA256,
                interface_frontier_companion_json=_COMPANION,
                expected_interface_frontier_companion_file_sha256=(
                    _COMPANION_FILE_SHA256
                ),
                expected_interface_frontier_companion_sha256=_COMPANION_SHA256,
                anchor_candidate_index=7,
                attention_installer=_forbidden_attention_installer,
                semantic_layer_stack_compiler=_preflight_layer_stack_compiler,
                preterminal_capsule_directory=None,
            )
        except _PreflightSucceeded as exc:
            success = exc
    finally:
        if output.exists() or output.is_symlink():
            raise AssertionError("preflight wrote an output checkpoint")

    if success is None:
        raise AssertionError("preflight unexpectedly completed the postfold runner")
    document = {
        **success.document,
        "output_checkpoint_path": str(output),
        "output_checkpoint_written": False,
        "preterminal_capsule_requested": False,
        "preterminal_capsule_written": False,
        "success_sentinel_caught": True,
    }
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
