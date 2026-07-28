from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments import run_lfm2_progressive_component_ablation as runner
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_progressive_component_ablation import (
    _module_state_sha256,
    build_lfm2_progressive_component_ablation_schedule,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def test_candidate_window_supports_authenticated_non_rerun_continuations() -> None:
    assert runner._candidate_window(
        candidate_count=59,
        start_index=43,
        stop_index=59,
    ) == (43, 59)
    assert runner._candidate_window(
        candidate_count=59,
        start_index=0,
        stop_index=None,
    ) == (0, 59)
    with pytest.raises(
        runner.Lfm2ProgressiveComponentAblationRunError,
        match="0 <= start < stop",
    ):
        runner._candidate_window(
            candidate_count=59,
            start_index=43,
            stop_index=43,
        )


def test_candidate_checkpoint_name_bounds_long_cumulative_id_without_losing_identity() -> None:
    candidate_id = "cumulative--" + "--".join(
        f"layer-{index:03d}-attention-and-mlp" for index in range(80)
    )
    first = runner._candidate_checkpoint_name(
        candidate_index=43,
        candidate_id=candidate_id,
    )
    second = runner._candidate_checkpoint_name(
        candidate_index=43,
        candidate_id=candidate_id,
    )

    assert first == second
    assert len(first.encode("utf-8")) < 255
    assert first.startswith("candidate-0043-")
    assert candidate_id not in first
    assert runner._candidate_checkpoint_name(
        candidate_index=44,
        candidate_id=candidate_id,
    ) != first


class _Attention(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.head_dim = 1
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.out_proj = torch.nn.Linear(hidden, hidden, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = _Attention(hidden)


class _Model(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Layer(hidden)])


def _write_plan(
    path: Path,
    *,
    bridge_sha256: str,
    parent: _Model,
    compact: _Model,
    encoder: torch.Tensor,
    decoder: torch.Tensor,
) -> dict:
    body = {
        "schema": runner.PLAN_SCHEMA,
        "status": runner.AUTOMATIC_PLAN_STATUS,
        "producer_schema": runner.PLAN_PRODUCER_SCHEMA,
        "parent_model_state_sha256": _sha("parent-model"),
        "compact_model_state_sha256": _sha("compact-model"),
        "parent_checkpoint_inventory_sha256": _sha("parent-checkpoint"),
        "compact_checkpoint_inventory_sha256": _sha("compact-checkpoint"),
        "source_authentication": {
            "source_run_result_sha256": _sha("source-run"),
            "parent_checkpoint_inventory_sha256": _sha("parent-checkpoint"),
            "compact_checkpoint_inventory_sha256": _sha("compact-checkpoint"),
            "component_fold_receipts": {
                "attention-layer-0": {
                    "receipt_sha256": _sha("fold-receipt"),
                    "receipt_seal_freshly_verified": True,
                }
            },
        },
        "bridge_archive": {
            "schema": runner.BRIDGE_ARCHIVE_SCHEMA,
            "filename": runner.BRIDGE_FILENAME,
            "file_sha256": bridge_sha256,
        },
        "components": [
            {
                "component_id": "attention-layer-0",
                "kind": "attention",
                "parent_module_path": "model.layers.0.self_attn",
                "compact_module_path": "model.layers.0.self_attn",
                "parent_module_sha256": _module_state_sha256(
                    parent.model.layers[0].self_attn
                ),
                "compact_module_sha256": _module_state_sha256(
                    compact.model.layers[0].self_attn
                ),
                "input_encoder_tensor": "attention.0.input_encoder",
                "output_decoder_tensor": "attention.0.output_decoder",
                "output_domain_decoder_tensor": None,
                "input_encoder_sha256": lfm2_authenticated_tensor_sha256(encoder),
                "output_decoder_sha256": lfm2_authenticated_tensor_sha256(decoder),
                "output_domain_decoder_sha256": None,
                "source_fold_receipt_sha256": _sha("fold-receipt"),
                "source_compact_checkpoint_sha256": _sha("compact-checkpoint"),
                "require_parameter_reduction": True,
            }
        ],
        "excluded_components": [],
        "component_granularity": "whole-subsystem",
        "q_k_v_o_or_individual_head_isolation_claimed": False,
        "all_included_components_require_literal_parameter_reduction_true": True,
        "diagnostic_hybrid_only": True,
        "standalone_compression_claimed": False,
    }
    document = {**body, "plan_sha256": canonical_sha256(body)}
    path.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return document


def _write_producer_receipt(
    path: Path,
    *,
    plan_path: Path,
    bridge_path: Path,
    plan: dict,
) -> dict:
    body = {
        "schema": runner.PLAN_PRODUCER_SCHEMA,
        "status": runner.PRODUCER_RECEIPT_STATUS,
        "plan_filename": runner.PLAN_FILENAME,
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": _file_sha256(plan_path),
        "bridge_filename": runner.BRIDGE_FILENAME,
        "bridge_file_sha256": _file_sha256(bridge_path),
        "source_run_result_sha256": _sha("source-run"),
        "parent_checkpoint_inventory_sha256": _sha("parent-checkpoint"),
        "compact_checkpoint_inventory_sha256": _sha("compact-checkpoint"),
        "component_count": len(plan["components"]),
        "excluded_component_count": len(plan["excluded_components"]),
        "component_granularity": "whole-subsystem",
        "automatic_topology_and_component_derivation_performed": True,
        "fresh_source_receipt_authentication_performed": True,
        "fresh_archive_authentication_performed": True,
        "human_authored_module_or_receipt_pins_accepted": False,
        "training_steps": 0,
        "model_forward_calls": 0,
    }
    document = {**body, "receipt_sha256": canonical_sha256(body)}
    path.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return document


def test_sealed_plan_builds_exact_capsule_from_pinned_bridge_archive(
    tmp_path: Path,
) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    parent = _Model(4).requires_grad_(False)
    compact = _Model(2).requires_grad_(False)
    encoder = torch.randn(4, 2)
    decoder = torch.randn(2, 4)
    bridge_path = tmp_path / "bridges.safetensors"
    safetensors.save_file(
        {
            "attention.0.input_encoder": encoder,
            "attention.0.output_decoder": decoder,
        },
        str(bridge_path),
    )
    bridge_sha256 = _file_sha256(bridge_path)
    plan_path = tmp_path / "plan.json"
    expected = _write_plan(
        plan_path,
        bridge_sha256=bridge_sha256,
        parent=parent,
        compact=compact,
        encoder=encoder,
        decoder=decoder,
    )

    plan = runner._load_plan(
        plan_path,
        expected_plan_sha256=expected["plan_sha256"],
        expected_file_sha256=_file_sha256(plan_path),
    )
    tensors = runner._load_bridge_tensors(
        bridge_path,
        expected_file_sha256=bridge_sha256,
        plan=plan,
    )
    capsules = runner._build_capsules(
        plan=plan,
        bridge_tensors=tensors,
        parent_model=parent,
        compact_model=compact,
    )
    schedule = build_lfm2_progressive_component_ablation_schedule(capsules)

    assert len(capsules) == 1
    assert capsules[0].component_id == "attention-layer-0"
    assert capsules[0].source_fold_receipt_sha256 == _sha("fold-receipt")
    torch.testing.assert_close(capsules[0].input_encoder, encoder)
    torch.testing.assert_close(capsules[0].output_decoder, decoder)
    assert [value.candidate_id for value in schedule] == ["isolated--attention-layer-0"]

    non_boolean_plan = copy.deepcopy(plan)
    non_boolean_plan["components"][0]["require_parameter_reduction"] = "true"
    with pytest.raises(
        runner.Lfm2ProgressiveComponentAblationRunError,
        match="literal JSON boolean true",
    ):
        runner._build_capsules(
            plan=non_boolean_plan,
            bridge_tensors=tensors,
            parent_model=parent,
            compact_model=compact,
        )


def test_plan_and_bridge_external_pins_fail_closed(tmp_path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    parent = _Model(4).requires_grad_(False)
    compact = _Model(2).requires_grad_(False)
    encoder = torch.randn(4, 2)
    decoder = torch.randn(2, 4)
    bridge_path = tmp_path / "bridges.safetensors"
    safetensors.save_file(
        {
            "attention.0.input_encoder": encoder,
            "attention.0.output_decoder": decoder,
        },
        str(bridge_path),
    )
    bridge_sha256 = _file_sha256(bridge_path)
    plan_path = tmp_path / "plan.json"
    expected = _write_plan(
        plan_path,
        bridge_sha256=bridge_sha256,
        parent=parent,
        compact=compact,
        encoder=encoder,
        decoder=decoder,
    )

    with pytest.raises(
        runner.Lfm2ProgressiveComponentAblationRunError,
        match="file differs",
    ):
        runner._load_plan(
            plan_path,
            expected_plan_sha256=expected["plan_sha256"],
            expected_file_sha256=_sha("wrong-file"),
        )
    plan = runner._load_plan(
        plan_path,
        expected_plan_sha256=expected["plan_sha256"],
        expected_file_sha256=_file_sha256(plan_path),
    )
    with pytest.raises(
        runner.Lfm2ProgressiveComponentAblationRunError,
        match="Bridge archive differs",
    ):
        runner._load_bridge_tensors(
            bridge_path,
            expected_file_sha256=_sha("wrong-bridge"),
            plan=plan,
        )


def test_automatic_producer_receipt_binds_publication_and_rejects_resealed_plan(
    tmp_path: Path,
) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    parent = _Model(4).requires_grad_(False)
    compact = _Model(2).requires_grad_(False)
    encoder = torch.randn(4, 2)
    decoder = torch.randn(2, 4)
    bridge_path = tmp_path / runner.BRIDGE_FILENAME
    safetensors.save_file(
        {
            "attention.0.input_encoder": encoder,
            "attention.0.output_decoder": decoder,
        },
        str(bridge_path),
    )
    plan_path = tmp_path / runner.PLAN_FILENAME
    plan = _write_plan(
        plan_path,
        bridge_sha256=_file_sha256(bridge_path),
        parent=parent,
        compact=compact,
        encoder=encoder,
        decoder=decoder,
    )
    receipt_path = tmp_path / runner.PRODUCER_RECEIPT_FILENAME
    receipt = _write_producer_receipt(
        receipt_path,
        plan_path=plan_path,
        bridge_path=bridge_path,
        plan=plan,
    )
    loaded_receipt = runner._load_producer_receipt(
        receipt_path,
        expected_receipt_sha256=receipt["receipt_sha256"],
        expected_file_sha256=_file_sha256(receipt_path),
    )
    loaded_plan = runner._load_plan(
        plan_path,
        expected_plan_sha256=plan["plan_sha256"],
        expected_file_sha256=_file_sha256(plan_path),
    )
    runner._authenticate_producer_publication(
        producer_receipt=loaded_receipt,
        producer_receipt_path=receipt_path,
        plan=loaded_plan,
        component_plan=plan_path,
        bridge_tensors=bridge_path,
        expected_component_plan_sha256=plan["plan_sha256"],
        expected_component_plan_file_sha256=_file_sha256(plan_path),
        expected_bridge_tensors_file_sha256=_file_sha256(bridge_path),
    )

    tampered_receipt = dict(receipt)
    tampered_receipt["fresh_archive_authentication_performed"] = False
    tampered_body = dict(tampered_receipt)
    tampered_body.pop("receipt_sha256")
    tampered_receipt["receipt_sha256"] = canonical_sha256(tampered_body)
    receipt_path.write_text(
        json.dumps(tampered_receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(
        runner.Lfm2ProgressiveComponentAblationRunError,
        match="producer receipt seal|Producer-receipt file|derivation policy",
    ):
        runner._load_producer_receipt(
            receipt_path,
            expected_receipt_sha256=receipt["receipt_sha256"],
            expected_file_sha256=_file_sha256(receipt_path),
        )

    hand_authored = copy.deepcopy(plan)
    hand_authored["hand_authored_but_self_sealed"] = True
    hand_body = dict(hand_authored)
    hand_body.pop("plan_sha256")
    hand_authored["plan_sha256"] = canonical_sha256(hand_body)
    plan_path.write_text(
        json.dumps(hand_authored, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    loaded_hand_authored = runner._load_plan(
        plan_path,
        expected_plan_sha256=hand_authored["plan_sha256"],
        expected_file_sha256=_file_sha256(plan_path),
    )
    with pytest.raises(
        runner.Lfm2ProgressiveComponentAblationRunError,
        match="not the publication bound",
    ):
        runner._authenticate_producer_publication(
            producer_receipt=loaded_receipt,
            producer_receipt_path=receipt_path,
            plan=loaded_hand_authored,
            component_plan=plan_path,
            bridge_tensors=bridge_path,
            expected_component_plan_sha256=hand_authored["plan_sha256"],
            expected_component_plan_file_sha256=_file_sha256(plan_path),
            expected_bridge_tensors_file_sha256=_file_sha256(bridge_path),
        )
