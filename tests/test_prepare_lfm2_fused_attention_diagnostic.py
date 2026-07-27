from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments import prepare_lfm2_fused_attention_diagnostic as producer
from experiments import (
    prepare_lfm2_progressive_component_ablation as component_producer,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from tests import test_hf_lfm2_fused_attention_diagnostic as diagnostic_fixture
from tests import test_prepare_lfm2_progressive_component_ablation as component_fixture


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _component_publication(tmp_path: Path, fixture: dict) -> tuple[Path, dict]:
    output = tmp_path / "component-publication"
    receipt = component_producer.prepare(
        output_directory=output,
        source_run_directory=fixture["source_root"],
        expected_source_run_result_sha256=fixture["run"]["result_sha256"],
        expected_source_run_file_sha256=_file_sha256(
            fixture["source_root"] / "run-receipt.json"
        ),
        parent_checkpoint=fixture["parent_checkpoint"],
        expected_parent_checkpoint_inventory_sha256=canonical_sha256(
            component_producer._checkpoint_inventory(fixture["parent_checkpoint"])
        ),
        compact_checkpoint=fixture["compact_checkpoint"],
        expected_compact_checkpoint_inventory_sha256=canonical_sha256(
            component_producer._checkpoint_inventory(fixture["compact_checkpoint"])
        ),
        expected_seed_archive_manifest_sha256=fixture["seed_auth"]["manifest_sha256"],
        expected_candidate_archive_manifest_sha256=fixture["candidate_auth"][
            "manifest_sha256"
        ],
    )
    return output, dict(receipt)


def test_schedule_orders_isolated_then_groups_prefixes_and_deduplicates() -> None:
    def layer(index: int, first: str, second: str) -> dict:
        return {
            "layer_index": index,
            "base_variant_ids": ["s0", "s2"],
            "explicit_group_variant_ids": [],
            "candidates": {
                "s0": {"candidate_state_sha256": first},
                "s2": {"candidate_state_sha256": second},
            },
        }

    schedule = producer._schedule_document(
        [
            layer(0, "a" * 64, "b" * 64),
            layer(2, "c" * 64, "d" * 64),
            layer(4, "e" * 64, "f" * 64),
        ]
    )
    modes = [value["mode"] for value in schedule["candidates"]]
    assert modes[:6] == ["isolated-layer-variant"] * 6
    assert modes[6:8] == ["all-attention-same-variant"] * 2
    assert modes[8:] == ["ordered-attention-prefix"] * 2
    assert schedule["schedule_sha256"] == canonical_sha256(
        {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    )

    deduped = producer._schedule_document(
        [
            layer(0, "a" * 64, "a" * 64),
            layer(2, "b" * 64, "b" * 64),
        ]
    )
    assert deduped["candidate_count"] == 3
    assert len(deduped["deduplicated_candidate_aliases"]) == 3


def test_prepare_authenticates_existing_publication_and_emits_sealed_bank(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = component_fixture._source_fixture(tmp_path)
    component_output, component_receipt = _component_publication(tmp_path, fixture)
    fake_source = diagnostic_fixture._source()
    fake_bank = diagnostic_fixture._bank(fake_source)
    fake_authentication = (
        diagnostic_fixture.fingerprint_lfm2_fused_attention_diagnostic_input(
            fake_source
        )
    )
    monkeypatch.setattr(
        producer,
        "fingerprint_lfm2_fused_attention_diagnostic_input",
        lambda _source: fake_authentication,
    )
    monkeypatch.setattr(
        producer,
        "compile_lfm2_fused_attention_diagnostic_bank",
        lambda _source, **_kwargs: fake_bank,
    )
    output = tmp_path / "fused-publication"
    receipt = producer.prepare(
        output_directory=output,
        source_run_directory=fixture["source_root"],
        expected_source_run_result_sha256=fixture["run"]["result_sha256"],
        expected_source_run_file_sha256=_file_sha256(
            fixture["source_root"] / "run-receipt.json"
        ),
        parent_checkpoint=fixture["parent_checkpoint"],
        expected_parent_checkpoint_inventory_sha256=canonical_sha256(
            component_producer._checkpoint_inventory(fixture["parent_checkpoint"])
        ),
        final_checkpoint=fixture["compact_checkpoint"],
        expected_final_checkpoint_inventory_sha256=canonical_sha256(
            component_producer._checkpoint_inventory(fixture["compact_checkpoint"])
        ),
        component_producer_receipt=(
            component_output / component_producer.PRODUCER_RECEIPT_FILENAME
        ),
        expected_component_producer_receipt_sha256=component_receipt["receipt_sha256"],
        expected_component_producer_receipt_file_sha256=_file_sha256(
            component_output / component_producer.PRODUCER_RECEIPT_FILENAME
        ),
        component_plan=component_output / component_producer.PLAN_FILENAME,
        expected_component_plan_sha256=component_receipt["plan_sha256"],
        expected_component_plan_file_sha256=_file_sha256(
            component_output / component_producer.PLAN_FILENAME
        ),
        component_bridges=component_output / component_producer.BRIDGE_FILENAME,
        expected_component_bridges_file_sha256=_file_sha256(
            component_output / component_producer.BRIDGE_FILENAME
        ),
        expected_seed_archive_manifest_sha256=fixture["seed_auth"]["manifest_sha256"],
        expected_structural_archive_manifest_sha256=fixture["candidate_auth"][
            "manifest_sha256"
        ],
    )
    plan = json.loads((output / producer.PLAN_FILENAME).read_text(encoding="ascii"))
    assert receipt["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    assert plan["source_authentication"]["all_source_tensor_pins_embedded_per_layer"]
    assert plan["operation_audit"]["model_generate_calls"] == 0
    assert plan["operation_audit"]["training_steps"] == 0
    assert plan["byte_identical_candidate_states_stored_once"] is True
    assert plan["schedule"]["candidate_count"] > 0
    assert (
        plan["parent_tokenizer_sha256"]
        == plan["source_authentication"]["parent_tokenizer_sha256"]
        == receipt["parent_tokenizer_sha256"]
    )
    assert (output / producer.BANK_FILENAME).is_file()


def test_prepare_rejects_output_nested_in_authenticated_checkpoint_without_mutation(
    tmp_path: Path,
) -> None:
    fixture = component_fixture._source_fixture(tmp_path)
    component_output, component_receipt = _component_publication(tmp_path, fixture)
    parent_inventory_before = component_producer._checkpoint_inventory(
        fixture["parent_checkpoint"]
    )
    forbidden = fixture["parent_checkpoint"] / "fused-attention-output"

    with pytest.raises(
        producer.Lfm2FusedAttentionPlanProducerError,
        match="outside every authenticated input directory",
    ):
        producer.prepare(
            output_directory=forbidden,
            source_run_directory=fixture["source_root"],
            expected_source_run_result_sha256=fixture["run"]["result_sha256"],
            expected_source_run_file_sha256=_file_sha256(
                fixture["source_root"] / "run-receipt.json"
            ),
            parent_checkpoint=fixture["parent_checkpoint"],
            expected_parent_checkpoint_inventory_sha256=canonical_sha256(
                parent_inventory_before
            ),
            final_checkpoint=fixture["compact_checkpoint"],
            expected_final_checkpoint_inventory_sha256=canonical_sha256(
                component_producer._checkpoint_inventory(fixture["compact_checkpoint"])
            ),
            component_producer_receipt=(
                component_output / component_producer.PRODUCER_RECEIPT_FILENAME
            ),
            expected_component_producer_receipt_sha256=component_receipt[
                "receipt_sha256"
            ],
            expected_component_producer_receipt_file_sha256=_file_sha256(
                component_output / component_producer.PRODUCER_RECEIPT_FILENAME
            ),
            component_plan=component_output / component_producer.PLAN_FILENAME,
            expected_component_plan_sha256=component_receipt["plan_sha256"],
            expected_component_plan_file_sha256=_file_sha256(
                component_output / component_producer.PLAN_FILENAME
            ),
            component_bridges=component_output / component_producer.BRIDGE_FILENAME,
            expected_component_bridges_file_sha256=_file_sha256(
                component_output / component_producer.BRIDGE_FILENAME
            ),
            expected_seed_archive_manifest_sha256=fixture["seed_auth"][
                "manifest_sha256"
            ],
            expected_structural_archive_manifest_sha256=fixture["candidate_auth"][
                "manifest_sha256"
            ],
        )

    assert not forbidden.exists()
    assert (
        component_producer._checkpoint_inventory(fixture["parent_checkpoint"])
        == parent_inventory_before
    )
