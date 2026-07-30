from __future__ import annotations

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_semantic_carrier_allocation import (
    LFM2_SEMANTIC_CARRIER_ALLOCATION_SCHEMA,
    LFM2_SEMANTIC_CARRIER_MATERIALIZATION_SCHEMA,
    Lfm2SemanticCarrierAllocationError,
    build_lfm2_semantic_carrier_demand_evidence,
    build_lfm2_semantic_carrier_inventory,
    compile_lfm2_semantic_carrier_allocation,
    materialize_lfm2_semantic_carrier_allocation,
    select_lfm2_semantic_carriers,
)
from neural_forge.hf_lfm2_semantic_step_token_fold import (
    bind_lfm2_separate_sparse_semantic_code_runtime,
    seal_lfm2_semantic_step_span_evidence,
)


def _csr(
    rows: list[tuple[list[int], list[float]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = [0]
    target_ids: list[int] = []
    coefficients: list[float] = []
    for row_ids, row_coefficients in rows:
        target_ids.extend(row_ids)
        coefficients.extend(row_coefficients)
        offsets.append(len(target_ids))
    return (
        torch.tensor(offsets, dtype=torch.int64),
        torch.tensor(target_ids, dtype=torch.int64),
        torch.tensor(coefficients, dtype=torch.float32),
    )


def _row(
    offsets: torch.Tensor,
    target_ids: torch.Tensor,
    coefficients: torch.Tensor,
    donor_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = int(offsets[donor_id])
    stop = int(offsets[donor_id + 1])
    return target_ids[start:stop], coefficients[start:stop]


def _case():
    mapping = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    direct_rows = [([target_id], [1.0]) for target_id in range(5)]
    input_code = _csr(
        direct_rows
        + [
            ([0], [0.5]),
            ([1], [0.5]),
            ([1], [0.25]),
        ]
    )
    output_code = _csr(
        direct_rows
        + [
            ([0], [0.5]),
            ([2], [0.5]),
            ([1], [0.75]),
        ]
    )
    target_table = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ],
        dtype=torch.bfloat16,
    )
    parent_table = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.25, 0.25, 0.5],
            [0.125, 0.625, 0.25],
            [0.75, 0.125, 0.125],
        ],
        dtype=torch.float64,
    )
    inventory = build_lfm2_semantic_carrier_inventory(
        input_offsets=input_code[0],
        input_target_ids=input_code[1],
        input_coefficients=input_code[2],
        output_offsets=output_code[0],
        output_target_ids=output_code[1],
        output_coefficients=output_code[2],
        target_to_donor=mapping,
    )
    evidence = build_lfm2_semantic_carrier_demand_evidence(
        torch.tensor([20, 20, 20, 0, 0, 1, 2, 2], dtype=torch.int64),
        torch.zeros(8, dtype=torch.int64),
        torch.tensor([0, 0, 0, 0, 0, 10, 5, 5], dtype=torch.float64),
        torch.zeros(8, dtype=torch.float64),
        source_artifact_sha256="a" * 64,
        source_corpus_sha256="b" * 64,
        parent_identity_sha256="c" * 64,
        fit_partition_manifest_sha256="d" * 64,
        producer_implementation_sha256="e" * 64,
        numeric_contract_sha256="f" * 64,
        base_semantic_binding_sha256=canonical_sha256(
            inventory.as_dict()["base_binding"]
        ),
        partition_role="fold-fit",
        selection_allowed=True,
    )
    selection = select_lfm2_semantic_carriers(
        inventory,
        evidence,
        promotion_count=2,
        protected_donor_ids=(),
        protection_artifact_sha256="1" * 64,
    )
    return (
        mapping,
        input_code,
        output_code,
        target_table,
        parent_table,
        inventory,
        evidence,
        selection,
    )


def _allocation_and_materialization():
    (
        mapping,
        input_code,
        output_code,
        target_table,
        parent_table,
        inventory,
        evidence,
        selection,
    ) = _case()
    input_replacements = _csr(
        [
            ([0, 1], [0.2, 0.8]),
            ([2], [1.0]),
        ]
    )
    output_replacements = _csr(
        [
            ([2], [0.5]),
            ([0, 1], [0.4, 0.6]),
        ]
    )
    allocation = compile_lfm2_semantic_carrier_allocation(
        selection,
        parent_table,
        target_table,
        input_evicted_offsets=input_replacements[0],
        input_evicted_target_ids=input_replacements[1],
        input_evicted_coefficients=input_replacements[2],
        output_evicted_offsets=output_replacements[0],
        output_evicted_target_ids=output_replacements[1],
        output_evicted_coefficients=output_replacements[2],
        input_replacement_artifact_sha256="2" * 64,
        output_replacement_artifact_sha256="3" * 64,
    )
    materialized = materialize_lfm2_semantic_carrier_allocation(
        allocation,
        target_table,
        target_to_donor=mapping,
        input_offsets=input_code[0],
        input_target_ids=input_code[1],
        input_coefficients=input_code[2],
        output_offsets=output_code[0],
        output_target_ids=output_code[1],
        output_coefficients=output_code[2],
    )
    return (
        mapping,
        input_code,
        output_code,
        target_table,
        parent_table,
        inventory,
        evidence,
        selection,
        allocation,
        materialized,
    )


def _positive_incidence_allocation_and_materialization():
    mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    direct_rows = [([target_id], [1.0]) for target_id in range(4)]
    input_code = _csr(
        direct_rows
        + [
            ([0, 1], [0.2, 0.3]),
            ([0], [0.5]),
            ([2], [0.6]),
            ([3], [0.7]),
        ]
    )
    output_code = _csr(
        direct_rows
        + [
            ([1], [0.4]),
            ([0, 1], [0.6, 0.7]),
            ([2], [0.8]),
            ([3], [0.9]),
        ]
    )
    target_table = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=torch.float16,
    )
    parent_table = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.2, 0.3, 0.5],
            [0.4, 0.1, 0.5],
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
        ],
        dtype=torch.float64,
    )
    inventory = build_lfm2_semantic_carrier_inventory(
        input_offsets=input_code[0],
        input_target_ids=input_code[1],
        input_coefficients=input_code[2],
        output_offsets=output_code[0],
        output_target_ids=output_code[1],
        output_coefficients=output_code[2],
        target_to_donor=mapping,
    )
    evidence = build_lfm2_semantic_carrier_demand_evidence(
        torch.tensor([0, 0, 0, 0, 0, 0, 50, 40], dtype=torch.int64),
        torch.zeros(8, dtype=torch.int64),
        torch.tensor([0, 0, 0, 0, 1, 1, 3, 2], dtype=torch.float64),
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0, 0, 0, 0], dtype=torch.float64),
        source_artifact_sha256="a" * 64,
        source_corpus_sha256="b" * 64,
        parent_identity_sha256="c" * 64,
        fit_partition_manifest_sha256="d" * 64,
        producer_implementation_sha256="e" * 64,
        numeric_contract_sha256="f" * 64,
        base_semantic_binding_sha256=canonical_sha256(
            inventory.as_dict()["base_binding"]
        ),
        partition_role="fold-fit",
        selection_allowed=True,
    )
    selection = select_lfm2_semantic_carriers(
        inventory,
        evidence,
        promotion_count=2,
        protected_donor_ids=(2, 3),
        protection_artifact_sha256="1" * 64,
        allow_positive_incidence=True,
    )
    input_replacements = _csr(
        [
            ([2], [0.11]),
            ([3], [0.12]),
            ([0, 2], [0.13, 0.14]),
            ([1, 3], [0.15, 0.16]),
        ]
    )
    output_replacements = _csr(
        [
            ([3], [0.21]),
            ([2], [0.22]),
            ([1, 3], [0.23, 0.24]),
            ([0, 2], [0.25, 0.26]),
        ]
    )
    allocation = compile_lfm2_semantic_carrier_allocation(
        selection,
        parent_table,
        target_table,
        input_evicted_offsets=input_replacements[0],
        input_evicted_target_ids=input_replacements[1],
        input_evicted_coefficients=input_replacements[2],
        output_evicted_offsets=output_replacements[0],
        output_evicted_target_ids=output_replacements[1],
        output_evicted_coefficients=output_replacements[2],
        input_replacement_artifact_sha256="2" * 64,
        output_replacement_artifact_sha256="3" * 64,
        input_solver_math_abi_sha256="4" * 64,
        output_solver_math_abi_sha256="5" * 64,
        numeric_contract_sha256="6" * 64,
    )
    materialized = materialize_lfm2_semantic_carrier_allocation(
        allocation,
        target_table,
        target_to_donor=mapping,
        input_offsets=input_code[0],
        input_target_ids=input_code[1],
        input_coefficients=input_code[2],
        output_offsets=output_code[0],
        output_target_ids=output_code[1],
        output_coefficients=output_code[2],
    )
    return (
        mapping,
        input_code,
        output_code,
        target_table,
        parent_table,
        inventory,
        evidence,
        selection,
        input_replacements,
        output_replacements,
        allocation,
        materialized,
    )


def test_inventory_uses_union_of_input_and_output_non_direct_supports() -> None:
    *_, inventory, _evidence, _selection = _case()

    assert inventory.input_non_direct_incidence.tolist() == [1, 2, 0, 0, 0]
    assert inventory.output_non_direct_incidence.tolist() == [1, 1, 1, 0, 0]
    assert inventory.union_non_direct_incidence.tolist() == [1, 2, 1, 0, 0]
    assert inventory.eligible_target_ids.tolist() == [3, 4]
    assert 2 not in inventory.eligible_target_ids.tolist()
    assert inventory.as_dict()[
        "carrier_requires_zero_union_incidence_and_zero_union_leverage"
    ]


def test_positive_incidence_selection_deduplicates_overlapping_reverse_union() -> None:
    (
        _mapping,
        _input_code,
        _output_code,
        _target_table,
        _parent_table,
        inventory,
        _evidence,
        selection,
        *_rest,
    ) = _positive_incidence_allocation_and_materialization()

    assert inventory.union_non_direct_incidence.tolist() == [2, 2, 1, 1]
    for target_id in (0, 1):
        start = int(inventory.union_non_direct_target_offsets[target_id])
        stop = int(inventory.union_non_direct_target_offsets[target_id + 1])
        assert inventory.union_non_direct_donor_ids[start:stop].tolist() == [4, 5]
    assert selection.carrier_target_ids.tolist() == [0, 1]
    assert selection.promoted_donor_ids.tolist() == [6, 7]
    assert selection.evicted_donor_ids.tolist() == [0, 1]
    assert selection.required_rerouted_donor_ids.tolist() == [0, 1, 4, 5]
    receipt = selection.as_dict()
    assert receipt["positive_incidence_target_rows_allowed"] is True
    assert receipt["required_rerouted_donor_ids"] == [0, 1, 4, 5]
    assert receipt["required_rerouted_donor_count"] == 4
    assert receipt["selection_rows"][0]["union_non_direct_incidence"] == 2
    assert (
        receipt["selection_rows"][0]["union_non_direct_l1_leverage"]
        < receipt["selection_rows"][1]["union_non_direct_l1_leverage"]
    )


def test_positive_incidence_allocation_rejects_one_missing_affected_route() -> None:
    (
        _mapping,
        _input_code,
        _output_code,
        target_table,
        parent_table,
        _inventory,
        _evidence,
        selection,
        _input_replacements,
        output_replacements,
        _allocation,
        _materialized,
    ) = _positive_incidence_allocation_and_materialization()
    missing_last_input_route = (
        torch.tensor([0, 1, 2, 3, 3], dtype=torch.int64),
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
    )

    with pytest.raises(Lfm2SemanticCarrierAllocationError, match="CSR structure"):
        compile_lfm2_semantic_carrier_allocation(
            selection,
            parent_table,
            target_table,
            input_evicted_offsets=missing_last_input_route[0],
            input_evicted_target_ids=missing_last_input_route[1],
            input_evicted_coefficients=missing_last_input_route[2],
            output_evicted_offsets=output_replacements[0],
            output_evicted_target_ids=output_replacements[1],
            output_evicted_coefficients=output_replacements[2],
            input_replacement_artifact_sha256="2" * 64,
            output_replacement_artifact_sha256="3" * 64,
            input_solver_math_abi_sha256="4" * 64,
            output_solver_math_abi_sha256="5" * 64,
            numeric_contract_sha256="6" * 64,
        )


def test_positive_incidence_materialization_keeps_independent_routes_and_dtype() -> (
    None
):
    (
        _mapping,
        input_code,
        output_code,
        target_table,
        _parent_table,
        _inventory,
        _evidence,
        selection,
        input_replacements,
        output_replacements,
        allocation,
        result,
    ) = _positive_incidence_allocation_and_materialization()

    assert result.target_table.dtype == target_table.dtype == torch.float16
    assert allocation.required_rerouted_donor_ids.tolist() == [0, 1, 4, 5]
    input_donor4 = _row(
        result.input_offsets,
        result.input_target_ids,
        result.input_coefficients,
        4,
    )
    output_donor4 = _row(
        result.output_offsets,
        result.output_target_ids,
        result.output_coefficients,
        4,
    )
    assert torch.equal(input_donor4[0], _row(*input_replacements, 2)[0])
    assert torch.equal(input_donor4[1], _row(*input_replacements, 2)[1])
    assert torch.equal(output_donor4[0], _row(*output_replacements, 2)[0])
    assert torch.equal(output_donor4[1], _row(*output_replacements, 2)[1])
    assert not torch.equal(input_donor4[1], output_donor4[1])
    for donor_id in (2, 3):
        assert torch.equal(
            _row(*input_code, donor_id)[1].view(torch.uint8),
            _row(
                result.input_offsets,
                result.input_target_ids,
                result.input_coefficients,
                donor_id,
            )[1].view(torch.uint8),
        )
        assert torch.equal(
            _row(*output_code, donor_id)[1].view(torch.uint8),
            _row(
                result.output_offsets,
                result.output_target_ids,
                result.output_coefficients,
                donor_id,
            )[1].view(torch.uint8),
        )
    allocation_receipt = allocation.as_dict()
    assert allocation_receipt["stored_evicted_coefficient_count"] == 4
    assert allocation_receipt["stored_rerouted_coefficient_count"] == 12
    assert allocation_receipt["stored_rerouted_coefficient_bytes"] == 48
    for channel, math_abi in (("input", "4" * 64), ("output", "5" * 64)):
        solver = allocation_receipt[f"{channel}_replacement_solver_receipt"]
        assert (
            solver["post_replant_target_table_sha256"]
            == allocation_receipt["post_replant_target_table_sha256"]
        )
        assert solver["rerouted_donor_ids"] == [0, 1, 4, 5]
        assert solver["solver_math_abi_sha256"] == math_abi
        assert solver["numeric_contract_sha256"] == "6" * 64
        assert solver["routes_are_complete_for_exact_rerouted_set"] is True
    result_receipt = result.as_dict()
    expected_coefficients = int(
        result.input_coefficients.numel() + result.output_coefficients.numel()
    )
    assert (
        result_receipt["complete_materialized_continuous_state_count"]
        == int(result.target_table.numel()) + expected_coefficients
    )
    assert result_receipt["target_table_counted_once_in_complete_continuous_state"]


def test_positive_incidence_materialized_routes_bind_to_runtime() -> None:
    *_, selection, _input_routes, _output_routes, _allocation, result = (
        _positive_incidence_allocation_and_materialization()
    )
    span_evidence = seal_lfm2_semantic_step_span_evidence(
        [[0], [1], [2], [3], [0], [1], [2], [3]],
        target_row_count=4,
    )

    runtime = bind_lfm2_separate_sparse_semantic_code_runtime(
        span_evidence,
        result.target_table,
        input_semantic_code_offsets=result.input_offsets,
        input_semantic_code_target_ids=result.input_target_ids,
        input_semantic_code_coefficients=result.input_coefficients,
        output_semantic_code_offsets=result.output_offsets,
        output_semantic_code_target_ids=result.output_target_ids,
        output_semantic_code_coefficients=result.output_coefficients,
        target_to_donor=result.target_to_donor,
    )
    with runtime.authenticated_live_table_session(result.target_table) as session:
        promoted_embeddings = runtime.reconstruct_selected_donor_embeddings(
            session, selection.promoted_donor_ids
        )

    assert result.target_to_donor.tolist() == [6, 7, 2, 3]
    assert torch.equal(
        promoted_embeddings,
        result.target_table.index_select(0, selection.carrier_target_ids),
    )


def test_default_selection_retains_zero_incidence_compatibility() -> None:
    *_, selection, allocation, _result = _allocation_and_materialization()

    assert selection.as_dict()["positive_incidence_target_rows_allowed"] is False
    assert selection.required_rerouted_donor_ids.tolist() == [3, 4]
    assert allocation.required_rerouted_donor_ids.tolist() == [3, 4]
    assert allocation.input_evicted_offsets.numel() == 3


def test_selection_is_evidence_driven_and_uses_deterministic_id_ties() -> None:
    *_, inventory, evidence, selection = _case()
    repeated = select_lfm2_semantic_carriers(
        inventory,
        evidence,
        promotion_count=2,
        protected_donor_ids=(),
        protection_artifact_sha256="1" * 64,
    )

    assert selection.promoted_donor_ids.tolist() == [6, 7]
    assert selection.carrier_target_ids.tolist() == [3, 4]
    assert selection.evicted_donor_ids.tolist() == [3, 4]
    assert selection.artifact_sha256 == repeated.artifact_sha256
    assert torch.equal(selection.promoted_donor_ids, repeated.promoted_donor_ids)


def test_selection_honors_the_authenticated_protected_donor_set() -> None:
    *_, inventory, evidence, _selection = _case()

    selected = select_lfm2_semantic_carriers(
        inventory,
        evidence,
        promotion_count=1,
        protected_donor_ids=(3,),
        protection_artifact_sha256="4" * 64,
    )

    assert selected.carrier_target_ids.tolist() == [4]
    assert selected.evicted_donor_ids.tolist() == [4]
    assert selected.as_dict()["protected_donor_ids"] == [3]


def test_selection_rejects_evidence_from_another_semantic_base() -> None:
    *_, inventory, evidence, _selection = _case()
    mismatched = build_lfm2_semantic_carrier_demand_evidence(
        evidence.donor_occurrence_counts,
        evidence.donor_winner_counts,
        evidence.donor_reconstruction_errors,
        evidence.donor_eviction_costs,
        source_artifact_sha256="a" * 64,
        source_corpus_sha256="b" * 64,
        parent_identity_sha256="c" * 64,
        fit_partition_manifest_sha256="d" * 64,
        producer_implementation_sha256="e" * 64,
        numeric_contract_sha256="f" * 64,
        base_semantic_binding_sha256="0" * 64,
        partition_role="fold-fit",
        selection_allowed=True,
    )

    with pytest.raises(Lfm2SemanticCarrierAllocationError, match="base binding"):
        select_lfm2_semantic_carriers(
            inventory,
            mismatched,
            promotion_count=1,
            protected_donor_ids=(),
            protection_artifact_sha256="1" * 64,
        )


def test_materializer_preserves_injectivity_dtypes_and_exact_promoted_routes() -> None:
    (
        _mapping,
        _input_code,
        _output_code,
        target_table,
        parent_table,
        _inventory,
        _evidence,
        selection,
        allocation,
        result,
    ) = _allocation_and_materialization()

    assert allocation.as_dict()["schema"] == LFM2_SEMANTIC_CARRIER_ALLOCATION_SCHEMA
    assert result.as_dict()["schema"] == LFM2_SEMANTIC_CARRIER_MATERIALIZATION_SCHEMA
    assert result.target_to_donor.tolist() == [0, 1, 2, 6, 7]
    assert (
        torch.unique(result.target_to_donor).numel() == result.target_to_donor.numel()
    )
    assert result.target_table.dtype == target_table.dtype == torch.bfloat16
    assert torch.equal(
        result.target_table[selection.carrier_target_ids],
        parent_table[selection.promoted_donor_ids].to(dtype=torch.bfloat16),
    )
    for offsets, target_ids, coefficients in (
        (result.input_offsets, result.input_target_ids, result.input_coefficients),
        (result.output_offsets, result.output_target_ids, result.output_coefficients),
    ):
        assert offsets.device.type == "cpu" and offsets.dtype == torch.int64
        assert target_ids.device.type == "cpu" and target_ids.dtype == torch.int64
        assert coefficients.device.type == "cpu"
        assert coefficients.dtype == torch.float32
        for carrier_id, promoted_id in zip(
            selection.carrier_target_ids.tolist(),
            selection.promoted_donor_ids.tolist(),
            strict=True,
        ):
            row_ids, row_coefficients = _row(
                offsets, target_ids, coefficients, promoted_id
            )
            assert row_ids.tolist() == [carrier_id]
            assert row_coefficients.tolist() == [1.0]
    receipt = result.as_dict()
    expected_coefficients = int(
        result.input_coefficients.numel() + result.output_coefficients.numel()
    )
    assert receipt["continuous_fold_coefficient_count"] == expected_coefficients
    assert receipt["continuous_fold_coefficient_bytes"] == expected_coefficients * 4
    assert receipt["target_table_replacement_parameter_addition"] == 0
    assert (
        receipt["effective_child_parameter_addition_from_semantic_bridge"]
        == expected_coefficients
    )
    assert receipt["complete_materialized_continuous_state_count"] == (
        result.target_table.numel() + expected_coefficients
    )


def test_materializer_keeps_every_unaffected_row_byte_identical() -> None:
    (
        mapping,
        input_code,
        output_code,
        target_table,
        _parent_table,
        _inventory,
        _evidence,
        selection,
        _allocation,
        result,
    ) = _allocation_and_materialization()

    changed_donors = set(selection.promoted_donor_ids.tolist()) | set(
        selection.evicted_donor_ids.tolist()
    )
    for donor_id in range(8):
        if donor_id in changed_donors:
            continue
        for before, after in (
            (
                _row(*input_code, donor_id),
                _row(
                    result.input_offsets,
                    result.input_target_ids,
                    result.input_coefficients,
                    donor_id,
                ),
            ),
            (
                _row(*output_code, donor_id),
                _row(
                    result.output_offsets,
                    result.output_target_ids,
                    result.output_coefficients,
                    donor_id,
                ),
            ),
        ):
            assert torch.equal(before[0].view(torch.uint8), after[0].view(torch.uint8))
            assert torch.equal(before[1].view(torch.uint8), after[1].view(torch.uint8))
    for target_id in (0, 1, 2):
        assert torch.equal(
            target_table[target_id].view(torch.uint8),
            result.target_table[target_id].view(torch.uint8),
        )
        assert int(result.target_to_donor[target_id]) == int(mapping[target_id])


def test_materialized_routes_bind_to_the_role_separated_runtime() -> None:
    *_, selection, _allocation, result = _allocation_and_materialization()
    span_evidence = seal_lfm2_semantic_step_span_evidence(
        [[0], [1], [2], [3], [4], [0], [1], [2]],
        target_row_count=5,
    )

    runtime = bind_lfm2_separate_sparse_semantic_code_runtime(
        span_evidence,
        result.target_table,
        input_semantic_code_offsets=result.input_offsets,
        input_semantic_code_target_ids=result.input_target_ids,
        input_semantic_code_coefficients=result.input_coefficients,
        output_semantic_code_offsets=result.output_offsets,
        output_semantic_code_target_ids=result.output_target_ids,
        output_semantic_code_coefficients=result.output_coefficients,
        target_to_donor=result.target_to_donor,
    )

    receipt = runtime.as_dict()
    assert receipt["input_output_coefficient_storage_shared"] is False
    assert receipt["input_continuous_semantic_code_coefficient_count"] == int(
        result.input_coefficients.numel()
    )
    assert receipt["output_continuous_semantic_code_coefficient_count"] == int(
        result.output_coefficients.numel()
    )
    with runtime.authenticated_live_table_session(result.target_table) as session:
        promoted_embeddings = runtime.reconstruct_selected_donor_embeddings(
            session,
            selection.promoted_donor_ids,
        )
        target_scores = runtime.score_target_tokens(
            session,
            torch.ones(3, dtype=torch.bfloat16),
        )
        donor_logits = runtime.compute_all_donor_logits_from_target_scores(
            session,
            target_scores,
        )
    assert torch.equal(
        promoted_embeddings,
        result.target_table.index_select(0, selection.carrier_target_ids),
    )
    assert promoted_embeddings.dtype == torch.bfloat16
    assert donor_logits.dtype == torch.float32
    assert session.as_dict()["routing_accumulation_dtype"] == "float32"
    assert (
        session.as_dict()["low_precision_routing_accumulation_widened_to_fp32"] is True
    )


def test_duplicate_base_mapping_is_rejected_before_allocation() -> None:
    mapping, input_code, output_code, *_ = _case()
    invalid_mapping = mapping.clone()
    invalid_mapping[-1] = invalid_mapping[-2]

    with pytest.raises(Lfm2SemanticCarrierAllocationError, match="injective"):
        build_lfm2_semantic_carrier_inventory(
            input_offsets=input_code[0],
            input_target_ids=input_code[1],
            input_coefficients=input_code[2],
            output_offsets=output_code[0],
            output_target_ids=output_code[1],
            output_coefficients=output_code[2],
            target_to_donor=invalid_mapping,
        )


@pytest.mark.parametrize(
    ("partition_role", "selection_allowed"),
    [("evaluation", True), ("manual-review", True), ("fold-fit", False)],
)
def test_review_or_evaluation_evidence_cannot_drive_selection(
    partition_role: str,
    selection_allowed: bool,
) -> None:
    with pytest.raises(
        Lfm2SemanticCarrierAllocationError, match="partition_role='fold-fit'"
    ):
        build_lfm2_semantic_carrier_demand_evidence(
            torch.ones(8, dtype=torch.int64),
            torch.zeros(8, dtype=torch.int64),
            torch.ones(8, dtype=torch.float64),
            torch.zeros(8, dtype=torch.float64),
            source_artifact_sha256="a" * 64,
            source_corpus_sha256="b" * 64,
            parent_identity_sha256="c" * 64,
            fit_partition_manifest_sha256="d" * 64,
            producer_implementation_sha256="e" * 64,
            numeric_contract_sha256="f" * 64,
            base_semantic_binding_sha256="0" * 64,
            partition_role=partition_role,
            selection_allowed=selection_allowed,
        )


def test_allocation_requires_nonempty_replacement_route_for_every_eviction() -> None:
    *_, target_table, parent_table, _inventory, _evidence, selection = _case()
    missing_first_route = (
        torch.tensor([0, 0, 1], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1.0], dtype=torch.float32),
    )
    valid_routes = _csr([([0], [1.0]), ([1], [1.0])])

    with pytest.raises(Lfm2SemanticCarrierAllocationError, match="CSR structure"):
        compile_lfm2_semantic_carrier_allocation(
            selection,
            parent_table,
            target_table,
            input_evicted_offsets=missing_first_route[0],
            input_evicted_target_ids=missing_first_route[1],
            input_evicted_coefficients=missing_first_route[2],
            output_evicted_offsets=valid_routes[0],
            output_evicted_target_ids=valid_routes[1],
            output_evicted_coefficients=valid_routes[2],
            input_replacement_artifact_sha256="2" * 64,
            output_replacement_artifact_sha256="3" * 64,
        )
