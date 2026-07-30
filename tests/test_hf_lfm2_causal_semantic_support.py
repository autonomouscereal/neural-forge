from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_causal_semantic_support import (
    Lfm2CausalSemanticSupportError,
    authenticate_lfm2_causal_support_selection,
    lfm2_causal_support_request,
    seal_lfm2_causal_support_feature_pack,
    select_lfm2_causal_semantic_supports,
)
from neural_forge import hf_lfm2_teacher_forced_trajectory_audit as trajectory


_D = "a" * 64


def _readout(
    *,
    winner: int,
    margin: float,
    matches: bool,
    competitor: int,
) -> dict[str, object]:
    return {
        "top1_token_id": winner if matches else competitor,
        "parent_winner_token_id": winner,
        "parent_winner_rank": 1 if matches else 2,
        "parent_winner_score": margin,
        "best_competing_token_id": competitor,
        "best_competing_score": 0.0,
        "parent_winner_margin": margin,
        "top1_matches_parent": matches,
    }


def _token(
    *,
    prefix: int,
    winner: int,
    hidden_relative_l2: float,
    exact_margin: float,
    exact_matches: bool,
    folded_margin: float,
    folded_matches: bool,
    competitor: int = 4,
) -> dict[str, object]:
    return {
        "prefix_index": prefix,
        "consumed_forced_token_count": prefix,
        "next_forced_token_id": 5 if prefix == 0 else None,
        "parent_winner_token_id": winner,
        "hidden_vs_cached_parent": {
            "cosine": 1.0,
            "cosine_distance": 0.0,
            "l2": hidden_relative_l2,
            "relative_l2": hidden_relative_l2,
            "candidate_norm": 1.0,
            "parent_norm": 1.0,
        },
        "hidden_vs_same_run_exact_exact": {
            "cosine": max(0.0, 1.0 - hidden_relative_l2),
            "cosine_distance": min(1.0, hidden_relative_l2),
            "l2": hidden_relative_l2,
            "relative_l2": hidden_relative_l2,
            "candidate_norm": 1.0,
            "parent_norm": 1.0,
        },
        "readouts": {
            "exact_parent_table": _readout(
                winner=winner,
                margin=exact_margin,
                matches=exact_matches,
                competitor=competitor,
            ),
            "folded_c_out": _readout(
                winner=winner,
                margin=folded_margin,
                matches=folded_matches,
                competitor=competitor,
            ),
        },
    }


def _prompt_result(prompt_id: str = "fit-a") -> dict[str, object]:
    exact = [
        _token(
            prefix=0,
            winner=3,
            hidden_relative_l2=0.0,
            exact_margin=2.0,
            exact_matches=True,
            folded_margin=-1.0,
            folded_matches=False,
        ),
        _token(
            prefix=1,
            winner=3,
            hidden_relative_l2=0.0,
            exact_margin=2.0,
            exact_matches=True,
            folded_margin=-0.5,
            folded_matches=False,
        ),
    ]
    folded_prompt = [
        _token(
            prefix=0,
            winner=3,
            hidden_relative_l2=0.8,
            exact_margin=-1.0,
            exact_matches=False,
            folded_margin=-1.0,
            folded_matches=False,
        ),
        _token(
            prefix=1,
            winner=3,
            hidden_relative_l2=0.7,
            exact_margin=-0.5,
            exact_matches=False,
            folded_margin=-0.5,
            folded_matches=False,
        ),
    ]
    exact_folded = [
        _token(
            prefix=0,
            winner=3,
            hidden_relative_l2=0.0,
            exact_margin=2.0,
            exact_matches=True,
            folded_margin=-1.0,
            folded_matches=False,
        ),
        _token(
            prefix=1,
            winner=3,
            hidden_relative_l2=0.0,
            exact_margin=2.0,
            exact_matches=True,
            folded_margin=-0.5,
            folded_matches=False,
        ),
    ]
    lanes = {
        "exact_exact": {"summary": {}, "tokens": exact},
        "folded_exact": {"summary": {}, "tokens": folded_prompt},
        "exact_folded": {"summary": {}, "tokens": exact_folded},
        "folded_folded": {"summary": {}, "tokens": folded_prompt},
    }
    body = {
        "schema": trajectory.LFM2_TEACHER_FORCED_TRAJECTORY_PROMPT_RESULT_SCHEMA,
        "batch_index": 0,
        "prompt_id": prompt_id,
        "prompt_token_count": 2,
        "forced_token_count": 1,
        "state_count": 2,
        "parent_reference_tokens": [
            {"prefix_index": 0, "next_forced_token_id": 5},
            {"prefix_index": 1, "next_forced_token_id": None},
        ],
        "lanes": lanes,
        "same_run_hidden_contrast_contract": dict(
            trajectory.SAME_RUN_HIDDEN_CONTRAST_CONTRACT
        ),
        "target_token_spans_used_as_body_steps": False,
        "operation_audit": dict(trajectory._OPERATION_AUDIT),
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def _audit() -> dict[str, object]:
    prompt = _prompt_result()
    body = {
        "schema": trajectory.LFM2_TEACHER_FORCED_TRAJECTORY_AUDIT_SCHEMA,
        "prompt_count": 1,
        "prompt_results": [prompt],
        "operation_audit": dict(trajectory._OPERATION_AUDIT),
    }
    return {**body, "audit_sha256": canonical_sha256(body)}


def _tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    donor = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.7],
            [-0.2, 1.0],
            [0.8, 0.8],
        ],
        dtype=torch.float32,
    )
    return donor, target, torch.tensor([0, 1, 2], dtype=torch.int64)


def _routes() -> tuple[torch.Tensor, ...]:
    offsets = torch.tensor([0, 1, 2, 3, 5, 7, 9], dtype=torch.int64)
    ids = torch.tensor([0, 1, 2, 0, 1, 1, 2, 0, 2], dtype=torch.int64)
    coefficients = torch.tensor(
        [1.0, 1.0, 1.0, 0.5, 0.5, 0.6, 0.4, 0.5, 0.5],
        dtype=torch.float32,
    )
    return (
        offsets,
        ids,
        coefficients,
        offsets.clone(),
        ids.clone(),
        coefficients.clone(),
    )


def _feature_pack(audit: dict[str, object]):
    return seal_lfm2_causal_support_feature_pack(
        input_feature_transform=torch.eye(2, dtype=torch.float64),
        # The output functional view sees x0-x1.  This intentionally gives a
        # different local support ordering than the input geometry.
        output_feature_transform=torch.tensor([[1.0, -1.0]], dtype=torch.float64),
        trajectory_audit_sha256=str(audit["audit_sha256"]),
        fit_prompt_ids=["fit-a"],
        manual_review_prompt_ids=["manual-review-a"],
        topology_sha256=_D,
        numeric_contract_sha256="b" * 64,
        source_artifact_sha256s={"trajectory": "c" * 64},
    )


def _arguments() -> dict[str, object]:
    audit = _audit()
    donor, target, mapping = _tables()
    routes = _routes()
    return {
        "trajectory_audit": audit,
        "feature_pack": _feature_pack(audit),
        "prompt_token_ids_by_prompt": {"fit-a": [3, 0]},
        "donor_table": donor,
        "target_table": target,
        "target_to_donor": mapping,
        "base_input_offsets": routes[0],
        "base_input_target_ids": routes[1],
        "base_input_coefficients": routes[2],
        "base_output_offsets": routes[3],
        "base_output_target_ids": routes[4],
        "base_output_coefficients": routes[5],
        "physical_child_parameter_count": 100,
        "maximum_combined_coefficient_count": 18,
        "input_support_width": 1,
        "output_support_width": 1,
        "maximum_input_donor_repairs": 1,
        "maximum_output_donor_repairs": 1,
        "position_bin_count": 2,
        "context_weight": 0.25,
        "solve_ridge": 1e-8,
        "solve_dtype": "float64",
    }


def _route_row(
    offsets: torch.Tensor,
    ids: torch.Tensor,
    coefficients: torch.Tensor,
    donor_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = int(offsets[donor_id].item())
    stop = int(offsets[donor_id + 1].item())
    return ids[start:stop], coefficients[start:stop]


def test_independent_causal_search_recloses_only_damaged_rows() -> None:
    arguments = _arguments()
    result = select_lfm2_causal_semantic_supports(**arguments)

    assert result.input_reselected_donor_ids.tolist() == [3]
    assert result.output_reselected_donor_ids.tolist() == [3]
    input_support, _ = _route_row(
        result.input_offsets,
        result.input_target_ids,
        result.input_coefficients,
        3,
    )
    output_support, _ = _route_row(
        result.output_offsets,
        result.output_target_ids,
        result.output_coefficients,
        3,
    )
    assert input_support.tolist() == [2]
    assert output_support.tolist() == [0]
    assert not torch.equal(input_support, output_support)

    routes = _routes()
    for donor_id in (0, 1, 2, 4, 5):
        base_input = _route_row(routes[0], routes[1], routes[2], donor_id)
        selected_input = _route_row(
            result.input_offsets,
            result.input_target_ids,
            result.input_coefficients,
            donor_id,
        )
        base_output = _route_row(routes[3], routes[4], routes[5], donor_id)
        selected_output = _route_row(
            result.output_offsets,
            result.output_target_ids,
            result.output_coefficients,
            donor_id,
        )
        assert all(
            torch.equal(left, right) for left, right in zip(base_input, selected_input)
        )
        assert all(
            torch.equal(left, right)
            for left, right in zip(base_output, selected_output)
        )

    receipt = result.as_dict()
    assert receipt["input_output_support_searches_independent"]
    assert not receipt["output_support_ids_inherited_from_input"]
    assert receipt["manual_review_prompts_consumed"] == 0
    assert (
        receipt["damage_diagnostics"]["raw_occurrence_frequency_used_as_priority"]
        is False
    )
    assert receipt["input_context_dimension_count"] > 0
    assert receipt["output_context_dimension_count"] > 0
    assert receipt["continuous_child_state_count"] == 16
    assert receipt["effective_child_parameter_count"] == 116
    assert receipt["operation_audit"]["training_steps"] == 0
    assert receipt["operation_audit"]["parent_forward_calls"] == 0


def test_request_is_content_addressed_and_binds_context_math() -> None:
    arguments = _arguments()
    first = lfm2_causal_support_request(**arguments)
    second = lfm2_causal_support_request(**arguments)
    changed = lfm2_causal_support_request(**{**arguments, "context_weight": 0.5})
    assert first["cache_key"] == second["cache_key"]
    assert first["cache_key"] != changed["cache_key"]
    assert first["selection_and_manual_review_prompt_ids_disjoint"]
    assert not first["selection_evidence_is_frequency_only"]
    assert first["dtype_contract"]["solve_dtype"] == "float64"
    assert first["topology_sha256"] == _D


def test_loaded_result_authentication_binds_tensor_and_cache_key() -> None:
    result = select_lfm2_causal_semantic_supports(**_arguments())
    loaded = authenticate_lfm2_causal_support_selection(
        input_offsets=result.input_offsets,
        input_target_ids=result.input_target_ids,
        input_coefficients=result.input_coefficients,
        output_offsets=result.output_offsets,
        output_target_ids=result.output_target_ids,
        output_coefficients=result.output_coefficients,
        input_reselected_donor_ids=result.input_reselected_donor_ids,
        output_reselected_donor_ids=result.output_reselected_donor_ids,
        receipt=result.as_dict(),
        expected_cache_key=result.cache_key,
    )
    assert loaded.result_sha256 == result.result_sha256

    tampered = result.output_coefficients.clone()
    tampered[-1] += 0.25
    with pytest.raises(Lfm2CausalSemanticSupportError, match="binding"):
        authenticate_lfm2_causal_support_selection(
            input_offsets=result.input_offsets,
            input_target_ids=result.input_target_ids,
            input_coefficients=result.input_coefficients,
            output_offsets=result.output_offsets,
            output_target_ids=result.output_target_ids,
            output_coefficients=tampered,
            input_reselected_donor_ids=result.input_reselected_donor_ids,
            output_reselected_donor_ids=result.output_reselected_donor_ids,
            receipt=result.as_dict(),
            expected_cache_key=result.cache_key,
        )


def test_feature_pack_rejects_review_prompt_leakage() -> None:
    audit = _audit()
    with pytest.raises(Lfm2CausalSemanticSupportError, match="disjoint"):
        seal_lfm2_causal_support_feature_pack(
            input_feature_transform=torch.eye(2),
            output_feature_transform=torch.eye(2),
            trajectory_audit_sha256=str(audit["audit_sha256"]),
            fit_prompt_ids=["same"],
            manual_review_prompt_ids=["same"],
            topology_sha256=_D,
            numeric_contract_sha256="b" * 64,
            source_artifact_sha256s={"trajectory": "c" * 64},
        )


def test_prompt_manifest_and_audit_tampering_fail_closed() -> None:
    arguments = _arguments()
    with pytest.raises(Lfm2CausalSemanticSupportError, match="length"):
        lfm2_causal_support_request(
            **{**arguments, "prompt_token_ids_by_prompt": {"fit-a": [3]}}
        )

    tampered_audit = deepcopy(arguments["trajectory_audit"])
    tampered_audit["prompt_count"] = 2
    with pytest.raises(Lfm2CausalSemanticSupportError, match="authentication"):
        lfm2_causal_support_request(**{**arguments, "trajectory_audit": tampered_audit})


def test_localized_growth_cannot_escape_counted_coefficient_budget() -> None:
    arguments = _arguments()
    with pytest.raises(Lfm2CausalSemanticSupportError, match="budget"):
        select_lfm2_causal_semantic_supports(
            **{
                **arguments,
                "input_support_width": 3,
                "output_support_width": 3,
                "maximum_combined_coefficient_count": 18,
            }
        )
