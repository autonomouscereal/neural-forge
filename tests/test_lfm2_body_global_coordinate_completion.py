from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from neural_forge.dynamic_weight_complete_basis import (
    FeatureOrientation,
    ResidualFacingOperator,
)
from neural_forge.lfm2_body_global_coordinate_completion import (
    LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECEIPT_SCHEMA,
    Lfm2BodyGlobalCoordinateCompletion,
    Lfm2BodyGlobalCoordinateCompletionError,
    compile_lfm2_body_global_coordinate_completion,
    lfm2_body_global_coordinate_completion_evidence_sha256,
)


_RECIPE = "bridge-procrustes-isometric"


def _case(
    *,
    target_width: int = 3,
    source_width: int = 7,
    trace_rank: int = 2,
    seed: int = 4100,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    child_basis = torch.linalg.qr(
        torch.randn(
            target_width,
            target_width,
            generator=generator,
            dtype=torch.float64,
        )
    ).Q
    child_rows = torch.cat(
        (
            child_basis[:, :trace_rank].T,
            1.7 * child_basis[:, :trace_rank].T,
        ),
        dim=0,
    ).contiguous()
    desired_decoder = torch.randn(
        target_width,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    parent_rows = (child_rows @ desired_decoder).contiguous()
    producer_encoder = torch.randn(
        source_width,
        target_width,
        generator=generator,
        dtype=torch.float64,
    )
    operators = (
        ResidualFacingOperator(
            role="attention_consumer",
            name="qkv",
            tensor=torch.randn(
                source_width + 2,
                source_width,
                generator=generator,
                dtype=torch.float64,
            ),
            orientation=FeatureOrientation.INPUT,
        ),
        ResidualFacingOperator(
            role="residual_writer",
            name="output",
            tensor=torch.randn(
                source_width,
                source_width + 1,
                generator=generator,
                dtype=torch.float64,
            ),
            orientation=FeatureOrientation.OUTPUT,
        ),
    )
    evidence = lfm2_body_global_coordinate_completion_evidence_sha256(
        child_rows,
        parent_rows,
        producer_encoder,
        operators,
        recipe=_RECIPE,
    )
    return {
        "child_rows": child_rows,
        "parent_rows": parent_rows,
        "producer_encoder": producer_encoder,
        "operators": operators,
        "evidence": evidence,
    }


def _compile(case: dict[str, object]):
    return compile_lfm2_body_global_coordinate_completion(
        case["child_rows"],
        case["parent_rows"],
        case["producer_encoder"],
        case["operators"],
        recipe=_RECIPE,
        expected_evidence_sha256=case["evidence"],
    )


def test_compiles_prior_centred_fit_and_preserves_its_fit_action() -> None:
    case = _case()
    child_rows = case["child_rows"]
    parent_rows = case["parent_rows"]
    encoder = case["producer_encoder"]
    prior = torch.linalg.pinv(encoder, atol=0.0, rtol=1e-12)
    expected_fit = prior + torch.linalg.pinv(
        child_rows,
        atol=0.0,
        rtol=1e-12,
    ) @ (parent_rows - child_rows @ prior)

    result = _compile(case)

    assert result.raw_decoder.shape == expected_fit.shape
    assert torch.linalg.matrix_rank(result.raw_decoder, rtol=1e-12).item() == 3
    assert torch.allclose(
        child_rows @ result.raw_decoder,
        child_rows @ expected_fit,
        rtol=2e-12,
        atol=2e-12,
    )
    assert torch.allclose(
        child_rows @ result.raw_decoder,
        parent_rows,
        rtol=2e-12,
        atol=2e-12,
    )
    assert result.receipt["schema"] == (
        LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECEIPT_SCHEMA
    )
    assert result.receipt["requested_recipe"] == _RECIPE
    assert result.receipt["recipe_selection_resolution"] in {
        "requested-recipe-materialized",
        "requested-recipe-materialized-as-identical-alias",
    }
    if result.receipt["materialized_candidate_recipe"] != _RECIPE:
        assert result.receipt["recipe_selection_resolution"] == (
            "requested-recipe-materialized-as-identical-alias"
        )
    assert result.receipt["full_rank_materialized"] is True
    assert result.receipt["coordinate_map_shape"] == list(result.raw_decoder.shape)
    assert (
        result.receipt["coordinate_map_sha256"]
        == (result.receipt["raw_decoder_sha256"])
    )
    assert (
        result.receipt["public_coordinate_map_sha256"]
        != (result.receipt["coordinate_map_sha256"])
    )
    assert result.receipt["evidence_sha256"] == case["evidence"]
    assert result.receipt["structural_rank"] == 3
    assert result.receipt["fit_partition_only"] is True
    assert result.receipt["quality_score_gate_applied"] is False
    assert result.receipt["model_forward_calls"] == 0
    assert result.receipt["training_steps"] == 0
    assert len(result.receipt["result_sha256"]) == 64
    with pytest.raises(TypeError):
        result.receipt["status"] = "tampered"


@pytest.mark.parametrize(
    ("target_width", "source_width", "trace_rank"),
    (
        (3, 9, 2),
        (7, 4, 2),
        (6, 6, 2),
    ),
    ids=(
        "narrow-child-wide-parent",
        "wide-child-narrow-parent",
        "equal-width-cross-family",
    ),
)
def test_architecture_neutral_rectangular_maps_reach_maximum_rank(
    target_width: int,
    source_width: int,
    trace_rank: int,
) -> None:
    case = _case(
        target_width=target_width,
        source_width=source_width,
        trace_rank=trace_rank,
        seed=4300 + target_width * 10 + source_width,
    )

    result = _compile(case)

    assert result.raw_decoder.shape == (target_width, source_width)
    assert torch.linalg.matrix_rank(result.raw_decoder, rtol=1e-12).item() == min(
        target_width,
        source_width,
    )
    assert result.receipt["required_full_rank"] == min(
        target_width,
        source_width,
    )
    assert torch.allclose(
        case["child_rows"] @ result.raw_decoder,
        case["parent_rows"],
        rtol=2e-12,
        atol=2e-12,
    )


def test_operator_order_and_caller_thread_count_do_not_change_result() -> None:
    case = _case(seed=4500)
    initial_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        first = _compile(case)
        reversed_operators = tuple(reversed(case["operators"]))
        reversed_evidence = lfm2_body_global_coordinate_completion_evidence_sha256(
            case["child_rows"],
            case["parent_rows"],
            case["producer_encoder"],
            reversed_operators,
            recipe=_RECIPE,
        )
        torch.set_num_threads(max(2, min(4, initial_threads)))
        second = compile_lfm2_body_global_coordinate_completion(
            case["child_rows"],
            case["parent_rows"],
            case["producer_encoder"],
            reversed_operators,
            recipe=_RECIPE,
            expected_evidence_sha256=reversed_evidence,
        )
    finally:
        torch.set_num_threads(initial_threads)

    assert case["evidence"] == reversed_evidence
    assert torch.equal(first.raw_decoder, second.raw_decoder)
    assert first.receipt["result_sha256"] == second.receipt["result_sha256"]


@pytest.mark.parametrize("field", ("child_rows", "parent_rows", "producer_encoder"))
def test_tampered_cached_tensor_cannot_reuse_evidence(field: str) -> None:
    case = _case(seed=4700)
    changed = dict(case)
    changed[field] = case[field].clone()
    changed[field][0, 0] += 0.125

    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="authentication failed",
    ):
        _compile(changed)


def test_mutated_parent_operator_cannot_reuse_authenticated_evidence() -> None:
    case = _case(seed=4800)
    case["operators"][0].tensor[0, 0] += 0.5

    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="changed after authentication",
    ):
        _compile(case)


def test_recipe_is_explicit_and_unknown_recipe_is_rejected() -> None:
    case = _case(seed=4900)
    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="Unsupported body-global completion recipe",
    ):
        compile_lfm2_body_global_coordinate_completion(
            case["child_rows"],
            case["parent_rows"],
            case["producer_encoder"],
            case["operators"],
            recipe="implicit-best-score",
            expected_evidence_sha256=case["evidence"],
        )


def test_inputs_are_owned_and_never_mutated() -> None:
    case = _case(seed=5000)
    child_before = case["child_rows"].clone()
    parent_before = case["parent_rows"].clone()
    encoder_before = case["producer_encoder"].clone()
    operator_before = tuple(value.tensor.clone() for value in case["operators"])

    result = _compile(case)

    assert torch.equal(case["child_rows"], child_before)
    assert torch.equal(case["parent_rows"], parent_before)
    assert torch.equal(case["producer_encoder"], encoder_before)
    assert all(
        torch.equal(operator.tensor, before)
        for operator, before in zip(
            case["operators"],
            operator_before,
            strict=True,
        )
    )
    assert result.raw_decoder.device.type == "cpu"
    assert result.raw_decoder.dtype == torch.float64
    assert result.raw_decoder.requires_grad is False


def test_result_wrapper_rejects_a_decoder_that_no_longer_matches_its_receipt() -> None:
    result = _compile(_case(seed=5050))
    tampered_decoder = result.raw_decoder.clone()
    tampered_decoder[0, 0] += 0.25

    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="does not authenticate raw_decoder",
    ):
        Lfm2BodyGlobalCoordinateCompletion(
            raw_decoder=tampered_decoder,
            frontier=result.frontier,
            receipt=result.receipt,
        )


def test_module_has_no_model_execution_training_or_filesystem_writes() -> None:
    source = Path(
        "src/neural_forge/lfm2_body_global_coordinate_completion.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called_names.intersection(
        {"forward", "generate", "backward", "step", "save", "open"}
    )
    assert not called_attributes.intersection(
        {
            "forward",
            "generate",
            "backward",
            "step",
            "save",
            "write_text",
            "write_bytes",
            "requires_grad_",
        }
    )
    assert "transformers" not in source.lower()
