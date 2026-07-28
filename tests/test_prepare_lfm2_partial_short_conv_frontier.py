from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments import prepare_lfm2_partial_short_conv_frontier as producer
from experiments import run_lfm2_partial_short_conv_frontier as runner
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_partial_short_conv_evidence import (
    Lfm2PartialShortConvCachedPhase,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidate(
    archive: dict[str, torch.Tensor],
    *,
    candidate_id: str,
    runtime_kind: str,
    state: dict[str, torch.Tensor],
    internal: int,
    source_elements: int,
    rank: int | None = None,
):
    return producer._candidate_row(
        candidate_id=candidate_id,
        runtime_kind=runtime_kind,
        external_hidden_size=8,
        internal_channels=internal,
        kernel_size=3,
        source_component_parameter_elements=source_elements,
        state=state,
        archive=archive,
        output_action_rank=rank,
    )


def test_canonical_phase_discovery_excludes_cached_prefill_and_is_contiguous() -> None:
    suffixes = [
        "prefill.layer.000.residual.pre",
        "cached_prefill.layer.000.residual.pre",
        "decode.000.layer.000.residual.pre",
        "decode.001.layer.000.residual.pre",
        "prefill.layer.001.residual.pre",
    ]
    assert producer._ordered_phases(tuple(sorted(suffixes))) == (
        "prefill",
        "decode.000",
        "decode.001",
    )
    selected = producer._phase_tensor_names(("prefill", "decode.000"))
    assert selected == tuple(sorted(set(selected)))
    assert len(selected) == 2 * len(producer._PHASE_SUFFIXES)
    assert all("cached_prefill" not in name for name in selected)
    with pytest.raises(
        producer.Lfm2PartialShortConvFrontierProducerError,
        match="contiguous",
    ):
        producer._ordered_phases(
            (
                "decode.001.layer.000.residual.pre",
                "prefill.layer.000.residual.pre",
            )
        )


def test_stable_row_ids_and_equal_trajectory_mass() -> None:
    first = producer._phase_row_ids(
        batch_index=7,
        phase="prefill",
        row_count=3,
    )
    second = producer._phase_row_ids(
        batch_index=7,
        phase="decode.000",
        row_count=1,
    )
    assert first == (
        "batch.000007/prefill/row.0000",
        "batch.000007/prefill/row.0001",
        "batch.000007/prefill/row.0002",
    )
    assert set(first).isdisjoint(second)
    weights = torch.full((len(first) + len(second),), 1.0 / 4.0)
    assert float(weights.sum()) == 1.0


def test_canonical_physical_component_counts_include_the_real_kernel() -> None:
    counts = producer._physical_component_counts(
        external_hidden_size=1024,
        source_internal_channels=1024,
        target_internal_channels=256,
        kernel_size=3,
        output_action_rank=128,
    )
    assert counts == {
        "source": 4_197_376,
        "qdeim": 1_049_344,
        "factored": 951_040,
    }
    assert counts["source"] == 4 * counts["qdeim"]


def test_output_action_rows_use_direct_cached_short_conv_o_not_endpoint_subtraction() -> (
    None
):
    row_ids = producer._phase_row_ids(
        batch_index=0,
        phase="prefill",
        row_count=2,
    )
    direct_action = torch.tensor(
        [[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
        dtype=torch.bfloat16,
    )
    phase = Lfm2PartialShortConvCachedPhase(
        case_id="batch.000000/prefill",
        phase="prefill",
        parent_pre_operator_residual=torch.zeros(1, 2, 3),
        parent_post_operator_residual=torch.ones(1, 2, 3),
        parent_short_conv_o=direct_action,
        parent_b_history=torch.ones(1, 2, 4),
        parent_c_history=torch.ones(1, 2, 4),
        parent_x_history=torch.ones(1, 2, 4),
        parent_context_from_fir=torch.ones(1, 2, 4),
        parent_pre_bx_state=torch.zeros(1, 4, 3),
        row_sample_weights=torch.full((1, 2), 0.5, dtype=torch.float64),
        row_ids=row_ids,
        parent_trajectory_receipt_sha256=_digest("trajectory"),
        cached_phase_receipt_sha256=_digest("phase"),
        aligned_row_identity_sha256=canonical_sha256(list(row_ids)),
    )
    observed_ids, action_rows, weights = producer._output_action_rows((phase,))
    assert observed_ids == row_ids
    torch.testing.assert_close(action_rows, direct_action.float().reshape(2, 3))
    assert not torch.equal(
        action_rows,
        (phase.parent_post_operator_residual - phase.parent_pre_operator_residual).reshape(
            2, 3
        ),
    )
    torch.testing.assert_close(weights, torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_four_candidate_publication_matches_runner_plan_and_archive_schema(
    tmp_path: Path,
) -> None:
    archive: dict[str, torch.Tensor] = {}
    source_elements = 8 * (4 * 8 + 3)
    projected = _candidate(
        archive,
        candidate_id="projected",
        runtime_kind="projected-parent-output",
        internal=8,
        source_elements=source_elements,
        state={"out_proj.weight": torch.randn(8, 8)},
        rank=2,
    )
    projected = dict(projected)
    projected["candidate_parameter_elements"] = source_elements
    projected["component_compression_ratio"] = 1.0
    body = dict(projected)
    body.pop("candidate_sha256")
    projected["candidate_sha256"] = canonical_sha256(body)
    structural = _candidate(
        archive,
        candidate_id="structural",
        runtime_kind="qdeim-structural",
        internal=2,
        source_elements=source_elements,
        state={
            "conv.weight": torch.randn(2, 1, 3),
            "in_proj.weight": torch.randn(6, 8),
            "out_proj.weight": torch.randn(8, 2),
        },
    )
    local = _candidate(
        archive,
        candidate_id="local",
        runtime_kind="qdeim-local-writer",
        internal=2,
        source_elements=source_elements,
        state={
            "conv.weight": torch.randn(2, 1, 3),
            "in_proj.weight": torch.randn(6, 8),
            "out_proj.weight": torch.randn(8, 2),
        },
    )
    factored = _candidate(
        archive,
        candidate_id="factored",
        runtime_kind="qdeim-local-writer-output-factor",
        internal=2,
        source_elements=source_elements,
        rank=1,
        state={
            "conv.weight": torch.randn(2, 1, 3),
            "in_proj.weight": torch.randn(6, 8),
            "out_decoder.weight": torch.randn(8, 1),
            "out_encoder.weight": torch.randn(1, 2),
        },
    )
    candidates = [projected, dict(structural), dict(local), dict(factored)]
    plan_body = {
        "schema": producer.PLAN_SCHEMA,
        "status": "test",
        "producer_schema": producer.PRODUCER_SCHEMA,
        "parent_checkpoint_inventory_sha256": _digest("inventory"),
        "parent_fp32_state_sha256": _digest("parent-state"),
        "parent_parameter_elements": 999,
        "parent_cache_binding": {"binding_sha256": _digest("binding")},
        "candidates": candidates,
        "candidate_ids": [row["candidate_id"] for row in candidates],
        "candidate_count": 4,
    }
    output = tmp_path / "publication"
    receipt = producer._publish(
        output=output,
        archive_tensors=archive,
        plan_body=plan_body,
    )
    plan_path = output / producer.PLAN_FILENAME
    archive_path = output / producer.ARCHIVE_FILENAME
    plan = json.loads(plan_path.read_text(encoding="ascii"))
    loaded_plan = runner._read_plan(
        plan_path,
        expected_plan_sha256=receipt["plan_sha256"],
        expected_file_sha256=receipt["plan_file_sha256"],
    )
    loaded_archive = runner._load_archive(
        archive_path,
        expected_file_sha256=receipt["archive_file_sha256"],
        records=loaded_plan["tensor_records"],
    )
    assert plan["schema"] == producer.PLAN_SCHEMA
    assert len(plan["candidates"]) == 4
    assert set(loaded_archive) == set(archive)
    assert plan["tensor_archive"]["tensor_count"] == len(archive)
    assert plan["candidates"][0]["candidate_parameter_elements"] == source_elements
    assert plan["candidates"][1]["candidate_parameter_elements"] * 4 == (
        source_elements
    )
    assert plan["candidates"][3]["candidate_parameter_elements"] < plan[
        "candidates"
    ][2]["candidate_parameter_elements"]


def test_producer_source_has_one_selective_session_and_no_model_execution() -> None:
    source_path = Path(producer.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    cache_opens = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "open_lfm2_circuit_trace_cache"
    ]
    assert len(cache_opens) == 1
    assert any(
        keyword.arg == "verify"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in cache_opens[0].keywords
    )
    selective_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "selective_authenticated_session"
    ]
    assert len(selective_calls) == 1
    forbidden_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in {"forward", "generate", "backward", "step"}
    }
    assert forbidden_attributes == set()
    action_seals = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "seal_lfm2_output_action_rows"
    ]
    assert len(action_seals) == 2
    for call in action_seals:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "branch_action_rows" in keywords
        assert "pre_rows" not in keywords
        assert "post_rows" not in keywords
    assert "layer.000.short_conv.o" in producer._PHASE_SUFFIXES
    assert producer.FIT_BATCH_INDICES == tuple(range(8))
    assert producer.HELDOUT_BATCH_INDICES == tuple(range(8, 11))
    assert producer.UNTOUCHED_BATCH_INDICES == tuple(range(11, 16))


def test_candidate_archives_are_independent_dense_fp32_storage() -> None:
    archive: dict[str, torch.Tensor] = {}
    source = torch.randn(4, 3, dtype=torch.float64)
    row = producer._candidate_row(
        candidate_id="independent",
        runtime_kind="qdeim-local-writer",
        external_hidden_size=4,
        internal_channels=3,
        kernel_size=1,
        source_component_parameter_elements=24,
        state={"out_proj.weight": source},
        archive=archive,
    )
    archived = archive[row["state_tensor_keys"]["out_proj.weight"]]
    assert archived.dtype == torch.float32
    assert archived.is_contiguous()
    assert archived.data_ptr() != source.data_ptr()
    source.zero_()
    assert torch.count_nonzero(archived).item() > 0
