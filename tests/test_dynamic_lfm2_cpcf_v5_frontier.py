from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_lfm2_cpcf_v5_frontier import (
    DEFAULT_COMPRESSION_RATIOS,
    Lfm2CpcfV5ApprovedReviewBindings,
    Lfm2CpcfV5FrontierError,
    Lfm2CpcfV5FrontierLifecycle,
    Lfm2CpcfV5ResumeReference,
    authenticate_lfm2_cpcf_v5_resume,
    authorize_lfm2_cpcf_v5_reviewed_rerun,
    plan_lfm2_cpcf_v5_frontier,
)
from neural_forge.dynamic_parameter_frontier import (
    DenseTargetTopologyCandidate,
)
from neural_forge.hf_token_align import (
    StrictTokenRowMapping,
    StrictTokenTransportProfile,
)
from neural_forge.hf_tokenizer_bridge import (
    CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY,
    PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY,
    STRICT_TOKENIZER_BRIDGE_SCHEMA,
    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
    TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY,
    StrictTokenizerBridgeArtifact,
    TokenizerArtifactFile,
)
from tests.test_dynamic_parameter_frontier import _lfm2_graph


REAL_PARENT_GRAPH_SHA256 = (
    "a6ea1b275bdf47aba906af4dd369e12c3960c9f8f039e18488856da9c1ea3155"
)
REAL_BRIDGE_ARTIFACT_SHA256 = (
    "5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384"
)
REAL_BRIDGE_PROFILE_SHA256 = (
    "e3fc98c666b70b367962e400435afe7588a9f1a2b13619dffc6f40c6458c5ca1"
)
REAL_DONOR_TOKENIZER_SEMANTICS_SHA256 = (
    "37f9fcf285f627896fe8fc868386c4d80a37362a9a54343c34483744eee7f564"
)
REAL_TARGET_TOKENIZER_SEMANTICS_SHA256 = (
    "cffb809c40ddd3382131a8ed8b1b8536457da4e46dfced85dff5669fb6d73342"
)

_EXPECTED_REAL_FRONTIER = {
    4: {
        "rank": 2,
        "parameters": 57_405_368,
        "candidate_fingerprint": (
            "bc74a6aba1e423b98545912acae55be24fcb301f3cc80538db0eaa815a58dc89"
        ),
        "hidden_size": 776,
        "intermediate_size": 797,
        "num_attention_heads": 10,
        "num_key_value_heads": 5,
        "head_dim": 64,
    },
    10: {
        "rank": 0,
        "parameters": 22_966_464,
        "candidate_fingerprint": (
            "faaa81d8c6e25360c0a5ef9bc9290111e0d9ccf3a130bf7a1eb2b0acd7af1017"
        ),
        "hidden_size": 192,
        "intermediate_size": 2548,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
    },
    25: {
        "rank": 0,
        "parameters": 9_183_872,
        "candidate_fingerprint": (
            "c6f2e1af6cc73b92332b379fd2d25f6f1100c5562e624e5a5b010aa3b2386f55"
        ),
        "hidden_size": 128,
        "intermediate_size": 1457,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
    },
    50: {
        "rank": 0,
        "parameters": 4_592_768,
        "candidate_fingerprint": (
            "72be613b37dc8d8b9ab6f142cf27af441da5dcc13006436ddbc6c20df2b2f18f"
        ),
        "hidden_size": 128,
        "intermediate_size": 603,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
    },
    100: {
        "rank": 0,
        "parameters": 2_291_840,
        "candidate_fingerprint": (
            "88bf2548e31924222a520b92d160031e41eef89c6cb8445d4284c39eee4a99e4"
        ),
        "hidden_size": 128,
        "intermediate_size": 175,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
    },
    250: {
        "rank": 0,
        "parameters": 916_928,
        "candidate_fingerprint": (
            "93a781ac1ecfa9905cebb38f7db24edd9ff093529053a36605b51f1f4eacdeda"
        ),
        "hidden_size": 64,
        "intermediate_size": 166,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
    },
}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sealed_bridge(root: Path) -> StrictTokenizerBridgeArtifact:
    root.mkdir()
    tokenizer_bytes = b'{"sealed_fixture":true}\n'
    tokenizer_path = root / "tokenizer.json"
    tokenizer_path.write_bytes(tokenizer_bytes)
    profile = StrictTokenTransportProfile(
        donor_tokenizer_class="SealedRealGeometryDonor",
        target_tokenizer_class="Sealed4096RowTarget",
        donor_vocab_size=65_536,
        target_vocab_size=4_096,
        donor_row_capacity=65_536,
        target_row_capacity=4_096,
        alignment_fingerprint=canonical_sha256(
            {"fixture": "strict-4096-row-transport"}
        ),
        row_mappings=tuple(
            StrictTokenRowMapping(
                target_row_index=index,
                donor_row_index=index,
                resolution="sealed-fixture-direct-row",
            )
            for index in range(4_096)
        ),
        alignment_resolution="sealed-fixture-direct-row",
    )
    donor_semantics = canonical_sha256({"fixture": "donor-semantics"})
    target_semantics = canonical_sha256({"fixture": "target-semantics"})
    file_record = TokenizerArtifactFile(
        relative_path="tokenizer.json",
        size_bytes=len(tokenizer_bytes),
        sha256=hashlib.sha256(tokenizer_bytes).hexdigest(),
    )
    body = {
        "schema": STRICT_TOKENIZER_BRIDGE_SCHEMA,
        "kind": "sealed-strict-target-tokenizer",
        "profile_fingerprint": profile.fingerprint(),
        "transport_profile": profile.to_dict(),
        "donor_tokenizer_semantics_sha256": donor_semantics,
        "target_tokenizer_semantics_sha256": target_semantics,
        "target_files": [file_record.to_dict()],
        "model_parameter_operations": {
            "training_steps": 0,
            "backward_calls": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
            "random_target_rows": 0,
        },
    }
    artifact = StrictTokenizerBridgeArtifact(
        root_dir=root,
        transport_profile=profile,
        donor_tokenizer_semantics_sha256=donor_semantics,
        target_tokenizer_semantics_sha256=target_semantics,
        target_files=(file_record,),
        artifact_sha256=_canonical_sha256(body),
    )
    artifact.verify_files()
    return artifact


def _real_geometry_graph(
    *,
    tokenizer_fingerprint: str,
):
    return replace(
        _lfm2_graph(
            vocab=65_536,
            hidden=1_024,
            intermediate=2_560,
            query_heads=16,
            key_value_heads=8,
            head_dim=64,
            layer_types=(
                "conv",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
            ),
        ),
        tokenizer_fingerprint=tokenizer_fingerprint,
    )


@pytest.fixture(scope="module")
def planned_frontier(tmp_path_factory):
    root = tmp_path_factory.mktemp("sealed-lfm2-v5-frontier")
    bridge = _sealed_bridge(root / "bridge")
    graph = _real_geometry_graph(
        tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256
    )
    assert graph.unique_parameter_count == 229_693_184
    schedule = plan_lfm2_cpcf_v5_frontier(
        parent_graph=graph,
        expected_parent_graph_fingerprint=graph.fingerprint,
        tokenizer_bridge=bridge,
        expected_bridge_artifact_sha256=bridge.artifact_sha256,
        maximum_candidates_per_ratio=8,
    )
    return graph, bridge, schedule


def _candidate_with_real_bridge_identity(
    candidate: DenseTargetTopologyCandidate,
) -> DenseTargetTopologyCandidate:
    configuration = dict(candidate.configuration)
    configuration.update(
        {
            TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY: (
                REAL_BRIDGE_ARTIFACT_SHA256
            ),
            TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY: REAL_BRIDGE_PROFILE_SHA256,
            PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY: (
                REAL_DONOR_TOKENIZER_SEMANTICS_SHA256
            ),
            CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY: (
                REAL_TARGET_TOKENIZER_SEMANTICS_SHA256
            ),
        }
    )
    identity = canonical_sha256(
        {
            "catalog_id": candidate.catalog_id,
            "configuration": configuration,
            "unique_parameters": candidate.unique_parameters,
            "logical_parameters": candidate.logical_parameters,
        }
    )
    return replace(
        candidate,
        candidate_id=f"lfm2-dense-{identity[:16]}",
        configuration=tuple(configuration.items()),
    )


def _approved(plan_fingerprint: str) -> Lfm2CpcfV5ApprovedReviewBindings:
    return Lfm2CpcfV5ApprovedReviewBindings(
        target_plan_fingerprint=plan_fingerprint,
        approved_candidate_state_sha256=canonical_sha256(
            {"kind": "candidate-state", "plan": plan_fingerprint}
        ),
        approved_candidate_outputs_sha256=canonical_sha256(
            {"kind": "candidate-outputs", "plan": plan_fingerprint}
        ),
        approved_manual_review_sha256=canonical_sha256(
            {"kind": "manual-review", "plan": plan_fingerprint}
        ),
    )


def _resume(plan_fingerprint: str) -> Lfm2CpcfV5ResumeReference:
    return Lfm2CpcfV5ResumeReference(
        target_plan_fingerprint=plan_fingerprint,
        resume_admission_sha256=canonical_sha256(
            {"kind": "resume-admission", "plan": plan_fingerprint}
        ),
        resume_artifact_sha256=canonical_sha256(
            {"kind": "resume-artifact", "plan": plan_fingerprint}
        ),
    )


def test_real_geometry_frontier_reproduces_all_six_automatic_targets(
    planned_frontier,
) -> None:
    _graph, _bridge, schedule = planned_frontier

    assert len(schedule.lanes) == 6
    assert DEFAULT_COMPRESSION_RATIOS == (4, 10, 25, 50, 100, 250)
    for ratio, lane in zip(
        DEFAULT_COMPRESSION_RATIOS,
        schedule.lanes,
        strict=True,
    ):
        expected = _EXPECTED_REAL_FRONTIER[ratio]
        configuration = dict(lane.selected_candidate.configuration)

        assert lane.selected_candidate_rank == expected["rank"]
        assert (
            lane.exact_compression.target_unique_parameters
            == expected["parameters"]
        )
        assert lane.target_plan.target_request.exact_unique_parameters == (
            expected["parameters"]
        )
        assert lane.exact_compression.requested_ratio == ratio
        assert lane.exact_compression.realized_ratio >= ratio
        assert lane.target_plan.target_certificate.selection_policy == (
            "supported_pareto_balanced_executable_v1"
        )
        assert lane.selected_candidate.compiler_executable is True
        assert lane.selected_candidate.missing_compiler_capabilities == ()
        assert configuration["vocab_size"] == 4_096
        assert configuration["num_hidden_layers"] == 14
        assert configuration["hidden_size"] == expected["hidden_size"]
        assert (
            configuration["intermediate_size"]
            == expected["intermediate_size"]
        )
        assert (
            configuration["num_attention_heads"]
            == expected["num_attention_heads"]
        )
        assert (
            configuration["num_key_value_heads"]
            == expected["num_key_value_heads"]
        )
        assert configuration["head_dim"] == expected["head_dim"]
        assert _candidate_with_real_bridge_identity(
            lane.selected_candidate
        ).fingerprint == expected["candidate_fingerprint"]

    # The 4x point is intentionally the automatic balanced executable rank 2,
    # not a caller-forced rank zero.
    assert schedule.lanes[0].selected_candidate_rank == 2
    assert schedule.as_dict()["caller_forced_candidate_rank"] is False


def test_real_production_graph_and_bridge_pins_are_explicit_contracts(
    planned_frontier,
) -> None:
    graph, bridge, _schedule = planned_frontier
    pins = (
        REAL_PARENT_GRAPH_SHA256,
        REAL_BRIDGE_ARTIFACT_SHA256,
        REAL_BRIDGE_PROFILE_SHA256,
        REAL_DONOR_TOKENIZER_SEMANTICS_SHA256,
        REAL_TARGET_TOKENIZER_SEMANTICS_SHA256,
    )
    assert all(len(value) == 64 for value in pins)
    assert len(set(pins)) == len(pins)
    assert bridge.transport_profile.target_row_capacity == 4_096
    assert bridge.transport_profile.donor_row_capacity == 65_536

    with pytest.raises(
        Lfm2CpcfV5FrontierError,
        match="Parent graph differs",
    ):
        plan_lfm2_cpcf_v5_frontier(
            parent_graph=graph,
            expected_parent_graph_fingerprint=REAL_PARENT_GRAPH_SHA256,
            tokenizer_bridge=bridge,
            expected_bridge_artifact_sha256=bridge.artifact_sha256,
            compression_ratios=(4,),
        )
    with pytest.raises(
        Lfm2CpcfV5FrontierError,
        match="independent artifact pin",
    ):
        plan_lfm2_cpcf_v5_frontier(
            parent_graph=graph,
            expected_parent_graph_fingerprint=graph.fingerprint,
            tokenizer_bridge=bridge,
            expected_bridge_artifact_sha256=REAL_BRIDGE_ARTIFACT_SHA256,
            compression_ratios=(4,),
        )


def test_capture_lifecycle_is_non_promotable_and_needs_no_review(
    planned_frontier,
) -> None:
    _graph, _bridge, schedule = planned_frontier

    assert (
        schedule.lifecycle
        is Lfm2CpcfV5FrontierLifecycle.CAPTURE_QUARANTINE
    )
    assert all(
        lane.lifecycle_contract.approved_review_bindings is None
        and lane.promotion_permitted_after_behavior_gate is False
        for lane in schedule.lanes
    )
    report = schedule.as_dict()
    assert report["training_steps"] == 0
    assert report["backward_calls"] == 0
    assert report["parameters_pruned"] == 0
    assert report["parameters_quantized"] == 0
    assert report["remote_execution_performed"] is False


def test_reviewed_rerun_requires_all_three_approved_plan_bindings(
    planned_frontier,
) -> None:
    _graph, _bridge, capture = planned_frontier
    approvals = {
        lane.target_plan_fingerprint: _approved(
            lane.target_plan_fingerprint
        )
        for lane in capture.lanes
    }

    reviewed = authorize_lfm2_cpcf_v5_reviewed_rerun(
        capture,
        approvals,
    )

    assert (
        reviewed.lifecycle
        is Lfm2CpcfV5FrontierLifecycle.REVIEWED_RERUN
    )
    assert reviewed.fingerprint != capture.fingerprint
    assert tuple(lane.target_plan for lane in reviewed.lanes) == tuple(
        lane.target_plan for lane in capture.lanes
    )
    assert all(
        lane.promotion_permitted_after_behavior_gate
        and lane.lifecycle_contract.approved_review_bindings
        == approvals[lane.target_plan_fingerprint]
        for lane in reviewed.lanes
    )

    missing = dict(approvals)
    missing.pop(next(iter(missing)))
    with pytest.raises(
        Lfm2CpcfV5FrontierError,
        match="every lane",
    ):
        authorize_lfm2_cpcf_v5_reviewed_rerun(capture, missing)


def test_capture_rejects_review_bindings_and_reviewed_planning_requires_all(
    planned_frontier,
) -> None:
    graph, bridge, capture = planned_frontier
    first = capture.lanes[0]
    one_approval = {
        first.target_plan_fingerprint: _approved(
            first.target_plan_fingerprint
        )
    }

    with pytest.raises(
        Lfm2CpcfV5FrontierError,
        match="accepts no reviewed",
    ):
        plan_lfm2_cpcf_v5_frontier(
            parent_graph=graph,
            expected_parent_graph_fingerprint=graph.fingerprint,
            tokenizer_bridge=bridge,
            expected_bridge_artifact_sha256=bridge.artifact_sha256,
            compression_ratios=(4,),
            approved_review_bindings=one_approval,
        )
    with pytest.raises(
        Lfm2CpcfV5FrontierError,
        match="every target plan",
    ):
        plan_lfm2_cpcf_v5_frontier(
            parent_graph=graph,
            expected_parent_graph_fingerprint=graph.fingerprint,
            tokenizer_bridge=bridge,
            expected_bridge_artifact_sha256=bridge.artifact_sha256,
            compression_ratios=(4,),
            lifecycle=Lfm2CpcfV5FrontierLifecycle.REVIEWED_RERUN,
        )


def test_resume_and_all_artifact_namespaces_are_width_isolated(
    planned_frontier,
) -> None:
    _graph, _bridge, schedule = planned_frontier
    first, second = schedule.lanes[:2]
    first_resume = _resume(first.target_plan_fingerprint)

    assert authenticate_lfm2_cpcf_v5_resume(first, first_resume) is (
        first_resume
    )
    with pytest.raises(
        Lfm2CpcfV5FrontierError,
        match="Cross-width resume",
    ):
        authenticate_lfm2_cpcf_v5_resume(second, first_resume)

    per_lane_namespaces = {
        value
        for lane in schedule.lanes
        for value in (
            lane.identity.candidate_namespace_sha256,
            lane.identity.resume_namespace_sha256,
            lane.identity.quarantine_namespace_sha256,
        )
    }
    assert len(per_lane_namespaces) == len(schedule.lanes) * 3
    resumed_first = replace(first, resume_reference=first_resume)
    assert resumed_first.resume_reference == first_resume
    with pytest.raises(
        Lfm2CpcfV5FrontierError,
        match="Cross-width resume",
    ):
        replace(second, resume_reference=first_resume)


def test_schedule_is_typed_bounded_and_does_not_execute(
    planned_frontier,
) -> None:
    _graph, _bridge, schedule = planned_frontier

    assert schedule.maximum_parallel_lanes == 1
    assert [len(wave.target_plan_fingerprints) for wave in schedule.waves] == [
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert tuple(
        plan
        for wave in schedule.waves
        for plan in wave.target_plan_fingerprints
    ) == tuple(lane.target_plan_fingerprint for lane in schedule.lanes)
    assert all(
        lane.target_plan.target_certificate.selected_candidate_fingerprint
        == lane.selected_candidate.fingerprint
        for lane in schedule.lanes
    )
    assert schedule.lane_for_plan_fingerprint(
        schedule.lanes[-1].target_plan_fingerprint
    ) is schedule.lanes[-1]
    report = schedule.as_dict()
    assert report["planning_only"] is True
    assert report["parent_or_child_model_loaded"] is False
    assert report["automatic_candidate_selection"] is True
    assert report["lane_internal_gpu_count"] == 2
    assert (
        report[
            "lane_internal_gpu_parallelism_is_separate_from_cross_lane_parallelism"
        ]
        is True
    )
    assert report["cross_lane_parallelism_requires_disjoint_gpu_pairs"] is True
    assert report["gpu_device_assignment_performed"] is False


def test_ratio_validation_has_no_severe_only_threshold(
    planned_frontier,
) -> None:
    graph, bridge, _schedule = planned_frontier

    with pytest.raises(
        Lfm2CpcfV5FrontierError,
        match="Duplicate compression ratios",
    ):
        plan_lfm2_cpcf_v5_frontier(
            parent_graph=graph,
            expected_parent_graph_fingerprint=graph.fingerprint,
            tokenizer_bridge=bridge,
            expected_bridge_artifact_sha256=bridge.artifact_sha256,
            compression_ratios=(4, 4),
        )
    with pytest.raises(
        Lfm2CpcfV5FrontierError,
        match="must be Lfm2CpcfV5FrontierLifecycle",
    ):
        plan_lfm2_cpcf_v5_frontier(
            parent_graph=graph,
            expected_parent_graph_fingerprint=graph.fingerprint,
            tokenizer_bridge=bridge,
            expected_bridge_artifact_sha256=bridge.artifact_sha256,
            compression_ratios=(4,),
            lifecycle="capture_quarantine",  # type: ignore[arg-type]
        )

    # A valid 4x lane in the shared fixture proves planning is not gated at
    # 100x or any other hardcoded severe-compaction threshold.
    assert _schedule.lanes[0].exact_compression.requested_ratio == 4
    assert _schedule.lanes[0].exact_compression.as_dict()[
        "ratio_admission_threshold_used"
    ] is False
