from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from experiments import (  # noqa: E402
    probe_lfm2_cpcf_v5_real_attention_layer as runner,
)
from neural_forge import (  # noqa: E402
    hf_lfm2_sequence_effective_attention_fold as sequence_fold,
)
from neural_forge.hf_lfm2_cpcf_v5_real_attention_probe import (  # noqa: E402
    H776_ATTENTION_CANDIDATE_IDS,
    H776_FULL_POLICY_SHA256,
    H776_ORIGIN_COMPILER_SHA256,
    H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
    H776_SEQUENCE_SOLVER_PLAN_SHA256,
    H776_SPLIT_PHASE_COUNTS,
    H776_TARGET_TOPOLOGY_SHA256,
    Lfm2CpcfV5RealAttentionProbeError,
    Lfm2CpcfV5RealAttentionProbePins,
    Lfm2CpcfV5RealAttentionResourceMonitor,
    Lfm2SequenceEffectivePcgCollector,
    _sha256,
    authenticate_h776_attention_result,
    authenticate_h776_boundary_frontier,
    authenticate_h776_k2_checkpoint,
    authenticate_h776_phase_inventory,
)


_DIGEST = "1" * 64
_OTHER_DIGEST = "2" * 64


def _pins(root: Path) -> Lfm2CpcfV5RealAttentionProbePins:
    return Lfm2CpcfV5RealAttentionProbePins(
        checkpoint_store_directory=root,
        checkpoint_key=_DIGEST,
        checkpoint_manifest_sha256=_DIGEST,
        expected_trace_sha256=_DIGEST,
        expected_runtime_compiler_implementation_sha256=_OTHER_DIGEST,
        expected_sequence_source_sha256=_DIGEST,
    )


def test_probe_pins_are_exact_and_cross_runtime(tmp_path: Path) -> None:
    pins = _pins(tmp_path)
    assert pins.layer_index == 2
    assert pins.maximum_process_rss_bytes == 64 * 1024**3
    assert pins.maximum_cuda_bytes_per_device == 20 * 1024**3
    assert (
        pins.expected_runtime_compiler_implementation_sha256
        != pins.expected_origin_compiler_sha256
    )
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        replace(pins, layer_index=1)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        replace(
            pins,
            expected_runtime_compiler_implementation_sha256=(
                H776_ORIGIN_COMPILER_SHA256
            ),
        )


class _Reference:
    def __init__(self, pins: Lfm2CpcfV5RealAttentionProbePins) -> None:
        self._pins = pins

    def as_dict(self) -> dict[str, str]:
        return {
            "artifact_kind": "checkpoint",
            "artifact_key": self._pins.checkpoint_key,
            "manifest_sha256": self._pins.checkpoint_manifest_sha256,
        }


class _Store:
    def final_reauthenticate_checkpoint(self, verified):
        return verified


def _checkpoint(
    pins: Lfm2CpcfV5RealAttentionProbePins,
    *,
    next_layer_index: int = 2,
):
    first_bank = "3" * 64
    second_bank = "4" * 64
    execution = {
        "closed_form_compilers": True,
        "parent_model_forward_calls": 0,
        "target_model_forward_calls": 0,
        "generation_calls": 0,
        "training_steps": 0,
    }
    receipts = [
        {
            "layer_index": 0,
            "operator_kind": "short_conv",
            "input_bank_sha256": "5" * 64,
            "next_bank_sha256": first_bank,
            "execution": dict(execution),
        },
        {
            "layer_index": 1,
            "operator_kind": "short_conv",
            "input_bank_sha256": first_bank,
            "next_bank_sha256": second_bank,
            "execution": dict(execution),
        },
    ]
    bindings = {
        "next_layer_index": next_layer_index,
        "target_topology_sha256": H776_TARGET_TOPOLOGY_SHA256,
        "full_policy_sha256": H776_FULL_POLICY_SHA256,
        "compiler_fingerprint_sha256": H776_ORIGIN_COMPILER_SHA256,
        "boundary": {
            "artifact_kind": "boundary",
            "artifact_key": "boundary-k2",
            "manifest_sha256": "6" * 64,
        },
    }
    completed = {
        "next_layer_index": next_layer_index,
        "incoming_state_layer_index": next_layer_index,
        "completed_layer_indices": [0, 1],
        "completed_layer_receipts": receipts,
        "incoming_state_bank_metadata": {
            "layer_index": next_layer_index,
            "boundary": "layer.001.post_layer",
            "actual_folded_upstream_states": True,
            "state_bank_sha256": second_bank,
        },
        "target_topology_snapshot": {
            "model_type": "lfm2",
            "layer_count": 14,
            "topology_sha256": H776_TARGET_TOPOLOGY_SHA256,
        },
        "target_topology_sha256": H776_TARGET_TOPOLOGY_SHA256,
    }
    return SimpleNamespace(
        reference=_Reference(pins),
        manifest={"bindings": bindings},
        completed_layers=completed,
    )


def test_k2_checkpoint_accepts_only_contiguous_two_layer_prefix(
    tmp_path: Path,
) -> None:
    pins = _pins(tmp_path)
    audit = authenticate_h776_k2_checkpoint(
        _Store(),
        _checkpoint(pins),
        pins,
    )
    assert audit["next_layer_index"] == 2
    assert audit["completed_layer_indices"] == [0, 1]
    assert audit["k1_rejected_as_insufficient"] is True
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_k2_checkpoint(
            _Store(),
            _checkpoint(pins, next_layer_index=1),
            pins,
        )


def _frontier(pins: Lfm2CpcfV5RealAttentionProbePins):
    context = {
        "source_model_state_sha256": pins.expected_source_state_sha256,
        "source_model_inventory_sha256": pins.expected_source_inventory_sha256,
        "cache_sha256": pins.expected_cache_sha256,
        "selective_catalog_sha256": pins.expected_selective_catalog_sha256,
        "selective_source_sha256": pins.expected_selective_source_sha256,
        "selective_provider_sha256": pins.expected_selective_provider_sha256,
        "trace_sha256": pins.expected_trace_sha256,
        "tokenizer_sha256": pins.expected_tokenizer_sha256,
        "token_bridge_sha256": pins.expected_token_bridge_sha256,
        "source_tied_table_sha256": pins.expected_source_tied_table_sha256,
        "blank_target_report_sha256": pins.expected_blank_report_sha256,
        "blank_target_state_sha256": pins.expected_blank_state_sha256,
        "target_topology_sha256": pins.expected_target_topology_sha256,
        "planned_target_unique_parameters": 57_405_368,
        "runtime_policy_sha256": pins.expected_full_policy_sha256,
        "compiler_implementation_sha256": pins.expected_origin_compiler_sha256,
        "context_sha256": "9" * 64,
    }
    return SimpleNamespace(
        report={
            "frontier_sha256": "a" * 64,
            "context": context,
            "next_layer_index": 2,
            "expected_layer_count": 14,
            "source_width": 1_024,
            "target_width": 776,
            "ordered_phase_sha256": ["b" * 64] * 2_304,
            "split_row_counts": {
                "fit": 40_656,
                "selection": 1,
                "validation": 13_199,
            },
            "layer_operator_kinds": [
                "short_conv",
                "short_conv",
                "full_attention",
                "short_conv",
                "full_attention",
                "short_conv",
                "full_attention",
                "short_conv",
                "full_attention",
                "short_conv",
                "full_attention",
                "short_conv",
                "full_attention",
                "short_conv",
            ],
            "completed_layer_charts": [
                {"operator_kind": "short_conv"},
                {"operator_kind": "short_conv"},
            ],
        }
    )


def test_boundary_frontier_preserves_origin_policy_and_compiler(
    tmp_path: Path,
) -> None:
    pins = _pins(tmp_path)
    audit = authenticate_h776_boundary_frontier(_frontier(pins), pins)
    assert audit["next_layer_index"] == 2
    assert audit["ordinary_resume_gate_weakened"] is False
    changed = _frontier(pins)
    changed.report["context"]["compiler_implementation_sha256"] = (
        pins.expected_runtime_compiler_implementation_sha256
    )
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_boundary_frontier(changed, pins)


def _complete_phases():
    names = (
        "prefill",
        "decode.000",
        "decode.001",
        "decode.002",
        "decode.003",
        "decode.004",
        "decode.005",
        "decode.006",
        "decode.007",
    )
    split_trajectory_counts = {"fit": 154, "selection": 52, "validation": 50}
    phases = []
    ordinal = 0
    for split, count in split_trajectory_counts.items():
        for _ in range(count):
            for name in names:
                phases.append(
                    SimpleNamespace(
                        batch_index=ordinal,
                        record_id=f"record-{ordinal}",
                        split=split,
                        phase=name,
                        batch_size=1,
                        sampled_suffix_tokens=1,
                    )
                )
            ordinal += 1
    fit_extra = 40_656 - H776_SPLIT_PHASE_COUNTS["fit"]
    validation_extra = 13_199 - H776_SPLIT_PHASE_COUNTS["validation"]
    phases[0].sampled_suffix_tokens += fit_extra
    validation_start = (
        H776_SPLIT_PHASE_COUNTS["fit"]
        + H776_SPLIT_PHASE_COUNTS["selection"]
    )
    phases[validation_start].sampled_suffix_tokens += validation_extra
    rows = {
        split: sum(
            phase.batch_size * phase.sampled_suffix_tokens
            for phase in phases
            if phase.split == split
        )
        for split in split_trajectory_counts
    }
    frontier = SimpleNamespace(report={"split_row_counts": rows})
    return phases, frontier


def test_phase_inventory_requires_all_trajectories_and_decode_steps() -> None:
    phases, frontier = _complete_phases()
    audit = authenticate_h776_phase_inventory(phases, frontier=frontier)
    assert audit["phase_count"] == 2_304
    assert audit["trajectory_count"] == 256
    assert audit["complete_prefill_plus_eight_decode_phases"] is True
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_phase_inventory(phases[:-1], frontier=frontier)


def _pcg_event(
    role: str = "sequence_effective_structured_value_iteration_1_group_0",
):
    events = []
    right = torch.tensor([1.0, 2.0], dtype=torch.float64)
    diagonal = torch.tensor([2.0, 3.0], dtype=torch.float64)

    def matvec(value):
        return diagonal * value

    with sequence_fold.observe_lfm2_sequence_effective_pcg(events.append):
        observed_solution, observed_report = sequence_fold._pcg_single_tensor(
            right=right,
            matvec=matvec,
            diagonal=diagonal,
            initial=torch.zeros_like(right),
            role=role,
        )
    plain_solution, plain_report = sequence_fold._pcg_single_tensor(
        right=right,
        matvec=matvec,
        diagonal=diagonal,
        initial=torch.zeros_like(right),
        role=role,
    )
    assert torch.equal(observed_solution, plain_solution)
    assert observed_report == plain_report
    assert len(events) == 1
    return events[0]


def test_scoped_pcg_observation_does_not_change_solver() -> None:
    event = _pcg_event()
    assert event["solver_result_or_arguments_modified"] is False
    right = torch.tensor([1.0], dtype=torch.float64)

    def mutate_detached_event(observed):
        observed["report"]["role"] = "observer-mutated-copy"

    with sequence_fold.observe_lfm2_sequence_effective_pcg(
        mutate_detached_event
    ):
        _solution, returned = sequence_fold._pcg_single_tensor(
            right=right,
            matvec=lambda value: 2.0 * value,
            diagonal=torch.tensor([2.0], dtype=torch.float64),
            initial=torch.zeros_like(right),
            role="returned-report-stays-authentic",
        )
    assert returned["role"] == "returned-report-stays-authentic"
    with sequence_fold.observe_lfm2_sequence_effective_pcg(lambda _event: None):
        with pytest.raises(sequence_fold.Lfm2SequenceEffectiveAttentionFoldError):
            with sequence_fold.observe_lfm2_sequence_effective_pcg(
                lambda _event: None
            ):
                pass


def test_pcg_collector_accepts_exact_five_group_sweeps() -> None:
    artifact = _sequence_artifact()
    collector = Lfm2SequenceEffectivePcgCollector()
    for group in range(5):
        collector(
            _pcg_event(
                "sequence_effective_structured_value_iteration_1_"
                f"group_{group}"
            )
        )
    audit = collector.finalize(sequence_artifact=artifact)
    assert audit["single_tensor_solve_count"] == 5
    assert audit["multi_rhs_solve_count"] == 0
    assert audit["expected_roles"] == audit["observed_roles"]
    duplicate = Lfm2SequenceEffectivePcgCollector()
    event = _pcg_event()
    duplicate(event)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        duplicate(event)
    incomplete = Lfm2SequenceEffectivePcgCollector()
    for group in range(4):
        incomplete(
            _pcg_event(
                "sequence_effective_structured_value_iteration_1_"
                f"group_{group}"
            )
        )
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        incomplete.finalize(sequence_artifact=artifact)


def test_resource_monitor_enforces_rss_cuda_and_zero_swap() -> None:
    def process():
        return {"VmRSS": 10, "VmHWM": 20, "VmSwap": 0}

    monitor = Lfm2CpcfV5RealAttentionResourceMonitor(
        process_sampler=process,
        cuda_sampler=lambda: {"cuda:0": 30, "cuda:1": 40},
        sample_interval_seconds=0.01,
    )
    with monitor:
        monitor.checkpoint("test")
    report = monitor.report()
    assert report["zero_swap_passed"] is True
    assert report["peak_cuda_reserved_bytes"] == {"cuda:0": 30, "cuda:1": 40}
    failing = Lfm2CpcfV5RealAttentionResourceMonitor(
        process_sampler=lambda: {"VmRSS": 10, "VmHWM": 20, "VmSwap": 1},
        cuda_sampler=lambda: {},
    )
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        failing.__enter__()


def _sequence_artifact(*, iterations: int = 1):
    effective_hash = _sha256({"solve": "effective"})
    seed_hash = _sha256({"solve": "seed"})
    anchor_hash = _sha256({"solve": "anchor"})
    frontier = [
        {
            "proposal_index": 0,
            "stage": "scalable_producer_shared_value_sequence_output_seed",
            "accepted": True,
            "solve_report_sha256": seed_hash,
        }
    ]
    solve_hashes = [effective_hash, seed_hash]
    accepted = 0
    for iteration in range(1, iterations + 1):
        value_hashes = [
            _sha256(
                {
                    "solve": "structured_value",
                    "iteration": iteration,
                    "group": group,
                }
            )
            for group in range(5)
        ]
        output_hash = _sha256(
            {"solve": "iteration_output", "iteration": iteration}
        )
        proposal_accepted = iteration < iterations
        accepted += int(proposal_accepted)
        frontier.append(
            {
                "proposal_index": iteration,
                "stage": (
                    "monotone_streamed_effective_space_value_output_als"
                ),
                "accepted": proposal_accepted,
                "value_solve_report_sha256": value_hashes,
                "output_solve_report_sha256": output_hash,
            }
        )
        solve_hashes.extend(value_hashes)
        solve_hashes.append(output_hash)
    frontier.append(
        {
            "proposal_index": len(frontier),
            "stage": "fixed_producer_value_subspace_anchor",
            "accepted": True,
            "selected_for_materialization": True,
            "output_solve_report_sha256": anchor_hash,
        }
    )
    solve_hashes.append(anchor_hash)
    body = {
        "candidate_id": H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
        "fixed_query_key_state_reused_exactly": True,
        "fit_only_compilation": True,
        "compiled_from_splits": ["fit"],
        "selection_case_count_consumed": 0,
        "validation_case_count_consumed": 0,
        "dense_target_native_qkvo": True,
        "ordinary_dense_target_value_output": True,
        "forbidden_value_coefficient_tensor_materialized": False,
        "forbidden_hidden_head_dim_gram_materialized": False,
        "solver_plan": {
            "strategy": "streamed_cross_head_statistics_pcg",
            "report_sha256": H776_SEQUENCE_SOLVER_PLAN_SHA256,
            "fit_rows": 40_656,
            "target_query_heads": 10,
            "target_hidden_size": 776,
            "target_output_size": 776,
            "head_dim": 64,
            "forbidden_value_coefficient_materialized": False,
            "forbidden_value_dense_gram_materialized": False,
            "complete_operator_dense_factorization_avoided": True,
        },
        "effective_operator_solve": {"report_sha256": effective_hash},
        "candidate_frontier": frontier,
        "candidate_frontier_sha256": _sha256(frontier),
        "bounded_proposal_count": len(frontier),
        "selected_proposal_index": len(frontier) - 1,
        "accepted_als_iterations": accepted,
        "executed_als_iteration_limit": 2,
        "solve_report_sha256": solve_hashes,
        "execution": {
            "parent_model_forward_calls": 0,
            "target_model_forward_calls": 0,
            "target_layer_forward_calls": 0,
            "generation_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "autograd_graphs": 0,
        },
    }
    return {**body, "report_sha256": _sha256(body)}


def _pcg_audit(artifact):
    collector = Lfm2SequenceEffectivePcgCollector()
    iterations = len(artifact["candidate_frontier"]) - 2
    for iteration in range(1, iterations + 1):
        for group in range(5):
            collector(
                _pcg_event(
                    "sequence_effective_structured_value_iteration_"
                    f"{iteration}_group_{group}"
                )
            )
    return collector.finalize(sequence_artifact=artifact)


def test_pcg_audit_accepts_exact_two_iteration_inventory() -> None:
    artifact = _sequence_artifact(iterations=2)
    audit = _pcg_audit(artifact)
    assert audit["als_iteration_count"] == 2
    assert audit["single_tensor_solve_count"] == 10
    assert audit["observed_roles"] == [
        (
            f"sequence_effective_structured_value_iteration_{iteration}_"
            f"group_{group}"
        )
        for iteration in (1, 2)
        for group in range(5)
    ]
    assert len(audit["event_to_artifact_solve_bindings"]) == 10


def _attention_result(*, selected: str, sequence_tied: bool = False):
    evaluations = {}
    for ordinal, candidate_id in enumerate(H776_ATTENTION_CANDIDATE_IDS):
        artifact = {"candidate_id": candidate_id}
        if candidate_id == H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID:
            artifact = _sequence_artifact()
        metric = float(ordinal + 2)
        if candidate_id == H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID:
            metric = 2.0 if sequence_tied else 1.0
        evaluations[candidate_id] = {
            "candidate_id": candidate_id,
            "operator_artifact": artifact,
            "operator_compiled_from_splits": ["fit"],
            "mlp_compiled_from_splits": ["fit"],
            "candidate_evaluated_on_splits": ["selection"],
            "validation_phase_count_consumed": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "selection_key": [metric] * 10 + [candidate_id],
        }
    if sequence_tied:
        evaluations[H776_ATTENTION_CANDIDATE_IDS[0]]["selection_key"] = (
            [2.0] * 10 + [H776_ATTENTION_CANDIDATE_IDS[0]]
        )
    elif selected == H776_ATTENTION_CANDIDATE_IDS[0]:
        evaluations[selected]["selection_key"] = (
            [0.5] * 10 + [selected]
        )
    selection = {
        "candidate_ids": list(H776_ATTENTION_CANDIDATE_IDS),
        "selected_candidate_id": selected,
        "candidate_count": 3,
        "selection_split_only": True,
        "winner_frozen_before_validation": True,
        "validation_phase_count_consumed_before_freeze": 0,
    }
    next_sha = "7" * 64
    body = {
        "layer_index": 2,
        "operator_kind": "full_attention",
        "input_bank_sha256": "8" * 64,
        "next_bank_sha256": next_sha,
        "split_phase_counts": dict(H776_SPLIT_PHASE_COUNTS),
        "split_metrics": {"validation": {"relative_rmse": 0.5}},
        "solver_evidence_policy": {
            "attention_candidate_selection": selection,
            "attention_candidate_evaluations": evaluations,
            "attention_each_candidate_has_own_fit_only_mlp": True,
            "attention_winner_frozen_before_validation": True,
            "attention_validation_replays_before_winner_freeze": 0,
        },
    }
    report = {**body, "result_sha256": _sha256(body)}
    return SimpleNamespace(
        report=report,
        next_bank=SimpleNamespace(layer_index=3, state_bank_sha256=next_sha),
    )


def test_attention_result_reports_positive_or_honest_negative(
    tmp_path: Path,
) -> None:
    pins = _pins(tmp_path)
    positive_result = _attention_result(
        selected=H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID
    )
    positive_artifact = positive_result.report["solver_evidence_policy"][
        "attention_candidate_evaluations"
    ][H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]["operator_artifact"]
    positive = authenticate_h776_attention_result(
        positive_result,
        input_bank_sha256="8" * 64,
        pcg_audit=_pcg_audit(positive_artifact),
        pins=pins,
    )
    assert positive["scientific_outcome"] == (
        "positive_sequence_effective_selection"
    )
    negative_result = _attention_result(
        selected=H776_ATTENTION_CANDIDATE_IDS[0]
    )
    negative_artifact = negative_result.report["solver_evidence_policy"][
        "attention_candidate_evaluations"
    ][H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]["operator_artifact"]
    negative = authenticate_h776_attention_result(
        negative_result,
        input_bank_sha256="8" * 64,
        pcg_audit=_pcg_audit(negative_artifact),
        pins=pins,
    )
    assert negative["scientific_outcome"] == (
        "honest_negative_or_control_selected"
    )


def test_attention_behavioral_tie_is_not_positive(tmp_path: Path) -> None:
    pins = _pins(tmp_path)
    tied_result = _attention_result(
        selected=H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
        sequence_tied=True,
    )
    artifact = tied_result.report["solver_evidence_policy"][
        "attention_candidate_evaluations"
    ][H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]["operator_artifact"]
    audit = authenticate_h776_attention_result(
        tied_result,
        input_bank_sha256="8" * 64,
        pcg_audit=_pcg_audit(artifact),
        pins=pins,
    )
    assert audit["sequence_effective_won"] is True
    assert (
        audit["sequence_effective_strictly_better_than_both_controls"]
        is False
    )
    assert audit["scientific_outcome"] == (
        "honest_negative_or_control_selected"
    )


def test_attention_rejects_resealed_pcg_cross_binding_tamper(
    tmp_path: Path,
) -> None:
    pins = _pins(tmp_path)
    result = _attention_result(
        selected=H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID
    )
    artifact = result.report["solver_evidence_policy"][
        "attention_candidate_evaluations"
    ][H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]["operator_artifact"]
    tampered = runner._plain(dict(_pcg_audit(artifact)))
    tampered.pop("audit_sha256")
    tampered["event_to_artifact_solve_bindings"][0][
        "artifact_value_solve_report_sha256"
    ] = "f" * 64
    tampered["audit_sha256"] = _sha256(tampered)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_attention_result(
            result,
            input_bank_sha256="8" * 64,
            pcg_audit=tampered,
            pins=pins,
        )


def _parser_args(tmp_path: Path) -> list[str]:
    paths = {}
    for name in (
        "parent",
        "donor",
        "target",
        "bridge",
        "cache",
        "corpus",
        "output",
        "resume",
        "catalog",
        "catalog_receipt",
    ):
        path = tmp_path / name
        path.mkdir(parents=True)
        paths[name] = path
    corpus = paths["corpus"] / "trajectories.json"
    corpus.write_text("[]", encoding="utf-8")
    receipt = paths["catalog_receipt"] / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    return [
        "--parent-model-directory",
        str(paths["parent"]),
        "--donor-tokenizer-directory",
        str(paths["donor"]),
        "--target-tokenizer-directory",
        str(paths["target"]),
        "--bridge-directory",
        str(paths["bridge"]),
        "--trace-cache-directory",
        str(paths["cache"]),
        "--trajectory-corpus",
        str(corpus),
        "--diagnostic-json",
        str(paths["output"] / "probe.json"),
        "--target-unique-parameters",
        "57405368",
        "--maximum-process-rss-gib",
        "64",
        "--expected-target-plan-fingerprint",
        "3f27369567214c7f01d3f437e9c4ec50bce05c8345bbee4bf23570c204be6153",
        "--resume-store-directory",
        str(paths["resume"]),
        "--selective-catalog-store-directory",
        str(paths["catalog"]),
        "--selective-catalog-receipt-json",
        str(receipt),
        "--expected-selective-catalog-receipt-sha256",
        _DIGEST,
        "--checkpoint-key",
        _DIGEST,
        "--checkpoint-manifest-sha256",
        _DIGEST,
        "--expected-trace-sha256",
        _DIGEST,
        "--expected-runtime-compiler-implementation-sha256",
        _OTHER_DIGEST,
        "--expected-sequence-source-sha256",
        _DIGEST,
    ]


def test_cli_accepts_only_exact_h776_resource_and_parameter_lane(
    tmp_path: Path,
) -> None:
    parser = runner.build_parser()
    args = parser.parse_args(_parser_args(tmp_path))
    config = runner.config_from_args(args)
    assert config.fold.compression.target_unique_parameters == 57_405_368
    assert config.fold.maximum_process_rss_bytes == 64 * 1024**3
    bad = _parser_args(tmp_path / "bad")
    target_index = bad.index("--target-unique-parameters") + 1
    bad[target_index] = "57405367"
    with pytest.raises(runner.Lfm2CpcfV5RealAttentionProbeRunnerError):
        runner.config_from_args(parser.parse_args(bad))


def _tree_bytes(root: Path):
    return {
        path.relative_to(root).as_posix(): (
            "directory" if path.is_dir() else path.read_bytes()
        )
        for path in sorted(root.rglob("*"))
    }


def test_overlapping_failure_diagnostic_never_mutates_checkpoint_store(
    tmp_path: Path,
) -> None:
    argv = _parser_args(tmp_path)
    resume = Path(argv[argv.index("--resume-store-directory") + 1])
    sentinel = resume / "sentinel.bin"
    sentinel.write_bytes(b"immutable-checkpoint-store")
    forbidden_parent = resume / "diagnostic-parent"
    forbidden = forbidden_parent / "failure.json"
    argv[argv.index("--diagnostic-json") + 1] = str(forbidden)
    before = _tree_bytes(resume)
    assert runner.main(argv) == 2
    assert _tree_bytes(resume) == before
    assert not forbidden_parent.exists()


def test_runtime_guard_starts_before_every_post_load_model_operation() -> None:
    guard = SimpleNamespace(
        optimizer_constructions=0,
        optimizer_steps=0,
        backward_calls=0,
        autograd_grad_calls=0,
        pruning_operations=0,
        quantization_operations=0,
    )
    report = runner._runtime_guard_report(
        guard,
        parent_execution={},
        target_execution={},
        allocator_policy={},
    )
    assert report["scope_started"] == "immediately_after_parent_model_load"
    assert report["parent_model_load_guarded"] is False
    assert report["parent_model_observer_started_immediately_after_load"] is True
    assert report["blank_target_synthesis_guarded"] is True
    assert report["target_model_observer_started_immediately_after_synthesis"] is True
    assert report["selective_provider_open_and_close_guarded"] is True
    assert report["source_tied_table_mapping_guarded"] is True
    assert report["layer_2_probe_guarded"] is True
    assert report["all_model_touching_operations_after_parent_load_guarded"] is True

    source = inspect.getsource(runner.run)
    parent_load = source.index("parent = load_local_lfm2_model")
    guard_start = source.index(
        "guard = dynamic_fold_executor._FoldOnlyTorchGuard()",
        parent_load,
    )
    guard_context = source.index("with torch.no_grad(), guard:", guard_start)
    parent_observer = source.index(
        "observe_model_execution(parent)",
        guard_context,
    )
    blank_synthesis = source.index(
        "blank = plan_and_synthesize_lfm2_cpcf_blank",
        parent_observer,
    )
    target_observer = source.index(
        "observe_model_execution(target)",
        blank_synthesis,
    )
    provider_open = source.index(
        "open_lfm2_cpcf_selective_evidence_provider_from_catalog",
        target_observer,
    )
    tied_mapping = source.index(
        "source_tied_table = base._mapped_source_tied_table",
        provider_open,
    )
    probe = source.index(
        "probe = execute_lfm2_cpcf_v5_real_attention_probe",
        tied_mapping,
    )
    assert (
        parent_load
        < guard_start
        < guard_context
        < parent_observer
        < blank_synthesis
        < target_observer
        < provider_open
        < tied_mapping
        < probe
    )
