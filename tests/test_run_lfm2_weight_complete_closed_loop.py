from __future__ import annotations

import copy

import pytest


torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")
pytest.importorskip("transformers")

from experiments import (  # noqa: E402
    run_lfm2_dynamic_structural_compression as baseline,
)
from experiments import run_lfm2_weight_complete_closed_loop as runner  # noqa: E402
from neural_forge.dynamic_balanced_circuit_basis import (  # noqa: E402
    _tensor_sha256 as _balanced_tensor_sha256,
)
from neural_forge.dynamic_weight_complete_basis import (  # noqa: E402
    _tensor_sha256,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
)
from neural_forge.hf_lfm2_dynamic_closed_loop_adapter import (  # noqa: E402
    Lfm2DynamicClosedLoopAdapterConfig,
)


def _sealed_baseline_report() -> dict:
    candidate = baseline._CANDIDATES_BY_NAME["h640_q10_kv5_ff1920"]
    mapping = "1" * 64
    state = "2" * 64
    report = {
        "candidate": candidate.as_dict(),
        "fold": {
            "folded_state_sha256": state,
            "global_residual_map": {
                "map_sha256": mapping,
                "prompt_independent_weight_basis_override": True,
                "dense_basis": True,
                "sparse_assignment_used": False,
            },
        },
        "residual_basis_selection": {
            "mode": "weight-complete",
            "prompt_independent_whole_weight_basis": True,
        },
        "fresh_local_reload": {
            "verified": True,
            "state_sha256": state,
        },
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "hub_loader_used": False,
        "dtype_or_packing_reduction_claimed": False,
    }
    report["run_sha256"] = baseline._json_sha256(report)
    return report


def test_baseline_contract_authenticates_the_weight_complete_fold() -> None:
    report = _sealed_baseline_report()
    candidate, mapping, state = runner._verified_baseline_contract(report)

    assert candidate.name == "h640_q10_kv5_ff1920"
    assert mapping == "1" * 64
    assert state == "2" * 64

    corrupt = copy.deepcopy(report)
    corrupt["training_steps"] = 1
    with pytest.raises(
        runner.Lfm2WeightCompleteClosedLoopError,
        match="seal",
    ):
        runner._verified_baseline_contract(corrupt)


def test_baseline_contract_authenticates_the_circuit_balanced_fold() -> None:
    report = _sealed_baseline_report()
    report["residual_basis_selection"] = {
        "mode": "circuit-balanced",
        "prompt_independent_whole_weight_basis": True,
        "cached_residual_controllability": True,
        "downstream_observability_balancing": True,
        "circuit_balanced_report": {"balanced_basis": {}},
    }
    report["run_sha256"] = baseline._json_sha256(
        {key: value for key, value in report.items() if key != "run_sha256"}
    )

    candidate, mapping, state = runner._verified_baseline_contract(report)

    assert candidate.name == "h640_q10_kv5_ff1920"
    assert mapping == "1" * 64
    assert state == "2" * 64


def test_persisted_weight_complete_basis_is_reauthenticated(
    tmp_path,
) -> None:
    basis = torch.eye(3, dtype=torch.float64)[:, :2].contiguous()
    full = torch.diag(
        torch.tensor([5.0, 4.0, 1.0], dtype=torch.float64)
    )
    balanced = torch.diag(
        torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    )
    combined = balanced.clone()
    basis_path = tmp_path / "residual-coordinate-basis.safetensors"
    safetensors.save_file(
        {
            "basis": basis,
            "full_weight_gram": full,
            "balanced_structural_gram": balanced,
            "combined_gram": combined,
        },
        str(basis_path),
    )
    basis_sha = _tensor_sha256(basis)
    generic_report = {
        "hashes": {
            "basis_sha256": basis_sha,
            "full_weight_gram_sha256": _tensor_sha256(full),
            "balanced_structural_gram_sha256": _tensor_sha256(balanced),
            "combined_gram_sha256": _tensor_sha256(combined),
        }
    }
    report = {
        "residual_basis_selection": {
            "artifact": {
                "file": basis_path.name,
                "file_sha256": runner._file_sha256(basis_path),
                "basis_sha256": basis_sha,
                "source_width": 3,
                "target_width": 2,
                "contains_complete_objective_grams": True,
            },
            "weight_complete_report": {"basis": generic_report},
        }
    }

    result, artifact = runner._load_weight_complete_basis(
        tmp_path,
        report,
        expected_map_sha256=basis_sha,
    )

    assert result.source_width == 3
    assert result.target_width == 2
    assert result.map_sha256 == basis_sha
    assert artifact["file_sha256"] == runner._file_sha256(basis_path)


def test_persisted_circuit_balanced_basis_is_reauthenticated(
    tmp_path,
) -> None:
    basis = torch.eye(3, dtype=torch.float64)[:, :2].contiguous()
    encoder = basis.T.contiguous()
    decoder = basis.clone().contiguous()
    full = torch.diag(torch.tensor([5.0, 4.0, 1.0], dtype=torch.float64))
    controllability = torch.diag(
        torch.tensor([4.0, 3.0, 1.0], dtype=torch.float64)
    )
    observability = torch.diag(
        torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    )
    heldout = torch.diag(
        torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64)
    )
    tensors = {
        "basis": basis,
        "balanced_encoder": encoder,
        "balanced_decoder": decoder,
        "full_weight_gram": full,
        "controllability_gram": controllability,
        "observability_gram": observability,
        "heldout_gram": heldout,
    }
    basis_path = tmp_path / "residual-coordinate-basis.safetensors"
    safetensors.save_file(tensors, str(basis_path))
    generic_hash_names = {
        "basis": "basis_sha256",
        "balanced_encoder": "balanced_encoder_sha256",
        "balanced_decoder": "balanced_decoder_sha256",
        "full_weight_gram": "full_weight_gram_sha256",
        "controllability_gram": "controllability_gram_sha256",
        "observability_gram": "observability_gram_sha256",
        "heldout_gram": "heldout_gram_sha256",
    }
    generic_report = {
        "hashes": {
            report_name: _balanced_tensor_sha256(tensors[tensor_name])
            for tensor_name, report_name in generic_hash_names.items()
        }
    }
    map_sha = generic_report["hashes"]["basis_sha256"]
    artifact_tensor_hashes = {
        name: baseline._tensor_sha256(value)
        for name, value in tensors.items()
    }
    report = {
        "residual_basis_selection": {
            "mode": "circuit-balanced",
            "artifact": {
                "file": basis_path.name,
                "file_sha256": runner._file_sha256(basis_path),
                "basis_sha256": map_sha,
                "source_width": 3,
                "target_width": 2,
                "tensor_names": sorted(tensors),
                "tensor_sha256": artifact_tensor_hashes,
                "reloaded_basis_tensor_sha256": (
                    artifact_tensor_hashes["basis"]
                ),
                "contains_complete_objective_grams": True,
                "contains_balanced_encoder_decoder": True,
                "fresh_tensor_reload_verified": True,
            },
            "circuit_balanced_report": {
                "balanced_basis": generic_report,
            },
        }
    }

    result, artifact = runner._load_residual_coordinate_map(
        tmp_path,
        report,
        expected_map_sha256=map_sha,
    )

    assert result.source_width == 3
    assert result.target_width == 2
    assert result.map_sha256 == map_sha
    assert artifact["mode"] == "circuit-balanced"
    assert artifact["file_sha256"] == runner._file_sha256(basis_path)


def test_replay_acceleration_is_explicit_opt_in() -> None:
    parser = runner.build_parser()
    required = [
        "--parent-model",
        "parent",
        "--baseline-child",
        "child",
        "--trace-cache",
        "cache",
        "--output",
        "output",
    ]
    default = parser.parse_args(required)
    accelerated = parser.parse_args(
        required
        + [
            "--replay-capture-mode",
            CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        ]
    )

    assert (
        default.replay_capture_mode
        == CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert (
        accelerated.replay_capture_mode
        == CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
    )


def test_output_interface_controls_are_explicit_and_match_adapter_defaults() -> None:
    parser = runner.build_parser()
    required = [
        "--parent-model",
        "parent",
        "--baseline-child",
        "child",
        "--trace-cache",
        "cache",
        "--output",
        "output",
    ]
    args = parser.parse_args(required)
    identity_args = parser.parse_args(
        required + ["--preserve-output-interface"]
    )
    defaults = Lfm2DynamicClosedLoopAdapterConfig()

    assert args.preserve_output_interface is False
    assert identity_args.preserve_output_interface is True
    assert args.output_ridge == defaults.output_ridge
    assert (
        args.output_max_prior_delta_ratio
        == defaults.output_max_prior_delta_ratio
    )
    assert (
        args.output_max_transform_condition_number
        == defaults.output_max_transform_condition_number
    )
    assert (
        args.output_min_transform_singular_value
        == defaults.output_min_transform_singular_value
    )
    assert (
        args.minimum_output_kl_improvement
        == defaults.minimum_output_kl_improvement
    )
    assert (
        args.minimum_output_known_sequence_nll_improvement
        == defaults.minimum_output_known_sequence_nll_improvement
    )
    assert (
        args.maximum_output_eos_margin_rmse_regression
        == defaults.maximum_output_eos_margin_rmse_regression
    )
    assert (
        args.output_validation_tolerance
        == defaults.output_validation_tolerance
    )
    assert (
        args.output_metric_chunk_rows
        == defaults.output_metric_chunk_rows
    )
    assert args.attention_consumer_mode == defaults.attention_consumer_mode
    assert (
        args.attention_value_output_iterations
        == defaults.attention_value_output_iterations
    )
    assert (
        args.attention_value_output_context_ridge
        == defaults.attention_value_output_context_ridge
    )
    assert (
        args.attention_value_output_fit_tolerance
        == defaults.attention_value_output_fit_tolerance
    )
    assert (
        args.attention_value_output_context_equivalence_tolerance
        == defaults.attention_value_output_context_equivalence_tolerance
    )
