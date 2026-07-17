from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_lfm2_shared_residual_basis_fold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_shared_residual_basis_fold",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _paths(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "source-child"
    source_cache = tmp_path / "source-cache"
    parent_cache = tmp_path / "parent-cache"
    whole_output_cache = tmp_path / "whole-output-parent-cache"
    tokenizer = tmp_path / "tokenizer"
    for path in (
        source,
        source_cache,
        parent_cache,
        whole_output_cache,
        tokenizer,
    ):
        path.mkdir(parents=True)
    outputs = tmp_path / "parent-outputs.json"
    outputs.write_text("{}\n", encoding="ascii")
    return {
        "source_child_model": source,
        "source_child_cache": source_cache,
        "parent_cache": parent_cache,
        "whole_output_parent_cache": whole_output_cache,
        "cached_parent_outputs": outputs,
        "cached_parent_outputs_sha256": "a" * 64,
        "tokenizer_directory": tokenizer,
        "work_root": tmp_path / "work",
        "checkpoint_destination": tmp_path / "candidate-checkpoint",
        "report_path": tmp_path / "report.json",
    }


def test_defaults_bind_strong_child_and_both_large_targets(
    tmp_path: Path,
) -> None:
    config = MODULE.SharedResidualBasisRunConfig(**_paths(tmp_path))

    assert (
        config.expected_source_child_unique_parameters
        == MODULE.STRONG_SOURCE_CHILD_UNIQUE_PARAMETERS
    )
    assert (
        config.target_total_unique_parameters
        == MODULE.DEFAULT_TARGET_TOTAL_UNIQUE_PARAMETERS
    )
    assert (
        config.fallback_target_total_unique_parameters
        == MODULE.DEFAULT_FALLBACK_TARGET_TOTAL_UNIQUE_PARAMETERS
    )
    assert config.target_schedule(
        MODULE.STRONG_SOURCE_CHILD_UNIQUE_PARAMETERS
    ) == (
        MODULE.DEFAULT_TARGET_TOTAL_UNIQUE_PARAMETERS,
        MODULE.DEFAULT_FALLBACK_TARGET_TOTAL_UNIQUE_PARAMETERS,
    )
    assert config.roles == ("w2",)
    assert config.ridge == pytest.approx(1e-6)
    assert config.basis_weighting == "layer_balanced"
    serialized = config.as_dict()
    assert serialized["factorization"][
        "uniform_rank_across_selected_depth"
    ] is True
    assert serialized["factorization"][
        "one_physical_shared_output_basis"
    ] is True
    assert serialized["factorization"][
        "distinct_private_input_factor_per_layer"
    ] is True
    assert serialized["factorization"][
        "late_layer_rank_starvation_permitted"
    ] is False
    assert serialized["factorization"]["basis_weighting"] == "layer_balanced"
    comparison = serialized["known_independent_15pct_comparison"]
    assert comparison["ranks_by_layer"] == [680] * 6 + [
        233,
        185,
        202,
        172,
        195,
        220,
        216,
        253,
    ]
    assert comparison["total_rank"] == 5756
    assert comparison["minimum_rank"] == 172
    assert comparison["reliable_task_retention"] == "0/23"
    assert serialized["original_parent_model_loaded"] is False
    assert serialized["new_parent_forwards_allowed"] is False
    assert serialized["new_parent_response_generation_allowed"] is False
    assert serialized["training_allowed"] is False
    assert serialized["standalone_checkpoint_available"] is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"roles": ("w1",)}, "supports w2 only"),
        (
            {"expected_source_child_unique_parameters": 200_000_000},
            "strong 10% incumbent",
        ),
        ({"ridge": 0.0}, "finite positive"),
        ({"basis_weighting": "loudest_first"}, "layer_balanced or raw_stack"),
    ),
)
def test_configuration_rejects_scope_drift(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    values = {**_paths(tmp_path), **overrides}
    with pytest.raises(
        MODULE.Lfm2SharedResidualBasisRunnerError,
        match=message,
    ):
        MODULE.SharedResidualBasisRunConfig(**values)


def test_backend_is_uniform_shared_w2_and_exposes_rank_contrast(
    tmp_path: Path,
) -> None:
    config = MODULE.SharedResidualBasisRunConfig(**_paths(tmp_path))
    backend = MODULE._backend(config)
    plan = SimpleNamespace(
        rank=597,
        independent_uniform_rank_at_same_factor_budget=411,
        factorization=SimpleNamespace(
            shared_basis_parameters=597 * 1024,
            validation_worst_layer_relative_rmse=0.3,
            validation_worst_layer_cosine=0.91,
            validation_mean_layer_cosine=0.97,
            layers=(
                SimpleNamespace(
                    layer_index=13,
                    rank=597,
                    method="exact_weight_projection",
                    fit_relative_rmse=0.1,
                    validation_relative_rmse=0.2,
                    validation_cosine=0.95,
                    projection_validation_relative_rmse=0.2,
                    dense_parameters=2_074_624,
                    private_factor_parameters=597 * 2026,
                ),
            ),
        ),
    )

    diagnostics = backend.role_diagnostics(plan)

    assert backend.name == "uniform_shared_residual_output_basis"
    assert backend.phase_prefix == "shared_residual_basis"
    assert (
        backend.factorized_linear_type
        is MODULE.Lfm2SharedResidualFactorizedLinear
    )
    assert diagnostics[0]["rank"] == 597
    assert diagnostics[0][
        "shared_rank_advantage_over_independent_uniform_rank"
    ] == 186
    assert diagnostics[0]["late_layer_rank_starvation"] is False
    assert (
        diagnostics[0][
            "global_worst_layer_heldout_validation_relative_rmse"
        ]
        == pytest.approx(0.3)
    )


def test_runner_delegates_to_existing_strict_cache_behavior_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = MODULE.SharedResidualBasisRunConfig(**_paths(tmp_path))
    observed = {}

    def fake_run(config_value, *, backend, run_schema):
        observed["config"] = config_value
        observed["backend"] = backend
        observed["run_schema"] = run_schema
        return {
            "schema": run_schema,
            "status": "rejected",
            "checkpoint_published": False,
        }

    monkeypatch.setattr(MODULE.base, "run_experiment", fake_run)
    report = MODULE.run_experiment(config)

    assert observed["config"] is config
    assert (
        observed["backend"].name
        == "uniform_shared_residual_output_basis"
    )
    assert observed["run_schema"] == MODULE.RUN_SCHEMA
    assert report["checkpoint_published"] is False


def test_parser_is_local_only_and_reserves_checkpoint_without_writing() -> None:
    parser = MODULE.build_parser()
    actions = {action.dest: action for action in parser._actions}

    assert {
        "source_child_model",
        "source_child_cache",
        "parent_cache",
        "whole_output_parent_cache",
        "cached_parent_outputs",
        "cached_parent_outputs_sha256",
        "tokenizer_directory",
        "target_total_unique_parameters",
        "fallback_target_total_unique_parameters",
        "expected_source_child_unique_parameters",
        "checkpoint_destination",
        "report_path",
        "ridge",
        "basis_weighting",
    }.issubset(actions)
    assert "parent_model" not in actions
    assert (
        actions["expected_source_child_unique_parameters"].default
        == MODULE.STRONG_SOURCE_CHILD_UNIQUE_PARAMETERS
    )
    assert (
        actions["target_total_unique_parameters"].default
        == MODULE.DEFAULT_TARGET_TOTAL_UNIQUE_PARAMETERS
    )
    assert actions["checkpoint_destination"].required is True
    assert actions["basis_weighting"].default == "layer_balanced"


def test_runner_source_has_no_network_training_parent_loader_or_publication() -> None:
    source = inspect.getsource(MODULE)

    assert "AutoModel" not in source
    assert "AutoTokenizer" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "save_pretrained" not in source
    assert "pickle" not in source
    assert "parent_model:" not in source
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in source
    assert "PUBLICATION_BLOCK_STATUS" in source
