from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_lfm2_sparse_residual_vocab_frontier.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_sparse_residual_vocab_frontier",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _base_paths(tmp_path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in (
        "strong_source_model",
        "strong_source_cache",
        "original_parent_model",
        "parent_trajectory_cache",
        "recovery_horizon_cache",
        "tokenizer_directory",
    ):
        path = tmp_path / name
        path.mkdir(parents=True)
        result[name] = path
    recovery = tmp_path / "recovery.json"
    recovery.write_bytes(b"{}\n")
    holdout = tmp_path / "holdout.json"
    holdout.write_bytes(b"{}\n")
    authorization = tmp_path / "authorization.json"
    authorization.write_bytes(b"{}\n")
    result.update(
        {
            "recovery_parent_envelope": recovery,
            "expected_recovery_parent_envelope_file_sha256": _sha(recovery),
            "expected_recovery_horizon_cache_key": "a" * 64,
            "expected_parent_state_fingerprint": "b" * 64,
            "expected_tokenizer_fingerprint": "c" * 64,
            "expected_strong_source_state_sha256": "d" * 64,
            "expected_strong_source_cache_key": "e" * 64,
            "expected_original_parent_state_sha256": "f" * 64,
            "expected_parent_trajectory_cache_key": "1" * 64,
            "independent_holdout_parent_envelope": holdout,
            "expected_holdout_parent_envelope_file_sha256": _sha(holdout),
            "expected_holdout_parent_envelope_sha256": "2" * 64,
            "expected_holdout_manifest_sha256": "3" * 64,
            "expected_holdout_parent_state_fingerprint": "4" * 64,
            "expected_holdout_parent_model_fingerprint": "5" * 64,
            "execution_authorization_file": authorization,
            "expected_execution_authorization_file_sha256": _sha(authorization),
            "work_root": tmp_path / "work",
            "checkpoint_root": tmp_path / "checkpoint",
            "report_path": tmp_path / "report.json",
        }
    )
    return result


def _authorized_config(tmp_path: Path):
    paths = _base_paths(tmp_path)
    provisional = MODULE.SparseResidualFrontierConfig(**paths)
    document = {
        "schema": MODULE.AUTHORIZATION_SCHEMA,
        "authorized": True,
        "gpu_uuid": "GPU-test-uuid",
        **provisional.authorization_binding(),
    }
    document["authorization_sha256"] = MODULE._json_sha256(document)
    authorization = paths["execution_authorization_file"]
    assert isinstance(authorization, Path)
    authorization.write_bytes(_canonical(document))
    paths["expected_execution_authorization_file_sha256"] = _sha(authorization)
    return MODULE.SparseResidualFrontierConfig(**paths), paths


def test_large_target_plan_solves_35_and_refuses_impossible_50_percent() -> None:
    plan = MODULE.plan_sparse_residual_allocations(
        source_label="strong_10pct_child",
        source_unique_parameters=(
            MODULE.EXPECTED_STRONG_SOURCE_UNIQUE_PARAMETERS
        ),
        fractions=MODULE.STRONG_COMPRESSION_FRACTIONS,
        rank_grid=MODULE.STRONG_RANK_GRID,
    )
    by_fraction = {
        item["requested_total_compression_fraction"]: item for item in plan
    }

    assert tuple(by_fraction) == (0.15, 0.20, 0.25, 0.35, 0.50)
    assert by_fraction[0.35]["status"] == "solved"
    assert by_fraction[0.35]["solved_allocation_count"] == 3
    assert by_fraction[0.50]["status"] == "impossible"
    assert by_fraction[0.50]["solved_allocation_count"] == 0
    assert all(
        attempt["status"] == "impossible"
        for attempt in by_fraction[0.50]["attempts"]
    )
    for fraction in (0.15, 0.20, 0.25, 0.35):
        for attempt in by_fraction[fraction]["attempts"]:
            target = attempt["target"]
            assert (
                target["physical_target_unique_parameters"]
                <= target["requested_target_unique_parameters"]
            )
            assert (
                target["physical_total_compression_fraction"] >= fraction
            )
            assert target["selected_row_budget"] >= 256


def test_authorization_binds_gpu_holdout_and_full_target_grid(
    tmp_path: Path,
) -> None:
    config, _paths = _authorized_config(tmp_path)

    result = MODULE.authenticate_execution_authorization(
        config,
        observed_gpu_uuid="GPU-test-uuid",
    )

    assert result["authenticated_before_model_load"] is True
    assert result["strong_compression_fractions"] == [
        0.15,
        0.20,
        0.25,
        0.35,
        0.50,
    ]
    assert result["strong_rank_grid"]["0.50"] == [64, 32, 16]
    assert result["run_direct_parent_control"] is True
    assert result["training_allowed"] is False
    assert result["new_parent_trace_generation_allowed"] is False
    assert result["new_parent_response_generation_allowed"] is False


def test_authorization_or_holdout_tamper_fails_before_execution(
    tmp_path: Path,
) -> None:
    config, paths = _authorized_config(tmp_path)
    holdout = paths["independent_holdout_parent_envelope"]
    assert isinstance(holdout, Path)
    holdout.write_bytes(b'{"tampered":true}\n')

    with pytest.raises(
        MODULE.Lfm2SparseResidualFrontierError,
        match="holdout file hash",
    ):
        MODULE.authenticate_execution_authorization(
            config,
            observed_gpu_uuid="GPU-test-uuid",
        )


def test_runner_is_zero_training_local_only_and_authorized_first() -> None:
    source = inspect.getsource(MODULE)
    run_source = inspect.getsource(MODULE.run_experiment)

    assert source.count("analyze_tied_vocabulary_modules(") == 1
    assert "residual_output_energy(" in source
    assert "select_sparse_residual_token_rows(" in source
    assert "evaluate_sparse_residual_logit_validation(" in source
    assert "install_lfm2_sparse_residual_vocabulary_pair(" in source
    assert "evaluate_lfm2_intelligence_holdout(" in source
    assert "except BaseException" in source
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in source
    assert "AutoModel" not in source
    assert "AutoTokenizer" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "save_pretrained" not in source
    assert "authenticate_execution_authorization(config)" in run_source
    assert run_source.index(
        "authenticate_execution_authorization(config)"
    ) < run_source.index("load_local_lfm2_model(")


def test_real_lfm2_embedding_norm_name_is_resolved_before_fallback() -> None:
    native = torch.nn.Identity()
    fallback = torch.nn.LayerNorm(2)
    model = SimpleNamespace(
        model=SimpleNamespace(embedding_norm=native, norm=fallback)
    )

    assert MODULE._resolve_lfm2_final_norm(model) is native


def test_parser_requires_recovery_holdout_and_authorization_bindings() -> None:
    actions = {
        action.dest: action for action in MODULE.build_parser()._actions
    }
    required = {
        "strong_source_model",
        "strong_source_cache",
        "original_parent_model",
        "parent_trajectory_cache",
        "recovery_horizon_cache",
        "recovery_parent_envelope",
        "expected_recovery_parent_envelope_file_sha256",
        "expected_recovery_horizon_cache_key",
        "expected_parent_state_fingerprint",
        "tokenizer_directory",
        "expected_tokenizer_fingerprint",
        "expected_strong_source_state_sha256",
        "expected_strong_source_cache_key",
        "expected_original_parent_state_sha256",
        "expected_parent_trajectory_cache_key",
        "independent_holdout_parent_envelope",
        "expected_holdout_parent_envelope_file_sha256",
        "expected_holdout_parent_envelope_sha256",
        "expected_holdout_manifest_sha256",
        "expected_holdout_parent_state_fingerprint",
        "expected_holdout_parent_model_fingerprint",
        "execution_authorization_file",
        "expected_execution_authorization_file_sha256",
        "work_root",
        "checkpoint_root",
        "report_path",
    }
    assert required.issubset(actions)
    assert all(actions[name].required for name in required)
    assert actions["strong_compression_fractions"].default == (
        0.15,
        0.20,
        0.25,
        0.35,
        0.50,
    )
    assert actions["run_direct_parent_control"].default is True
    assert "--skip-direct-parent-control" in actions[
        "run_direct_parent_control"
    ].option_strings


def test_focused_strong_source_authorization_can_skip_direct_parent_control(
    tmp_path: Path,
) -> None:
    paths = _base_paths(tmp_path)
    paths["strong_compression_fractions"] = (0.35,)
    paths["run_direct_parent_control"] = False
    provisional = MODULE.SparseResidualFrontierConfig(**paths)
    document = {
        "schema": MODULE.AUTHORIZATION_SCHEMA,
        "authorized": True,
        "gpu_uuid": "GPU-test-uuid",
        **provisional.authorization_binding(),
    }
    document["authorization_sha256"] = MODULE._json_sha256(document)
    authorization = paths["execution_authorization_file"]
    assert isinstance(authorization, Path)
    authorization.write_bytes(_canonical(document))
    paths["expected_execution_authorization_file_sha256"] = _sha(
        authorization
    )
    config = MODULE.SparseResidualFrontierConfig(**paths)

    result = MODULE.authenticate_execution_authorization(
        config,
        observed_gpu_uuid="GPU-test-uuid",
    )

    assert result["strong_compression_fractions"] == [0.35]
    assert result["strong_rank_grid"] == {"0.35": [128, 96, 64]}
    assert result["run_direct_parent_control"] is False
