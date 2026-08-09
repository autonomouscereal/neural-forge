"""Focused production-runner contract for fold-only AHB integration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "experiments" / "run_lfm2_dynamic_catalog.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_dynamic_catalog_ahb_runner",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _required_args(root: Path) -> list[str]:
    return [
        "--parent-model",
        str(root / "parent"),
        "--calibration",
        str(root / "calibration.json"),
        "--cached-parent-outputs",
        str(root / "parent-outputs.json"),
        "--cached-parent-outputs-sha256",
        "d" * 64,
        "--cache-root",
        str(root / "run" / "cache"),
        "--snapshot-root",
        str(root / "run" / "snapshots"),
        "--publish-destination",
        str(root / "run" / "published"),
        "--ledger",
        str(root / "run" / "ledger.jsonl"),
        "--receipt",
        str(root / "run" / "receipt.json"),
    ]


def _config(root: Path, **overrides):
    return MODULE.ProductionLfm2CatalogConfig(
        parent_model=root / "parent",
        calibration=None,
        sealed_parent_cache_directory=root / "sealed-parent-cache",
        cached_parent_outputs=root / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=root / "run" / "cache",
        snapshot_root=root / "run" / "snapshots",
        publish_destination=root / "run" / "published",
        ledger_path=root / "run" / "ledger.jsonl",
        receipt_path=root / "run" / "receipt.json",
        device="cpu",
        dtype="float32",
        **overrides,
    )


def _wiring_kwargs() -> dict[str, object]:
    return {
        "parent_cache_payload": None,
        "tokenizer_fingerprint": "tokenizer",
        "evidence_builder": None,
        "final_evaluator": None,
        "model_saver": None,
        "model_loader": None,
        "fresh_reload_evaluator": None,
        "ledger": None,
        "device": "cpu",
        "dtype": MODULE.torch.float32,
    }


def test_ahb_defaults_preserve_the_existing_runner_identity(tmp_path) -> None:
    config = _config(tmp_path)

    assert config.uses_autoregressive_hankel_observability is False
    assert config.dense_autoregressive_hankel_future_lags == ()
    assert "autoregressive_hankel_cache" not in config.identity()
    assert (
        "dense_autoregressive_hankel_observability" not in config.identity()["target"]
    )
    assert MODULE._autoregressive_hankel_cache_receipt_fields(config) == {}


def test_ahb_cli_binds_sparse_lags_solver_and_stable_cache(tmp_path) -> None:
    parser = MODULE.build_parser()
    args = parser.parse_args(
        [
            *_required_args(tmp_path),
            "--compression-ratio",
            "4",
            "--autoregressive-hankel-cache-root",
            str(tmp_path / "stable" / "ahb"),
            "--dense-autoregressive-hankel-future-lags",
            "0,1,2,4,8,16,32",
            "--dense-autoregressive-hankel-state-ridge-fraction",
            "2e-8",
            "--dense-autoregressive-hankel-future-ridge-fraction",
            "3e-8",
            "--dense-autoregressive-hankel-maximum-condition-number",
            "2e9",
        ]
    )
    config = MODULE.config_from_args(args)

    assert config.uses_autoregressive_hankel_observability is True
    assert config.dense_autoregressive_hankel_future_lags == (
        0,
        1,
        2,
        4,
        8,
        16,
        32,
    )
    policy = config.identity()["target"]["dense_autoregressive_hankel_observability"]
    assert policy["future_lags"] == [0, 1, 2, 4, 8, 16, 32]
    assert policy["fold_only_closed_form"] is True
    assert policy["training_steps"] == 0
    receipt = MODULE._autoregressive_hankel_cache_receipt_fields(config)[
        "autoregressive_hankel_cache"
    ]
    assert receipt["checkpoint_granularity_batches"] == 1
    assert receipt["bounded_memory_cumulative_aggregation_required"] is True
    assert receipt["all_statistics_shards_retained_in_memory"] is False
    assert receipt["parent_trace_regeneration"] is False


@pytest.mark.parametrize("lags", ["", "1,2", "0", "0,2,1", "0,1,1", "0,-1,2"])
def test_ahb_cli_refuses_noncausal_or_ambiguous_lag_schedules(lags) -> None:
    with pytest.raises(SystemExit):
        MODULE.build_parser().parse_args(
            [
                "--parent-model",
                "parent",
                "--cached-parent-outputs",
                "outputs.json",
                "--cached-parent-outputs-sha256",
                "d" * 64,
                "--cache-root",
                "cache",
                "--snapshot-root",
                "snapshots",
                "--publish-destination",
                "published",
                "--ledger",
                "ledger.jsonl",
                "--receipt",
                "receipt.json",
                "--dense-autoregressive-hankel-future-lags",
                lags,
            ]
        )


def test_ahb_requires_paired_cache_and_lags_and_automatic_balanced_fold(
    tmp_path,
) -> None:
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="cache root and nonempty future-lag schedule together",
    ):
        _config(
            tmp_path,
            autoregressive_hankel_cache_root=tmp_path / "stable" / "ahb",
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="compression_ratio request",
    ):
        _config(
            tmp_path,
            autoregressive_hankel_cache_root=tmp_path / "stable" / "ahb",
            dense_autoregressive_hankel_future_lags=(0, 1),
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="circuit_balanced",
    ):
        _config(
            tmp_path,
            compression_ratio=4.0,
            autoregressive_hankel_cache_root=tmp_path / "stable" / "ahb",
            dense_autoregressive_hankel_future_lags=(0, 1),
            dense_residual_basis_mode="weight_complete",
        )


def test_ahb_wiring_seam_fails_closed_on_configuration_artifact_mismatch(
    tmp_path,
) -> None:
    enabled = _config(
        tmp_path,
        compression_ratio=4.0,
        autoregressive_hankel_cache_root=tmp_path / "stable" / "ahb",
        dense_autoregressive_hankel_future_lags=(0, 1, 2),
    )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="fully authenticated aggregate observability artifact",
    ):
        MODULE.build_production_catalog_wiring(enabled, **_wiring_kwargs())

    disabled = _config(tmp_path / "disabled")
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="configuration did not bind",
    ):
        MODULE.build_production_catalog_wiring(
            disabled,
            autoregressive_hankel_observability=object(),
            **_wiring_kwargs(),
        )
