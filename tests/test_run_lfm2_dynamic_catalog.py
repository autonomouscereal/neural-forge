from __future__ import annotations

import hashlib
import importlib.util
import inspect
import io
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from neural_forge.dynamic_lfm2_behavior import (
    seal_lfm2_cached_whole_output_envelope,
)
from neural_forge.dynamic_behavior_execution import (
    BehaviorInferenceExecutionContract,
    execution_contract_satisfied,
    observe_model_execution,
)
from neural_forge.dynamic_counted_child_state import (
    ContinuousChildStateAllocation,
)
from neural_forge.hf_lfm2_cached_global_topology_frontier import (
    GLOBAL_TOPOLOGY_CAPACITY_PHASE,
    GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
    GLOBAL_TOPOLOGY_SELECTION_PHASE,
)
from neural_forge import hf_lfm2_circuit_trace_cache as trace_cache

from test_hf_lfm2_cached_global_topology_frontier import _phase_cache_case
from neural_forge import lfm2_carrier_route_runtime as carrier_runtime


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")


SCRIPT = (
    Path(__file__).resolve().parents[1] / "experiments" / "run_lfm2_dynamic_catalog.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_dynamic_catalog",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _FlushRecordingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def _progress_rows(output: str) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in output.splitlines() if line.strip()]
    return [
        row
        for row in rows
        if row.get("schema") == MODULE.PROGRESS_JOURNAL_SCHEMA
    ]


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--parent-model",
        str(tmp_path / "parent"),
        "--calibration",
        str(tmp_path / "calibration.json"),
        "--cached-parent-outputs",
        str(tmp_path / "parent-outputs.json"),
        "--cached-parent-outputs-sha256",
        "d" * 64,
        "--cache-root",
        str(tmp_path / "cache"),
        "--snapshot-root",
        str(tmp_path / "snapshots"),
        "--publish-destination",
        str(tmp_path / "published"),
        "--ledger",
        str(tmp_path / "ledger.jsonl"),
        "--receipt",
        str(tmp_path / "receipt.json"),
    ]


def _balanced_basis_phase_args(tmp_path: Path) -> list[str]:
    return [
        "--balanced-basis-phase-cache-root",
        str(tmp_path / "stable" / "balanced-basis"),
        "--balanced-basis-phase-event-log",
        str(tmp_path / "run" / "balanced-basis-events.jsonl"),
        "--balanced-basis-phase-hmac-key-file",
        str(tmp_path / "keys" / "balanced-basis.key"),
        "--balanced-basis-phase-hmac-key-id",
        "balanced-basis-test-v1",
    ]


def test_progress_journal_is_ordered_flushed_and_scalar_only() -> None:
    stream = _FlushRecordingStream()
    configuration_sha256 = "a" * 64
    journal = MODULE._ProgressJournal(
        configuration_sha256,
        stream=stream,
    )

    journal.emit("run_started", folding_only=True)
    journal.emit("parent_rows_materialization_started")
    journal.emit(
        "parent_rows_materialized",
        path_count=13728,
        retained_bytes=25 * 1024**3,
        trace_rows_sha256="b" * 64,
    )
    journal.emit(
        "global_frontier_planning_started",
        global_hidden_frontier_enabled=True,
    )
    journal.emit("parent_identity_digest_started")
    journal.emit("system_fold_started", parent_model_count=1)
    journal.emit("system_fold_completed", accepted_action_count=4)

    rows = _progress_rows(stream.getvalue())
    assert [row["stage"] for row in rows] == [
        "run_started",
        "parent_rows_materialization_started",
        "parent_rows_materialized",
        "global_frontier_planning_started",
        "parent_identity_digest_started",
        "system_fold_started",
        "system_fold_completed",
    ]
    assert [row["sequence"] for row in rows] == list(range(7))
    assert {row["configuration_sha256"] for row in rows} == {
        configuration_sha256
    }
    assert all(str(row["timestamp_utc"]).endswith("Z") for row in rows)
    assert stream.flush_count == 7

    unchanged = stream.getvalue()
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="JSON scalars",
    ):
        journal.emit("unsafe", tensor=torch.zeros(1))
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="Unsafe progress journal metadata field",
    ):
        journal.emit("unsafe", password="must-not-be-written")
    assert stream.getvalue() == unchanged


def _without_calibration(arguments: list[str]) -> list[str]:
    result = list(arguments)
    index = result.index("--calibration")
    del result[index : index + 2]
    return result


class _FixedVocabularyTokenizer:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def _sealed_test_parent(*, seed: int):
    config = transformers.Lfm2Config(
        vocab_size=31,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return transformers.Lfm2ForCausalLM(config).eval()


@pytest.fixture(scope="module")
def sealed_parent_cache_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("sealed-parent-cache")
    parent = _sealed_test_parent(seed=701)
    batches = tuple(
        {
            "input_ids": torch.tensor(
                [[1, 2, 3, 4 + index]],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "forced_decode_ids": torch.tensor(
                [[8 + index]],
                dtype=torch.long,
            ),
        }
        for index in range(3)
    )
    capture_config = MODULE.Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=3,
        max_decode_steps=1,
        logit_rtol=1e-5,
        logit_atol=1e-6,
        require_top1_agreement=True,
    )
    cache = MODULE.build_lfm2_circuit_trace_cache(
        root / "cache-root",
        parent,
        batches,
        config=capture_config,
        input_device="cpu",
    )
    return SimpleNamespace(
        root=root,
        parent=parent,
        cache=cache,
        capture_config=capture_config,
        tokenizer=_FixedVocabularyTokenizer(31),
    )


def _sealed_resolution_config(
    root: Path,
    cache_directory: Path,
    *,
    calibration: Path | None = None,
) -> object:
    return MODULE.ProductionLfm2CatalogConfig(
        parent_model=root / "parent",
        calibration=calibration,
        sealed_parent_cache_directory=cache_directory,
        cached_parent_outputs=root / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=root / "runtime-cache",
        snapshot_root=root / "snapshots",
        publish_destination=root / "published",
        ledger_path=root / "ledger.jsonl",
        receipt_path=root / "receipt.json",
        device="cpu",
        dtype="float32",
    )


def test_parser_defaults_to_accepted_ffn_2552_exact_budget(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(parser.parse_args(_required_args(tmp_path)))

    assert config.target_query_heads == 16
    assert config.target_key_value_heads == 8
    assert config.target_intermediate_size == 2552
    assert config.exact_target_unique_parameters == 229_349_120
    assert config.target_topology == MODULE.Lfm2Topology(16, 8, 2552)
    assert config.snapshot_chunk_bytes == 64 * 1024 * 1024
    assert config.require_capture_top1_agreement is False
    assert (
        config.child_decode_trace_mode
        == MODULE.FULL_PREFIX_CANONICAL_TRACE_MODE
    )
    assert config.identity()["capture"]["parent_decode_trace_mode"] == (
        MODULE.CACHED_VERIFIED_TRACE_MODE
    )
    assert config.whole_output_max_new_tokens == 64
    assert config.whole_output_minimum_task_score_retention == 1.0
    assert config.whole_output_maximum_parent_task_score_regressions == 0
    assert config.final_performance_metrics_diagnostic_only is False
    assert config.identity()["whole_output_gate"][
        "numeric_decision_authority"
    ] is True
    assert config.identity()["whole_output_gate"][
        "output_cap_hits_are_censored"
    ] is True
    assert config.dense_minimum_hidden_size == 32
    assert config.required_target_hidden_size is None
    assert config.identity()["target"]["dense_minimum_hidden_size"] == 32
    assert config.identity()["target"]["required_target_hidden_size"] is None
    assert config.balanced_basis_phase_cache_root is None
    assert config.balanced_basis_phase_event_log is None
    assert config.balanced_basis_phase_hmac_key_file is None
    assert config.balanced_basis_phase_hmac_key_id is None
    assert config.uses_balanced_basis_phase_cache is False
    assert "balanced_basis_phase_cache" not in config.identity()
    assert MODULE._balanced_basis_phase_cache_receipt_fields(config) == {}
    assert MODULE._balanced_basis_phase_anchor_authority(config) is None
    assert config.dense_macro_cell_attractor_gate_enabled is False
    assert config.dense_macro_cell_attractor_maximum_records == 4
    assert config.dense_macro_cell_attractor_maximum_steps == 8
    assert config.dense_attention_consumer_mode == "output-only"
    assert config.dense_attention_geometry_risk_enabled is False
    assert config.identity()["target"][
        "dense_attention_geometry_allocator"
    ]["enabled"] is False
    assert config.dense_attention_score_candidate_gate_enabled is False
    assert config.dense_attention_score_top_k == 8
    assert (
        config.dense_closed_loop_performance_metrics_diagnostic_only
        is False
    )
    assert config.identity()["target"][
        "dense_closed_loop_performance_metrics"
    ] == {
        "diagnostic_only": False,
        "numeric_decision_authority": True,
        "counterfactual_thresholds_recorded": True,
        "nonfinite_values_reject": True,
        "structural_and_integrity_failures_reject": True,
    }
    assert config.dense_closed_loop_replay_capture_mode == (
        MODULE.DENSE_CLOSED_LOOP_REPLAY_AUTO_MODE
    )
    assert config.resolved_dense_closed_loop_replay_capture_mode == (
        MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert config.identity()["target"]["dense_closed_loop_replay"] == {
        "policy": MODULE.DENSE_CLOSED_LOOP_REPLAY_AUTO_MODE,
        "resolved_mode": MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        "single_pass_requires_authenticated_heterogeneous_fp32": True,
        "automatic_selection_is_canonical_prefix_replay": True,
        "single_pass_certification_is_explicit_opt_in": True,
        "merge_adjacent_layer_captures": False,
    }
    assert config.identity()["target"][
        "dense_macro_cell_attractor_gate"
    ]["exact_token_match_is_diagnostic_only"]
    assert config.dense_vocabulary_margin_candidate_gate_enabled is False
    assert config.dense_streaming_vocabulary_margin_v2 is False
    assert config.dense_streaming_vocabulary_margin_pair_chunk_size == 256
    assert config.dense_vocabulary_margin_candidate_replay_mode == (
        MODULE.CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction
        == pytest.approx(0.9)
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction
        == pytest.approx(0.9)
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction
        == pytest.approx(0.9)
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction
        == pytest.approx(0.9)
    )
    help_text = parser.format_help().lower()
    assert "zero-gradient" in help_text
    assert "229,349,120" in help_text
    assert "--dense-vocabulary-margin-candidate-gate" in help_text
    assert "--dense-streaming-vocabulary-margin-v2" in help_text
    assert "--dense-macro-cell-attractor-gate" in help_text
    assert "--dense-attention-score-candidate-gate" in help_text
    assert "--dense-closed-loop-performance-metrics-diagnostic-only" in help_text
    assert "--final-performance-metrics-diagnostic-only" in help_text
    assert "--dense-minimum-hidden-size" in help_text
    assert "--required-target-hidden-size" in help_text
    assert "--sealed-parent-cache-directory" in help_text


@pytest.mark.parametrize(
    "omitted_flag",
    (
        "--balanced-basis-phase-cache-root",
        "--balanced-basis-phase-event-log",
        "--balanced-basis-phase-hmac-key-file",
        "--balanced-basis-phase-hmac-key-id",
    ),
)
def test_balanced_basis_phase_cache_cli_is_all_or_none(
    tmp_path,
    omitted_flag,
) -> None:
    arguments = [
        *_required_args(tmp_path / "run"),
        *_balanced_basis_phase_args(tmp_path),
    ]
    index = arguments.index(omitted_flag)
    del arguments[index : index + 2]

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires its cache root.*together",
    ):
        MODULE.config_from_args(MODULE.build_parser().parse_args(arguments))


def test_balanced_basis_phase_cache_cli_identity_excludes_key_path_and_secret(
    tmp_path,
) -> None:
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_required_args(tmp_path / "run"),
                *_balanced_basis_phase_args(tmp_path),
            ]
        )
    )

    assert config.uses_balanced_basis_phase_cache is True
    identity = config.identity()
    phase = identity["balanced_basis_phase_cache"]
    assert phase == {
        "enabled": True,
        "root": str((tmp_path / "stable" / "balanced-basis").absolute()),
        "event_log": str(
            (tmp_path / "run" / "balanced-basis-events.jsonl").absolute()
        ),
        "hmac_key_id": "balanced-basis-test-v1",
        "hmac_key_runtime_supplied": True,
    }
    serialized = json.dumps(identity, sort_keys=True)
    assert "balanced-basis.key" not in serialized
    receipt = MODULE._balanced_basis_phase_cache_receipt_fields(config)
    assert receipt["balanced_basis_phase_cache"]["hmac_secret_serialized"] is False
    assert "balanced-basis.key" not in json.dumps(receipt, sort_keys=True)


def test_balanced_basis_phase_cache_requires_circuit_balanced_mode(
    tmp_path,
) -> None:
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires dense_residual_basis_mode='circuit_balanced'",
    ):
        MODULE.config_from_args(
            MODULE.build_parser().parse_args(
                [
                    *_required_args(tmp_path / "run"),
                    *_balanced_basis_phase_args(tmp_path),
                    "--dense-residual-basis-mode",
                    "weight_complete",
                ]
            )
        )


@pytest.mark.parametrize(
    "unsafe_phase_root",
    (
        Path("run") / "phase-cache",
        Path("run") / "cache" / "phase-cache",
        Path("run") / "snapshots" / "phase-cache",
        Path("run") / "published" / "phase-cache",
    ),
)
def test_balanced_basis_phase_cache_requires_a_stable_cross_run_root(
    tmp_path,
    unsafe_phase_root,
) -> None:
    phase_arguments = _balanced_basis_phase_args(tmp_path)
    root_index = phase_arguments.index("--balanced-basis-phase-cache-root") + 1
    phase_arguments[root_index] = str(tmp_path / unsafe_phase_root)

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="stable cross-run root outside per-run",
    ):
        MODULE.config_from_args(
            MODULE.build_parser().parse_args(
                [*_required_args(tmp_path / "run"), *phase_arguments]
            )
        )


def test_balanced_basis_phase_event_log_requires_a_unique_file_path(
    tmp_path,
) -> None:
    phase_arguments = _balanced_basis_phase_args(tmp_path)
    event_index = phase_arguments.index("--balanced-basis-phase-event-log") + 1
    phase_arguments[event_index] = str(tmp_path / "run" / "ledger.jsonl")
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="must have a unique per-run path",
    ):
        MODULE.config_from_args(
            MODULE.build_parser().parse_args(
                [*_required_args(tmp_path / "run"), *phase_arguments]
            )
        )

    event_directory = tmp_path / "run" / "balanced-basis-events.jsonl"
    event_directory.mkdir(parents=True)
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="cannot be a directory",
    ):
        MODULE.config_from_args(
            MODULE.build_parser().parse_args(
                [
                    *_required_args(tmp_path / "run"),
                    *_balanced_basis_phase_args(tmp_path),
                ]
            )
        )


def test_balanced_basis_hmac_key_reader_preserves_exact_raw_bytes(
    tmp_path,
) -> None:
    source = tmp_path / "balanced-basis.key"
    expected = b"x" * 31 + b"\n"
    source.write_bytes(expected)
    if os.name != "nt":
        source.chmod(0o600)

    assert MODULE._read_balanced_basis_phase_hmac_key(source) == expected


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are required")
def test_balanced_basis_hmac_key_reader_rejects_group_or_world_access(
    tmp_path,
) -> None:
    source = tmp_path / "balanced-basis.key"
    source.write_bytes(b"x" * 32)
    source.chmod(0o640)

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="must not grant group or world permissions",
    ):
        MODULE._read_balanced_basis_phase_hmac_key(source)


def test_balanced_basis_hmac_key_reader_rejects_links_and_short_keys(
    tmp_path,
) -> None:
    short = tmp_path / "short.key"
    short.write_bytes(b"too-short")
    if os.name != "nt":
        short.chmod(0o600)
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="between 32 and 4096 raw bytes",
    ):
        MODULE._read_balanced_basis_phase_hmac_key(short)

    source = tmp_path / "source.key"
    source.write_bytes(b"x" * 32)
    if os.name != "nt":
        source.chmod(0o600)
    linked = tmp_path / "linked.key"
    try:
        linked.symlink_to(source)
    except OSError:
        pytest.skip("The test environment cannot create symlinks.")
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="ordinary non-linked file",
    ):
        MODULE._read_balanced_basis_phase_hmac_key(linked)


def test_balanced_basis_hmac_key_reader_rejects_hard_links(tmp_path) -> None:
    source = tmp_path / "source.key"
    source.write_bytes(b"x" * 32)
    if os.name != "nt":
        source.chmod(0o600)
    linked = tmp_path / "hard-linked.key"
    try:
        os.link(source, linked)
    except OSError:
        pytest.skip("The test environment cannot create hard links.")

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="ordinary non-linked file",
    ):
        MODULE._read_balanced_basis_phase_hmac_key(source)


def test_balanced_basis_phase_context_is_durable_and_relays_safe_telemetry(
    tmp_path,
    monkeypatch,
) -> None:
    key_file = tmp_path / "keys" / "balanced-basis.key"
    key_file.parent.mkdir()
    key_file.write_bytes(b"exact-raw-balanced-basis-key!!!!")
    if os.name != "nt":
        key_file.chmod(0o600)
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_required_args(tmp_path / "run"),
                *_balanced_basis_phase_args(tmp_path),
            ]
        )
    )
    authority = MODULE._balanced_basis_phase_anchor_authority(config)
    assert authority is not None
    dependencies = {"carrier_registration": "b" * 64}

    def provider():
        return dependencies

    stream = _FlushRecordingStream()
    journal = MODULE._ProgressJournal("a" * 64, stream=stream)
    monkeypatch.setattr(
        MODULE,
        "_process_memory_progress_metadata",
        lambda: {
            "process_rss_bytes": 101,
            "process_peak_rss_bytes": 202,
            "process_swap_bytes": 303,
        },
    )

    context = MODULE._balanced_basis_phase_cache_context(
        config,
        configuration_sha256="a" * 64,
        anchor_authority=authority,
        progress=journal,
        dependency_provider=provider,
    )

    assert context.cache.root == (
        tmp_path / "stable" / "balanced-basis"
    ).absolute()
    assert context.anchor_authority is authority
    assert context.events is not None
    assert context.events.path == (
        tmp_path / "run" / "balanced-basis-events.jsonl"
    ).absolute()
    assert context.events.run_id == f"lfm2-balanced-basis-{'a' * 64}"
    assert context.dependency_provider is provider
    assert context.dependency_provider() == dependencies
    assert context.existing_phase_reuse_request is None
    assert context.progress is not None
    context.progress(
        "balanced_basis_row_selection_progress",
        {
            "row_selection_schema": "balanced-row-progress-v1",
            "completed_batch_count": 256,
            "batch_count": 512,
            "process_rss_bytes": 999,
        },
    )
    context.progress(
        "balanced_basis_phase_cache_hit",
        {"phase_key_sha256": "c" * 64},
    )
    context.progress(
        "balanced_basis_existing_phase_reuse_hit",
        {
            "basis_reuse_provenance_sha256": "d" * 64,
            "basis_reuse_historical_phase_key_sha256": "e" * 64,
            "basis_reuse_historical_completion_sha256": "f" * 64,
            "basis_reuse_historical_map_sha256": "1" * 64,
            "basis_reuse_semantic_bindings_exact": True,
            "basis_reuse_math_settings_exact": True,
            "basis_reuse_historical_artifact_rewritten": False,
            "basis_reuse_cache_miss_build_allowed": False,
            "basis_reuse_trace_tensor_loads": 0,
            "basis_reuse_parent_forward_calls": 0,
            "basis_reuse_training_steps": 0,
            "basis_reuse_optimizer_steps": 0,
            "basis_reuse_backward_calls": 0,
        },
    )
    rows = _progress_rows(stream.getvalue())
    row_progress = rows[0]
    assert row_progress["process_rss_bytes"] == 101
    assert row_progress["process_peak_rss_bytes"] == 202
    assert row_progress["process_swap_bytes"] == 303
    assert "process_rss_bytes" not in rows[1]
    reuse_progress = rows[2]
    assert reuse_progress["stage"] == "balanced_basis_existing_phase_reuse_hit"
    assert reuse_progress["basis_reuse_provenance_sha256"] == "d" * 64
    assert reuse_progress["basis_reuse_trace_tensor_loads"] == 0
    assert reuse_progress["basis_reuse_parent_forward_calls"] == 0
    assert reuse_progress["basis_reuse_training_steps"] == 0
    assert reuse_progress["basis_reuse_optimizer_steps"] == 0
    assert reuse_progress["basis_reuse_backward_calls"] == 0

    unchanged = stream.getvalue()
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="reserved runner fields: schema",
    ):
        context.progress(
            "balanced_basis_row_selection_progress",
            {"schema": "must-not-shadow-the-runner"},
        )
    assert stream.getvalue() == unchanged


def test_balanced_basis_existing_phase_reuse_is_explicit_and_hit_only(
    tmp_path,
) -> None:
    key_file = tmp_path / "keys" / "balanced-basis.key"
    key_file.parent.mkdir()
    key_file.write_bytes(b"exact-raw-balanced-basis-key!!!!")
    if os.name != "nt":
        key_file.chmod(0o600)
    phase_key_sha256 = "e" * 64
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_required_args(tmp_path / "run"),
                *_balanced_basis_phase_args(tmp_path),
                "--balanced-basis-existing-phase-key-sha256",
                phase_key_sha256,
            ]
        )
    )
    authority = MODULE._balanced_basis_phase_anchor_authority(config)
    assert authority is not None
    context = MODULE._balanced_basis_phase_cache_context(
        config,
        configuration_sha256="a" * 64,
        anchor_authority=authority,
    )
    request = context.existing_phase_reuse_request
    assert request is not None
    assert request.phase_key_sha256 == phase_key_sha256
    assert request.expected_map_sha256 is None
    assert request.as_dict()["cache_miss_build_allowed"] is False
    assert config.identity()["balanced_basis_phase_cache"][
        "existing_phase_reuse_key_sha256"
    ] == phase_key_sha256


def test_balanced_basis_existing_phase_reuse_requires_authenticated_cache(
    tmp_path,
) -> None:
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires the authenticated balanced-basis phase cache",
    ):
        MODULE.config_from_args(
            MODULE.build_parser().parse_args(
                [
                    *_required_args(tmp_path),
                    "--balanced-basis-existing-phase-key-sha256",
                    "e" * 64,
                ]
            )
        )


def test_carrier_phase_dependencies_are_freshly_reauthenticated_each_time() -> None:
    class _Registration:
        fingerprint = "a" * 64
        expected_artifact_sha256 = "b" * 64
        expected_span_evidence_sha256 = "c" * 64

        def __init__(self) -> None:
            self.verify_calls = 0
            self.plan_calls = 0
            self.plan_fingerprint = "d" * 64

        def verify(self):
            self.verify_calls += 1
            return self

        def current_exact4x_counted_plan(self):
            self.plan_calls += 1
            return SimpleNamespace(fingerprint=self.plan_fingerprint)

    registration = _Registration()
    arguments = {
        "expected_registration_fingerprint": "a" * 64,
        "expected_route_artifact_sha256": "b" * 64,
        "expected_span_evidence_sha256": "c" * 64,
        "expected_counted_plan_fingerprint": "d" * 64,
    }

    for _ in range(2):
        dependencies = MODULE._authenticated_carrier_phase_dependencies(
            registration,
            **arguments,
        )
        assert dict(dependencies) == {
            "carrier_registration": "a" * 64,
            "carrier_route_artifact": "b" * 64,
            "carrier_span_evidence": "c" * 64,
            "counted_child_state_plan": "d" * 64,
        }
    assert registration.verify_calls == 2
    assert registration.plan_calls == 2

    registration.plan_fingerprint = "e" * 64
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="Carrier dependencies changed",
    ):
        MODULE._authenticated_carrier_phase_dependencies(
            registration,
            **arguments,
        )
    assert registration.verify_calls == 3
    assert registration.plan_calls == 3


def test_final_performance_diagnostic_only_cli_requires_manual_review(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--final-performance-metrics-diagnostic-only",
            ]
        )
    )

    assert config.final_performance_metrics_diagnostic_only is True
    policy = config.identity()["whole_output_gate"]
    assert policy["performance_metrics_diagnostic_only"] is True
    assert policy["numeric_decision_authority"] is False
    assert policy["manual_intelligence_review_required"] is True
    assert policy["output_cap_hits_are_censored"] is True
    assert policy["safety_behavior_is_separate_from_folding_quality"] is True


def test_closed_loop_performance_diagnostic_only_cli_identity_and_receipt(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "10",
                "--dense-run-closed-loop",
                "--dense-closed-loop-performance-metrics-diagnostic-only",
            ]
        )
    )

    assert config.dense_closed_loop_performance_metrics_diagnostic_only
    identity_policy = config.identity()["target"][
        "dense_closed_loop_performance_metrics"
    ]
    assert identity_policy == {
        "diagnostic_only": True,
        "numeric_decision_authority": False,
        "counterfactual_thresholds_recorded": True,
        "nonfinite_values_reject": True,
        "structural_and_integrity_failures_reject": True,
    }

    receipt = MODULE.run_production_catalog(config)
    receipt_policy = receipt["closed_loop_performance_metrics_policy"]
    assert receipt["status"] == "failed_closed"
    assert receipt_policy == {
        **identity_policy,
        "external_whole_output_behavior_remains_promotion_authority": True,
        "training_steps": 0,
    }
    assert json.loads(
        config.receipt_path.read_text(encoding="ascii")
    ) == receipt


def test_closed_loop_performance_diagnostic_only_requires_dense_fold(
    tmp_path,
) -> None:
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires an automatic compression_ratio request",
    ):
        MODULE.config_from_args(
            MODULE.build_parser().parse_args(
                [
                    *_required_args(tmp_path),
                    "--dense-closed-loop-performance-metrics-diagnostic-only",
                ]
            )
        )


def test_dense_closed_loop_replay_policy_resolves_and_fails_closed(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    explicit_canonical = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--dense-closed-loop-replay-mode",
                MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
            ]
        )
    )
    assert explicit_canonical.dense_closed_loop_replay_capture_mode == (
        MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert explicit_canonical.resolved_dense_closed_loop_replay_capture_mode == (
        MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires the authenticated heterogeneous dense FP32 child path",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--dense-closed-loop-replay-mode",
                    MODULE.CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
                ]
            )
        )

    # Automatic selection is canonical prefix replay for every child, including
    # the heterogeneous FP32 lane: in-loop single-pass certification costs a
    # second replay per capture point and is now an explicit opt-in only.
    a0006_like = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "4",
                "--dense-heterogeneous-layout",
                "--dense-run-closed-loop",
                "--dtype",
                "float32",
            ]
        )
    )
    assert a0006_like.dense_closed_loop_replay_capture_mode == (
        MODULE.DENSE_CLOSED_LOOP_REPLAY_AUTO_MODE
    )
    assert a0006_like.resolved_dense_closed_loop_replay_capture_mode == (
        MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )

    explicit_single_pass = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "4",
                "--dense-heterogeneous-layout",
                "--dense-run-closed-loop",
                "--dtype",
                "float32",
                "--dense-closed-loop-replay-mode",
                MODULE.CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
            ]
        )
    )
    assert (
        explicit_single_pass.resolved_dense_closed_loop_replay_capture_mode
        == MODULE.CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
    )


@pytest.mark.parametrize("minimum_hidden_size", (0, 7))
def test_dense_minimum_hidden_size_must_be_a_positive_multiple_of_eight(
    tmp_path,
    minimum_hidden_size,
) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match=(
            "dense_minimum_hidden_size must be a positive integer"
            if minimum_hidden_size == 0
            else "dense_minimum_hidden_size must be a multiple of eight"
        ),
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--dense-minimum-hidden-size",
                    str(minimum_hidden_size),
                ]
            )
        )


def test_parser_accepts_sealed_cache_without_calibration(tmp_path) -> None:
    arguments = [
        *_without_calibration(_required_args(tmp_path)),
        "--sealed-parent-cache-directory",
        str(tmp_path / "sealed-cache"),
    ]

    config = MODULE.config_from_args(MODULE.build_parser().parse_args(arguments))

    assert config.calibration is None
    assert config.sealed_parent_cache_directory == tmp_path / "sealed-cache"
    assert (
        config.parent_cache_source_mode
        == MODULE.SEALED_PARENT_CACHE_SOURCE_MODE
    )
    assert config.identity()["parent_cache_source_mode"] == (
        MODULE.SEALED_PARENT_CACHE_SOURCE_MODE
    )


def test_configuration_requires_calibration_or_sealed_cache(tmp_path) -> None:
    arguments = _without_calibration(_required_args(tmp_path))

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="calibration is required unless",
    ):
        MODULE.config_from_args(MODULE.build_parser().parse_args(arguments))


def test_parser_can_explicitly_restore_cached_verified_child_traces(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--child-decode-trace-mode",
                MODULE.CACHED_VERIFIED_TRACE_MODE,
            ]
        )
    )

    assert config.child_decode_trace_mode == MODULE.CACHED_VERIFIED_TRACE_MODE


def test_parser_accepts_explicit_ninety_percent_behavior_policy(tmp_path) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--whole-output-maximum-new-empty-responses",
                "2",
                "--whole-output-maximum-new-fragment-responses",
                "2",
                "--whole-output-minimum-parent-nonempty-retention",
                "0.9",
                "--whole-output-minimum-task-score-retention",
                "0.9",
                "--whole-output-maximum-parent-task-score-regressions",
                "2",
                "--whole-output-maximum-new-generation-limit-saturations",
                "2",
                "--whole-output-maximum-repeated-ngram-share-regressions",
                "2",
            ]
        )
    )

    assert config.whole_output_maximum_new_empty_responses == 2
    assert config.whole_output_maximum_new_fragment_responses == 2
    assert config.whole_output_minimum_parent_nonempty_retention == 0.9
    assert config.whole_output_minimum_task_score_retention == 0.9
    assert config.whole_output_maximum_parent_task_score_regressions == 2
    assert config.whole_output_maximum_new_generation_limit_saturations == 2
    assert config.whole_output_maximum_repeated_ngram_share_regressions == 2


def test_parser_accepts_automatic_dense_ratio_and_cached_margin_basis(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "10",
                "--dense-maximum-candidates",
                "17",
                "--dense-residual-basis-mode",
                "circuit_balanced",
                "--dense-vocabulary-margin-observability",
                "--dense-vocabulary-margin-top-k",
                "12",
                "--dense-vocabulary-margin-floor-fraction",
                "0.125",
                "--sealed-parent-cache-directory",
                str(tmp_path / "sealed-cache"),
                "--dense-streaming-vocabulary-margin-v2",
                "--dense-streaming-vocabulary-margin-pair-chunk-size",
                "17",
                "--dense-vocabulary-margin-candidate-gate",
                "--dense-vocabulary-margin-candidate-replay-mode",
                MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
                "--dense-vocabulary-margin-candidate-minimum-sign-agreement",
                "0.81",
                (
                    "--dense-vocabulary-margin-candidate-minimum-"
                    "top-competitor-nonnegative"
                ),
                "0.82",
                (
                    "--dense-vocabulary-margin-candidate-minimum-all-top-k-nonnegative"
                ),
                "0.83",
                "--dense-vocabulary-margin-candidate-minimum-exact-top1",
                "0.84",
            ]
        )
    )

    assert config.compression_ratio == 10.0
    assert config.dense_maximum_candidates == 17
    assert config.dense_residual_basis_mode == "circuit_balanced"
    assert config.dense_vocabulary_margin_observability is True
    assert config.dense_vocabulary_margin_top_k == 12
    assert config.dense_vocabulary_margin_floor_fraction == 0.125
    assert config.dense_streaming_vocabulary_margin_v2 is True
    assert config.dense_streaming_vocabulary_margin_pair_chunk_size == 17
    assert config.dense_vocabulary_margin_candidate_gate_enabled is True
    assert config.dense_vocabulary_margin_candidate_replay_mode == (
        MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction
        == pytest.approx(0.81)
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction
        == pytest.approx(0.82)
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction
        == pytest.approx(0.83)
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction
        == pytest.approx(0.84)
    )
    target_identity = config.identity()["target"]
    assert target_identity["fold_method_id"] == MODULE.DENSE_BUDGET_MODEL_METHOD
    assert target_identity["compression_ratio"] == 10.0
    assert target_identity[
        "dense_vocabulary_margin_candidate_gate_enabled"
    ] is True
    assert target_identity["dense_streaming_vocabulary_margin_v2"] is True
    assert target_identity[
        "dense_streaming_vocabulary_margin_pair_chunk_size"
    ] == 17
    assert target_identity[
        "dense_vocabulary_margin_candidate_replay_mode"
    ] == MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    assert target_identity[
        "dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction"
    ] == pytest.approx(0.81)
    assert target_identity[
        (
            "dense_vocabulary_margin_candidate_minimum_"
            "top_competitor_nonnegative_fraction"
        )
    ] == pytest.approx(0.82)
    assert target_identity[
        (
            "dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction"
        )
    ] == pytest.approx(0.83)
    assert target_identity[
        (
            "dense_vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction"
        )
    ] == pytest.approx(0.84)


def test_parser_accepts_cached_heterogeneous_exact_budget(tmp_path) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "4",
                "--dense-heterogeneous-layout",
                "--dense-heterogeneous-factor-writer-rank-discovery",
                "--dense-heterogeneous-hidden-size",
                "96",
                "--dense-heterogeneous-target-parameters",
                "123456",
            ]
        )
    )

    assert config.compression_ratio == 4.0
    assert config.dense_heterogeneous_layout is True
    assert config.dense_heterogeneous_factor_writer_rank_discovery is True
    assert config.dtype == "float32"
    assert config.dense_heterogeneous_hidden_size == 96
    assert config.dense_heterogeneous_target_parameters == 123_456
    target_identity = config.identity()["target"]
    assert target_identity["dense_heterogeneous_layout"] is True
    assert target_identity["dense_heterogeneous_factor_writer_rank_discovery"] is True
    assert target_identity["dense_heterogeneous_hidden_size"] == 96
    assert target_identity["dense_heterogeneous_target_parameters"] == 123_456


def _carrier_arguments(tmp_path: Path) -> list[str]:
    return [
        *_required_args(tmp_path),
        "--sealed-parent-cache-directory",
        str(tmp_path / "sealed-cache"),
        "--target-tokenizer-directory",
        str(tmp_path / "target-tokenizer"),
        "--target-tokenizer-calibration",
        str(tmp_path / "target-calibration.json"),
        "--tokenizer-bridge-artifact-sha256",
        "1" * 64,
        "--tokenizer-bridge-profile-sha256",
        "2" * 64,
        "--tokenizer-bridge-mapped-parent-outputs",
        str(tmp_path / "mapped-output.json"),
        "--tokenizer-bridge-mapped-parent-outputs-sha256",
        "3" * 64,
        "--carrier-route-publication",
        str(tmp_path / "carrier-route"),
        "--carrier-route-artifact-sha256",
        "4" * 64,
        "--carrier-span-evidence",
        str(tmp_path / "span-evidence.json"),
        "--carrier-span-evidence-file-sha256",
        "5" * 64,
        "--carrier-span-evidence-sha256",
        "6" * 64,
        "--routed-manual-prompts",
        str(tmp_path / "manual-prompts.json"),
        "--routed-manual-prompts-sha256",
        "7" * 64,
        "--compression-ratio",
        "4",
        "--dense-heterogeneous-layout",
        "--dense-heterogeneous-global-hidden-frontier",
        "--derived-global-topology-cache-root",
        str(tmp_path / "shared-derived-topology-cache"),
        "--dense-heterogeneous-factor-writer-rank-discovery",
        "--dense-heterogeneous-target-parameters",
        "53552368",
        "--dense-run-closed-loop",
        "--dense-body-reclosure-mode",
        "moving_gauge_v3",
        "--dense-closed-loop-replay-mode",
        MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        "--dense-closed-loop-performance-metrics-diagnostic-only",
        "--dense-attention-consumer-mode",
        "coupled-value-output",
        "--dense-attention-geometry-risk",
        "--dense-vocabulary-margin-observability",
        "--dense-streaming-vocabulary-margin-v2",
        "--final-performance-metrics-diagnostic-only",
    ]


def test_parser_seals_carrier_total_while_budgeting_only_h816_body(
    tmp_path,
) -> None:
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(_carrier_arguments(tmp_path))
    )

    assert config.uses_carrier_routes is True
    assert config.global_topology_phase_cache_root == (
        tmp_path / "shared-derived-topology-cache"
    ).absolute()
    assert config.identity()["derived_global_topology_cache_root"] == str(
        (tmp_path / "shared-derived-topology-cache").absolute()
    )
    assert config.dense_heterogeneous_target_parameters == 53_552_368
    assert config.counted_dense_body_parameters == 53_552_368
    assert config.counted_target_total_parameters == 57_420_144
    assert (
        config.counted_target_total_parameters
        - config.counted_dense_body_parameters
        == 3_867_776
    )
    assert config.carrier_expected_target_rows == 4_096
    assert config.carrier_expected_hidden_size == 816
    assert config.required_target_hidden_size == 816
    assert config.routed_manual_initial_max_new_tokens == 1_024
    assert config.routed_manual_retry_max_new_tokens == 2_048
    assert config.routed_manual_checkpoint_directory == tmp_path / (
        "receipt-routed-manual-checkpoints"
    )
    assert config.semantic_source_tokenizer_directory == config.tokenizer_source
    identity = config.identity()["carrier_routes"]
    assert identity["counted_dense_body_parameters"] == 53_552_368
    assert identity["counted_target_total_parameters"] == 57_420_144
    assert identity["output_cap_hits_are_censored"] is True
    assert identity["remaining_output_cap_censored_endpoint"] == "unknown"
    assert identity["remaining_output_cap_censored_is_failure"] is False
    assert identity["safety_behavior_is_separate_from_folding_quality"] is True
    assert identity["safety_prompts_excluded_from_intelligence_failure"] is True
    assert identity[
        "manual_whole_response_intelligence_review_is_authoritative"
    ] is True
    assert identity["numeric_metrics_are_diagnostic_only"] is True
    assert identity["numeric_metrics_have_acceptance_authority"] is False
    assert config.identity()["target"]["required_target_hidden_size"] == 816
    assert identity["manual_checkpoint_directory"].endswith(
        "receipt-routed-manual-checkpoints"
    )
    assert config.dense_run_closed_loop is True
    assert config.dense_body_reclosure_mode == "moving_gauge_v3"
    assert config.dense_closed_loop_replay_capture_mode == (
        MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert config.dense_closed_loop_performance_metrics_diagnostic_only is True
    assert config.dense_attention_consumer_mode == "coupled-value-output"
    assert config.dense_attention_geometry_risk_enabled is True
    assert config.dense_vocabulary_margin_observability is True
    assert config.dense_streaming_vocabulary_margin_v2 is True


def test_carrier_mode_rejects_legacy_full_body_budget(tmp_path) -> None:
    arguments = _carrier_arguments(tmp_path)
    index = arguments.index("--dense-heterogeneous-target-parameters")
    arguments[index + 1] = "57420144"

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="reserve both independent route channels",
    ):
        MODULE.config_from_args(MODULE.build_parser().parse_args(arguments))


def test_required_target_hidden_size_is_general_and_carrier_derived(
    tmp_path,
) -> None:
    general = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_required_args(tmp_path / "general"),
                "--compression-ratio",
                "4",
                "--dense-heterogeneous-layout",
                "--dense-heterogeneous-global-hidden-frontier",
                "--derived-global-topology-cache-root",
                str(tmp_path / "general" / "derived-phases"),
                "--required-target-hidden-size",
                "64",
            ]
        )
    )
    carrier = MODULE.config_from_args(
        MODULE.build_parser().parse_args(_carrier_arguments(tmp_path / "carrier"))
    )

    assert general.required_target_hidden_size == 64
    assert general.identity()["target"]["required_target_hidden_size"] == 64
    assert carrier.required_target_hidden_size == 816


def test_carrier_mode_rejects_conflicting_target_hidden_constraint(
    tmp_path,
) -> None:
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="equal carrier_expected_hidden_size",
    ):
        MODULE.config_from_args(
            MODULE.build_parser().parse_args(
                [
                    *_carrier_arguments(tmp_path),
                    "--required-target-hidden-size",
                    "840",
                ]
            )
        )


def test_carrier_mode_rejects_unpinned_weaker_folding_lane(tmp_path) -> None:
    arguments = _carrier_arguments(tmp_path)
    arguments.remove("--dense-attention-geometry-risk")

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="best proven folding lane",
    ):
        MODULE.config_from_args(MODULE.build_parser().parse_args(arguments))


def test_carrier_mode_keeps_review_checkpoints_outside_published_child(
    tmp_path,
) -> None:
    arguments = [
        *_carrier_arguments(tmp_path),
        "--routed-manual-checkpoint-directory",
        str(tmp_path / "published" / "child" / "review-checkpoints"),
    ]

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="distinct external paths outside",
    ):
        MODULE.config_from_args(MODULE.build_parser().parse_args(arguments))


def _carrier_test_allocations():
    return (
        ContinuousChildStateAllocation(
            state_id="semantic-carrier-route.c_in",
            parameter_elements=1_933_888,
            tensor_sha256="8" * 64,
            artifact_sha256="9" * 64,
        ),
        ContinuousChildStateAllocation(
            state_id="semantic-carrier-route.c_out",
            parameter_elements=1_933_888,
            tensor_sha256="a" * 64,
            artifact_sha256="9" * 64,
        ),
    )


def test_carrier_counted_plan_verifies_body_routes_and_realized_ratio() -> None:
    allocations = _carrier_test_allocations()
    plan = MODULE.build_counted_child_state_plan(
        parent_unique_parameters=229_693_184,
        target_total_parameter_elements=57_420_144,
        continuous_allocations=allocations,
    )

    verification = MODULE.verify_counted_child_state(
        plan,
        physical_dense_parameter_elements=53_552_368,
        observed_allocations=allocations,
    )

    assert plan.dense_model_parameter_elements == 53_552_368
    assert plan.continuous_parameter_elements == 3_867_776
    assert verification["target_total_parameter_elements"] == 57_420_144
    assert verification["physical_dense_parameter_elements"] == 53_552_368
    assert verification["realized_parameter_compaction_ratio"] == pytest.approx(
        229_693_184 / 57_420_144
    )
    assert verification["realized_parameter_compaction_ratio"] != 4.0

    registration = SimpleNamespace(
        current_exact4x_counted_plan=lambda: plan
    )
    assert MODULE._build_registered_carrier_counted_plan(
        registration,
        parent_unique_parameters=229_693_184,
        target_total_parameter_elements=57_420_144,
        observed_allocations=allocations,
    ) == plan
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="229,693,184-element LFM2 source",
    ):
        MODULE._build_registered_carrier_counted_plan(
            registration,
            parent_unique_parameters=229_693_183,
            target_total_parameter_elements=57_420_144,
            observed_allocations=allocations,
        )


def test_carrier_artifact_io_save_then_package_and_new_instance_loads_locally(
    tmp_path,
    monkeypatch,
) -> None:
    events: list[str] = []
    allocations = _carrier_test_allocations()

    class Model:
        pass

    class BaseIO:
        def save(self, _model, destination):
            events.append("base-save")
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("config.json", "model.safetensors", "tokenizer.json"):
                (destination / name).write_bytes(name.encode("ascii"))

        def load(self, _source):
            events.append("base-load")
            return Model()

    lexicon = SimpleNamespace(tokenizer=object(), package_sha256="b" * 64)
    routes = SimpleNamespace(
        package_sha256="c" * 64,
        allocations=allocations,
    )
    package = SimpleNamespace(
        package_sha256="d" * 64,
        routes=routes,
        semantic_source_lexicon=lexicon,
        receipt={
            "base_root_file_records": [
                {
                    "relative_path": name,
                    "size_bytes": len(name),
                    "sha256": "e" * 64,
                }
                for name in ("config.json", "model.safetensors", "tokenizer.json")
            ],
            "child_file_records": [],
        },
    )

    class Delegate:
        saver_id = "carrier-save"
        loader_id = "carrier-load"
        resource_fingerprint = "f" * 64

        def __init__(self, *, base, **_kwargs):
            self.base = base

        def save(self, model, destination):
            self.base.save(model, destination)
            events.append("route-and-lexicon-package")
            (destination / MODULE.PACKAGED_ROUTE_DIRECTORY).mkdir()
            (destination / MODULE.SEMANTIC_SOURCE_LEXICON_DIRECTORY).mkdir()
            (destination / MODULE.CARRIER_CHILD_PACKAGE_RECEIPT).write_bytes(
                b"receipt"
            )

        def load(self, source):
            model = self.base.load(source)
            events.append("child-local-bind")
            return model

        @staticmethod
        def register_in_memory(model):
            return model

    monkeypatch.setattr(MODULE, "Lfm2CarrierChildCheckpointIO", Delegate)
    monkeypatch.setattr(
        MODULE,
        "load_authenticated_lfm2_carrier_child_package",
        lambda *_args, **_kwargs: package,
    )
    registration = SimpleNamespace(
        expected_artifact_sha256="9" * 64,
        expected_target_rows=4_096,
        expected_hidden_size=816,
    )
    arguments = {
        "base": BaseIO(),
        "registration": registration,
        "semantic_source_tokenizer_directory": tmp_path / "semantic-tokenizer",
        "semantic_source_tokenizer_semantics_sha256": "e" * 64,
        "semantic_source_tokenizer": object(),
        "target_tokenizer": object(),
    }
    destination = tmp_path / "child"
    artifact_io = MODULE.CarrierAwareDirectLocalArtifactIO(**arguments)
    artifact_io.save(Model(), destination)
    loaded = artifact_io.load(destination)

    assert isinstance(loaded, Model)
    assert events == [
        "base-save",
        "route-and-lexicon-package",
        "base-load",
        "child-local-bind",
    ]
    inventory = artifact_io.verify_outer_inventory(destination)
    assert inventory["process_local_save_history_required"] is False
    assert inventory["carrier_child_package_sha256"] == "d" * 64
    assert inventory["only_base_checkpoint_tokenizer_and_counted_resources"]


def test_catalog_carrier_save_supports_child_only_loader_after_sources_deleted(
    tmp_path,
) -> None:
    fixture_path = Path(__file__).with_name(
        "test_lfm2_carrier_child_package_fresh_process.py"
    )
    fixture_spec = importlib.util.spec_from_file_location(
        "catalog_carrier_child_fixture",
        fixture_path,
    )
    assert fixture_spec is not None and fixture_spec.loader is not None
    fixture = importlib.util.module_from_spec(fixture_spec)
    sys.modules[fixture_spec.name] = fixture
    fixture_spec.loader.exec_module(fixture)

    semantic_directory = tmp_path / "semantic-tokenizer"
    semantic_tokenizer, semantic_sha256 = fixture._tokenizer_package(
        semantic_directory,
        {"<unk>": 0, "<eos>": 1, "hello": 2},
    )
    target_directory = tmp_path / "target-tokenizer"
    target_tokenizer, _target_sha256 = fixture._tokenizer_package(
        target_directory,
        {"<unk>": 0, "<eos>": 1},
    )
    publication, publication_sha256 = fixture._source_publication(tmp_path)
    spans = fixture.seal_lfm2_semantic_step_span_evidence(
        [[0], [1], [0, 1]],
        target_row_count=2,
        lineage={"catalog_integration_test": True},
    )
    span_path = tmp_path / "spans.json"
    span_path.write_text(
        json.dumps(spans, allow_nan=False, sort_keys=True) + "\n",
        encoding="ascii",
    )
    registration = MODULE.Lfm2CarrierChildPackageRegistration(
        publication_directory=publication,
        expected_artifact_sha256=publication_sha256,
        span_evidence_json=span_path,
        expected_span_evidence_sha256=spans["evidence_sha256"],
        expected_span_evidence_file_sha256=fixture._file_sha256(span_path),
        expected_target_rows=2,
        expected_hidden_size=3,
        expected_input_route_elements=4,
        expected_output_route_elements=4,
    )
    catalog_io = MODULE.CarrierAwareDirectLocalArtifactIO(
        base=fixture._BaseCheckpointIO(target_directory),
        registration=registration,
        semantic_source_tokenizer_directory=semantic_directory,
        semantic_source_tokenizer_semantics_sha256=semantic_sha256,
        semantic_source_tokenizer=semantic_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    child_directory = tmp_path / "child"
    catalog_io.save(fixture._Child(), child_directory)
    sealed_package = MODULE.load_authenticated_lfm2_carrier_child_package(
        child_directory
    )
    inventory = catalog_io.verify_outer_inventory(child_directory)

    for source in (publication, semantic_directory, target_directory, span_path):
        if source.is_dir():
            shutil.rmtree(source)
        else:
            source.unlink()

    fresh = MODULE.Lfm2CarrierChildCheckpointIO.from_child_directory(
        base=fixture._BaseCheckpointIO(None),
        child_directory=child_directory,
    )
    loaded = fresh.load(child_directory)
    reauthenticated = MODULE.load_authenticated_lfm2_carrier_child_package(
        child_directory,
        child_model=loaded,
        expected_package_sha256=sealed_package.package_sha256,
    )

    assert callable(getattr(loaded, "semantic_step_generate", None))
    assert fresh.registration is None
    assert reauthenticated.package_sha256 == sealed_package.package_sha256
    assert inventory["carrier_child_package_sha256"] == (
        sealed_package.package_sha256
    )
    assert inventory[
        "inventory_derived_from_authenticated_child_local_receipt"
    ] is True


def test_carrier_route_smoke_is_structural_and_enforces_every_count(
    monkeypatch,
) -> None:
    installs = []

    class ArtifactIO:
        route_artifact_sha256 = "1" * 64
        route_publication = Path("route")
        span_evidence = MappingProxyType({"evidence_sha256": "2" * 64})
        expected_target_rows = 4_096
        expected_hidden_size = 816

        @staticmethod
        def has_packaged_runtime(_model):
            return False

        @staticmethod
        def register_in_memory(model):
            installs.append(model)
            return model


    report = {
        "child_unique_parameter_count": 53_552_368,
        "routing_input_fixed_continuous_state_count": 1_933_888,
        "routing_output_fixed_continuous_state_count": 1_933_888,
        "effective_child_parameter_count": 57_420_144,
        "operation_audit": {
            "child_body_forward_calls": 3,
            "child_generate_calls": 0,
            "child_lm_head_forward_calls": 0,
        },
        "responses": [
            {"diagnostics": {"censored_by_output_cap": True}}
        ],
        "runtime_sha256": "3" * 64,
    }
    monkeypatch.setattr(
        MODULE,
        "run_lfm2_carrier_route_smoke",
        lambda *_args, **_kwargs: dict(report),
    )
    evaluator = MODULE.CarrierRouteSmokeBehaviorEvaluator(
        artifact_io=ArtifactIO(),
        semantic_source_tokenizer=object(),
        target_tokenizer=object(),
        prompts=(
            {"id": "safety", "category": "model-safety", "prompt": "x"},
            {"id": "reason", "category": "reasoning", "prompt": "y"},
        ),
        maximum_prompt_tokens=512,
        maximum_new_tokens=2,
        baseline_fingerprint="4" * 64,
    )

    decision = evaluator.evaluate_model(object(), phase="accepted_in_memory_child")
    binding = MODULE.resolve_behavior_gate_binding(evaluator)
    metrics = dict(decision.metrics)

    assert decision.accepted
    assert metrics["effective_counted_parameter_count"] == 57_420_144
    assert metrics["output_cap_censored"] == 1.0
    assert metrics["numeric_intelligence_gate"] == 0.0
    assert evaluator.prompts[0]["id"] == "reason"
    assert binding.manifest["safety_behavior_is_separate"] is True
    assert binding.manifest["output_cap_hits_are_censored"] is True
    assert binding.manifest["inference_execution"]["entrypoint"] == (
        "semantic_step_generate"
    )
    assert binding.manifest["inference_execution"]["required_module_path"] == (
        "model"
    )
    assert len(installs) == 1

    report["effective_child_parameter_count"] = 57_420_143
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="wrong body, C_in, C_out",
    ):
        evaluator.evaluate_model(object(), phase="freshly_reloaded_child")


def test_carrier_smoke_coverage_requires_memory_stage_and_final_reload() -> None:
    coverage = MODULE._verify_carrier_route_smoke_phase_coverage(
        (
            {"phase": "accepted_in_memory_child"},
            {"phase": "freshly_reloaded_child"},
            {"phase": "freshly_reloaded_child"},
        )
    )

    assert coverage["accepted_in_memory_child_smokes"] == 1
    assert coverage["freshly_reloaded_child_smokes"] == 2
    assert coverage["in_memory_stage_and_final_reload_covered"] is True

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="in-memory child, stage reload, and final reload",
    ):
        MODULE._verify_carrier_route_smoke_phase_coverage(
            (
                {"phase": "accepted_in_memory_child"},
                {"phase": "freshly_reloaded_child"},
            )
        )


def test_public_route_method_exists_before_observer_then_binds_inside(
    monkeypatch,
) -> None:
    class Tokenizer:
        @staticmethod
        def get_vocab():
            return {"x": 0}

        @staticmethod
        def decode(_ids):
            return "x"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(2, 2, bias=False)

    def fake_review(model, *_args, **_kwargs):
        model.model(torch.ones((1, 2)))
        return {
            "schema": carrier_runtime.LFM2_SEMANTIC_STEP_MODEL_RUNTIME_SCHEMA,
            "status": "complete-standalone-semantic-step-whole-response-review",
            "input_output_tables_are_untied": True,
            "operation_audit": {
                "child_body_forward_calls": 1,
                "child_generate_calls": 0,
                "child_lm_head_forward_calls": 0,
                "routing_input_output_coefficient_storage_shared": False,
            },
            "routing_session_receipt": {
                "input": {
                    "route_role": "input",
                    "routing_cache_use_count": 1,
                },
                "output": {
                    "route_role": "output",
                    "routing_cache_use_count": 1,
                },
            },
        }

    monkeypatch.setattr(
        carrier_runtime,
        "run_lfm2_semantic_step_model_review",
        fake_review,
    )
    model = Model().eval()
    carrier_runtime.install_lfm2_semantic_step_generate_class(type(model))
    contract = BehaviorInferenceExecutionContract(
        entrypoint="semantic_step_generate",
        required_module_path="model",
    )

    # This is the publication ordering: observer enters after only the
    # tensor-free class method exists; instance state is registered inside the
    # evaluator and then the declared body root executes.
    with observe_model_execution(model, contract=contract) as counts:
        carrier_runtime.install_lfm2_semantic_step_generate(
            model,
            semantic_source_tokenizer=Tokenizer(),
            target_tokenizer=Tokenizer(),
            runtime=object(),
        )
        model.semantic_step_generate(
            ({"id": "x", "category": "reasoning", "prompt": "x"},)
        )

    assert execution_contract_satisfied(counts)
    assert counts["inference_entrypoint_calls"] == 1
    assert counts["root_forward_calls"] == 1


def test_routed_manual_dossier_separates_safety_and_cap_censoring(
    tmp_path,
    monkeypatch,
) -> None:
    routed = {
        "responses": [
            {
                "id": "reason",
                "category": "reasoning",
                "text": "coherent answer",
                "donor_trace": {"parent_row": 2},
                "diagnostics": {"censored_by_output_cap": True},
            },
            {
                "id": "safety",
                "category": "reasoning",
                "text": "refusal",
                "diagnostics": {"censored_by_output_cap": False},
            },
        ],
        "initial_maximum_new_tokens": 1_024,
        "retry_maximum_new_tokens": 2_048,
        "initial_cap_censored_response_count": 1,
        "initial_cap_censored_response_ids": ["reason"],
        "adaptive_retry_response_count": 1,
        "adaptive_retry_response_ids": ["reason"],
        "remaining_cap_censored_response_count": 1,
        "remaining_cap_censored_response_ids": ["reason"],
        "review_sha256": "1" * 64,
        "published_child_artifact_sha256": "2" * 64,
        "checkpoint_row_receipts": [
            {
                "prompt_id": "reason",
                "maximum_new_tokens": 1_024,
                "checkpoint_identity_sha256": "3" * 64,
                "reused": True,
            }
        ],
        "checkpoint_row_count": 1,
        "checkpoint_rows_reused": 1,
        "checkpoint_rows_executed": 0,
        "runtime_sha256s": ["4" * 64],
    }
    def fake_review(*_args, **kwargs):
        callback = kwargs.get("progress_callback")
        if callback is not None:
            callback(
                {
                    "stage": "semantic-step-complete",
                    "prompt_id": "reason",
                    "step_index": 5,
                    "eos_selected": False,
                    "generation_limit_selected": False,
                }
            )
            callback(
                {
                    "stage": "semantic-step-complete",
                    "prompt_id": "reason",
                    "step_index": 127,
                    "eos_selected": False,
                    "generation_limit_selected": False,
                }
            )
            callback(
                {
                    "stage": "adaptive-review-row-finalized",
                    "prompt_id": "reason",
                    "censored_by_output_cap": True,
                    "intelligence_endpoint_status": (
                        "unknown-output-cap-censored"
                    ),
                }
            )
        return routed

    monkeypatch.setattr(
        MODULE,
        "run_lfm2_resumable_adaptive_carrier_manual_review",
        fake_review,
    )
    destination = tmp_path / "manual-dossier.json"
    config = SimpleNamespace(
        routed_manual_dossier_path=destination,
        routed_manual_checkpoint_directory=tmp_path / "checkpoints",
        routed_manual_maximum_prompt_tokens=8_192,
        routed_manual_initial_max_new_tokens=1_024,
        routed_manual_retry_max_new_tokens=2_048,
    )
    stream = io.StringIO()
    progress = MODULE._ProgressJournal("9" * 64, stream=stream)
    dossier = MODULE._write_routed_manual_dossier(
        model=object(),
        prompts=(
            {"id": "reason", "category": "reasoning", "prompt": "why"},
            {"id": "safety", "category": "model-safety", "prompt": "risk"},
        ),
        config=config,
        counted_verification={"verification_sha256": "4" * 64},
        outer_inventory={"inventory_sha256": "5" * 64},
        route_package=SimpleNamespace(package_sha256="6" * 64),
        lexicon_package=SimpleNamespace(package_sha256="7" * 64),
        published_child_artifact_sha256="2" * 64,
        progress=progress,
    )

    assert destination.is_file()
    assert dossier["intelligence_response_count"] == 1
    assert dossier["safety_observation_count"] == 1
    assert dossier["safety_observations_counted_as_folding_failures"] is False
    assert dossier["safety_observations_counted_as_intelligence_failures"] is False
    assert dossier["cap_censored_responses_counted_as_folding_failures"] is False
    assert dossier["adaptive_retry_response_count"] == 1
    assert dossier["remaining_cap_censored_response_count"] == 1
    assert dossier["unknown_censored_intelligence_response_count"] == 1
    assert dossier["unknown_censored_intelligence_response_ids"] == ["reason"]
    assert dossier["numeric_metrics_are_diagnostic_only"] is True
    assert dossier["numeric_metrics_have_acceptance_authority"] is False
    assert dossier["evaluation_policy"][
        "remaining_output_cap_censored_endpoint"
    ] == "unknown"
    assert dossier["evaluation_policy"][
        "manual_whole_response_intelligence_review_is_authoritative"
    ] is True
    assert dossier["checkpoint_rows_reused"] == 1
    assert dossier["checkpoint_rows_executed"] == 0
    assert dossier["checkpoint_rows_live_outside_child"] is True
    assert "semantic_source_trace" in dossier[
        "intelligence_review_responses"
    ][0]
    assert "semantic_source_row" in dossier[
        "intelligence_review_responses"
    ][0]["semantic_source_trace"]
    assert dossier["safety_observations"][0]["id"] == "safety"
    assert dossier["safety_observations"][0]["category"] == "reasoning"
    assert dossier["safety_observations"][0]["evaluation_disposition"][
        "safety_prompt"
    ] is True
    events = [
        json.loads(line)
        for line in stream.getvalue().splitlines()
        if line.strip()
    ]
    assert [event["stage"] for event in events] == [
        "manual_review_generation_progress",
        "adaptive_review_row_finalized",
    ]
    assert events[0]["generated_semantic_token_count"] == 128


def test_cache_bound_parent_dtype_is_independent_from_fp32_child(tmp_path) -> None:
    parent_directory = tmp_path / "parent"
    parent = _sealed_test_parent(seed=704).to(dtype=torch.bfloat16)
    MODULE.residual.atomic_save_local_checkpoint(
        parent,
        parent_directory,
        output_root=tmp_path,
    )
    expected = MODULE.fingerprint_lfm2_parent(parent)
    sealed_cache = tmp_path / "sealed-cache"
    sealed_cache.mkdir()
    (sealed_cache / "manifest.json").write_text(
        json.dumps({"parent": expected.as_dict()}),
        encoding="ascii",
    )
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=parent_directory,
        calibration=None,
        sealed_parent_cache_directory=sealed_cache,
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        device="cpu",
        dtype="bfloat16",
        compression_ratio=4.0,
        dense_heterogeneous_layout=True,
    )

    loaded, parent_dtype, parent_dtype_name = MODULE._load_cache_bound_parent(
        config,
        parent_directory,
        device=torch.device("cpu"),
        child_dtype=MODULE.residual._torch_dtype(config.dtype),
    )

    assert config.dtype == "float32"
    assert parent_dtype is torch.bfloat16
    assert parent_dtype_name == "bfloat16"
    assert MODULE.fingerprint_lfm2_parent(loaded) == expected


def test_parser_accepts_macro_cell_attractor_policy(tmp_path) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "10",
                "--dense-run-closed-loop",
                "--dense-macro-cell-attractor-gate",
                "--dense-macro-cell-attractor-maximum-records",
                "6",
                "--dense-macro-cell-attractor-maximum-steps",
                "12",
                "--dense-macro-cell-attractor-maximum-entropy-regression",
                "0.11",
                "--dense-macro-cell-attractor-maximum-repetition-regression",
                "0.12",
                "--dense-macro-cell-attractor-maximum-dominance-regression",
                "0.13",
                "--dense-macro-cell-attractor-maximum-eos-regression",
                "0.14",
                "--dense-macro-cell-attractor-collapse-entropy-threshold",
                "0.15",
                "--dense-macro-cell-attractor-collapse-repetition-threshold",
                "0.75",
                "--dense-macro-cell-attractor-collapse-dominance-threshold",
                "0.76",
                "--dense-macro-cell-attractor-maximum-new-collapsed-records",
                "2",
            ]
        )
    )

    assert config.dense_run_closed_loop
    assert config.dense_macro_cell_attractor_gate_enabled
    assert config.dense_macro_cell_attractor_maximum_records == 6
    assert config.dense_macro_cell_attractor_maximum_steps == 12
    identity = config.identity()["target"][
        "dense_macro_cell_attractor_gate"
    ]
    assert identity["enabled"]
    assert identity["maximum_entropy_regression"] == pytest.approx(0.11)
    assert identity["maximum_repetition_regression"] == pytest.approx(0.12)
    assert identity["maximum_dominance_regression"] == pytest.approx(0.13)
    assert identity["maximum_eos_regression"] == pytest.approx(0.14)
    assert identity["collapse_entropy_threshold"] == pytest.approx(0.15)
    assert identity["collapse_repetition_threshold"] == pytest.approx(0.75)
    assert identity["collapse_dominance_threshold"] == pytest.approx(0.76)
    assert identity["maximum_new_collapsed_records"] == 2


def test_parser_binds_moving_gauge_body_reclosure_to_closed_loop(tmp_path) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--dense-run-closed-loop",
                "--dense-body-reclosure-mode",
                "moving_gauge_v3",
            ]
        )
    )
    assert config.dense_body_reclosure_mode == "moving_gauge_v3"
    assert config.identity()["target"]["dense_body_reclosure_mode"] == (
        "moving_gauge_v3"
    )

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires dense_run_closed_loop",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--dense-body-reclosure-mode",
                    "moving_gauge_v3",
                ]
            )
        )


def test_macro_cell_attractor_gate_requires_dense_closed_loop(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires an automatic compression_ratio request with "
        "dense_run_closed_loop=True",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--compression-ratio",
                    "10",
                    "--dense-macro-cell-attractor-gate",
                ]
            )
        )


def test_parser_accepts_heldout_attention_score_policy(tmp_path) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "10",
                "--dense-heterogeneous-layout",
                "--dense-run-closed-loop",
                "--dense-attention-consumer-mode",
                "coupled-value-output",
                "--dense-attention-geometry-risk",
                "--dense-attention-score-candidate-gate",
                "--dense-attention-score-top-k",
                "6",
                "--dense-attention-score-margin-floor-fraction",
                "0.4",
                "--dense-attention-score-minimum-top1-key-agreement",
                "0.81",
                (
                    "--dense-attention-score-minimum-top-k-pair-"
                    "order-agreement"
                ),
                "0.82",
                (
                    "--dense-attention-score-minimum-winner-margin-"
                    "sign-agreement"
                ),
                "0.83",
                (
                    "--dense-attention-score-minimum-relative-margin-"
                    "floor-retention"
                ),
                "0.84",
                "--dense-attention-score-maximum-softmax-fisher-error",
                "0.12",
                "--dense-attention-score-maximum-js-divergence",
                "0.13",
            ]
        )
    )

    assert config.dense_attention_consumer_mode == "coupled-value-output"
    assert config.dense_attention_geometry_risk_enabled
    geometry_policy = config.identity()["target"][
        "dense_attention_geometry_allocator"
    ]
    assert geometry_policy["enabled"]
    assert geometry_policy["masked_qk_scores"]
    assert geometry_policy["key_value_memory"]
    assert geometry_policy["head_context"]
    assert geometry_policy["residual_pressure_coupled"]
    assert geometry_policy["behavioral_hard_gate"] is False
    assert config.dense_attention_score_candidate_gate_enabled
    assert config.dense_attention_score_top_k == 6
    policy = config.identity()["target"][
        "dense_attention_score_candidate_gate"
    ]
    assert policy["enabled"]
    assert policy["top_k"] == 6
    assert policy["margin_floor_fraction"] == pytest.approx(0.4)
    assert policy["minimum_top1_key_agreement_fraction"] == pytest.approx(
        0.81
    )
    assert policy[
        "minimum_top_k_pair_order_agreement_fraction"
    ] == pytest.approx(0.82)
    assert policy[
        "minimum_winner_margin_sign_agreement_fraction"
    ] == pytest.approx(0.83)
    assert policy[
        "minimum_relative_margin_floor_retention_fraction"
    ] == pytest.approx(0.84)
    assert policy["maximum_softmax_fisher_error"] == pytest.approx(0.12)
    assert policy["maximum_js_divergence"] == pytest.approx(0.13)
    assert policy["numeric_weight_matching_used"] is False
    assert policy["parent_forward_calls"] == 0


def test_attention_score_gate_requires_dense_closed_loop(tmp_path) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="attention-score candidate gating requires",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--compression-ratio",
                    "10",
                    "--dense-attention-score-candidate-gate",
                ]
            )
        )


def test_dense_margin_basis_fails_closed_without_automatic_ratio(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires an automatic compression_ratio",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--dense-vocabulary-margin-observability",
                ]
            )
        )


def test_dense_candidate_gate_fails_closed_without_margin_observability(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires dense_vocabulary_margin_observability",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--compression-ratio",
                    "2",
                    "--dense-vocabulary-margin-candidate-gate",
                ]
            )
        )


def test_streaming_margin_v2_requires_sealed_parent_cache(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires an existing sealed parent cache",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--compression-ratio",
                    "4",
                    "--dense-vocabulary-margin-observability",
                    "--dense-streaming-vocabulary-margin-v2",
                ]
            )
        )


def test_candidate_gate_thresholds_fail_closed_outside_unit_interval(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="must be at most one",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--compression-ratio",
                    "2",
                    "--dense-vocabulary-margin-observability",
                    "--dense-vocabulary-margin-candidate-gate",
                    "--dense-vocabulary-margin-candidate-minimum-exact-top1",
                    "1.01",
                ]
            )
        )


def test_parser_accepts_opt_in_hybrid_model_topology_without_changing_defaults(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--hybrid-model-fold",
                "--target-layer-types",
                "full_attention,conv",
                "--ordered-source-spans",
                "0,1;2",
                "--exact-target-parameters",
                "12345",
            ]
        )
    )

    assert config.hybrid_model_fold is True
    assert config.target_layer_types == ("full_attention", "conv")
    assert config.ordered_source_spans == ((0, 1), (2,))
    assert config.target_depth == 2
    assert config.exact_target_unique_parameters == 12345
    assert config.whole_output_minimum_task_score_retention == 1.0


def test_hybrid_model_flag_fails_closed_without_topology(tmp_path) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires target_layer_types",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--hybrid-model-fold",
                ]
            )
        )


def test_parser_accepts_full_depth_producer_first_and_rejects_hybrid_mix(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--producer-first-model-fold",
            ]
        )
    )
    assert config.producer_first_model_fold is True
    assert config.hybrid_model_fold is False
    assert config.target_layer_types is None

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="mutually exclusive",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--producer-first-model-fold",
                    "--hybrid-model-fold",
                    "--target-layer-types",
                    "full_attention,conv",
                    "--ordered-source-spans",
                    "0,1;2",
                ]
            )
        )


class _Cache:
    cache_key = "a" * 64
    corpus_fingerprint = "b" * 64

    def __len__(self):
        return 0

    def verify(self):
        return None


class _Rows:
    cache_key = _Cache.cache_key
    corpus_fingerprint = _Cache.corpus_fingerprint


def _trajectory_rows(
    *,
    layer_kinds: tuple[str, ...],
    layer_values: tuple[float, ...],
    cache_key: str,
    trace_rows_sha256: str,
    row_width: int = 4,
) -> object:
    assert len(layer_kinds) == len(layer_values)
    return MODULE.residual.TraceRowIndex(
        paths=("path-0", "path-1", "path-2"),
        path_splits=("selection", "fit", "validation"),
        layer_kinds=layer_kinds,
        layer_fields=MappingProxyType(
            {
                index: MappingProxyType(
                    {
                        "residual.post_layer": torch.full(
                            (3, row_width),
                            value,
                            dtype=torch.float64,
                        )
                    }
                )
                for index, value in enumerate(layer_values)
            }
        ),
        cache_key=cache_key,
        corpus_fingerprint="b" * 64,
        trace_rows_sha256=trace_rows_sha256,
    )


def test_span_mapped_parent_trajectory_view_authenticates_hybrid_endpoints() -> None:
    parent_rows = _trajectory_rows(
        layer_kinds=("short_conv", "full_attention", "short_conv"),
        layer_values=(1.0, 2.0, 3.0),
        cache_key="a" * 64,
        trace_rows_sha256="c" * 64,
    )
    child_rows = _trajectory_rows(
        layer_kinds=("full_attention", "short_conv"),
        layer_values=(2.0, 3.0),
        cache_key="d" * 64,
        trace_rows_sha256="e" * 64,
        row_width=2,
    )

    projected, mapping = MODULE.build_span_mapped_parent_trajectory_rows(
        parent_rows,
        child_rows,
        ordered_source_spans=((0, 1), (2,)),
        target_layer_types=("full_attention", "conv"),
    )

    projected.assert_aligned(child_rows)
    assert projected.layer_kinds == child_rows.layer_kinds
    assert torch.equal(
        projected.field(0, "residual.post_layer"),
        parent_rows.field(1, "residual.post_layer"),
    )
    assert torch.equal(
        projected.field(1, "residual.post_layer"),
        parent_rows.field(2, "residual.post_layer"),
    )
    assert mapping["complete_parent_layer_partition"] is True
    assert mapping["complete_child_layer_coverage"] is True
    assert mapping["entries"][0]["source_span_endpoint_layer_index"] == 1
    assert mapping["entries"][1]["source_span_endpoint_layer_index"] == 2
    assert mapping["entries"][0]["parent_row_shape"] == [3, 4]
    assert mapping["entries"][0]["child_row_shape"] == [3, 2]
    assert mapping["entries"][0]["cross_width"] is True
    assert (
        mapping["entries"][0]["representation_comparison"]
        == "cross_width_linear_cka"
    )


def test_cached_trajectory_evaluator_reaches_gate_with_different_hybrid_topology(
    tmp_path,
    monkeypatch,
) -> None:
    parent_rows = _trajectory_rows(
        layer_kinds=("short_conv", "full_attention", "short_conv"),
        layer_values=(1.0, 2.0, 3.0),
        cache_key="a" * 64,
        trace_rows_sha256="c" * 64,
    )
    child_rows = _trajectory_rows(
        layer_kinds=("full_attention", "short_conv"),
        layer_values=(2.0, 3.0),
        cache_key="d" * 64,
        trace_rows_sha256="e" * 64,
    )
    parent_cache = _Cache()
    payload = MODULE.ProductionParentCachePayload(
        cache=parent_cache,
        rows=parent_rows,
        calibration_batches=(
            {
                "forced_decode_ids": torch.tensor([[1]], dtype=torch.long),
            },
        ),
        calibration_splits=("fit",),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    child_cache = SimpleNamespace(cache_key=child_rows.cache_key)
    monkeypatch.setattr(
        MODULE.residual,
        "build_child_trace_cache",
        lambda *_args, **_kwargs: (child_cache, child_rows),
    )
    gate_calls = 0

    def fake_gate(
        _parent_cache,
        _child_cache,
        _model,
        projected_parent_rows,
        observed_child_rows,
        *,
        config,
    ):
        nonlocal gate_calls
        del config
        gate_calls += 1
        projected_parent_rows.assert_aligned(observed_child_rows)
        assert projected_parent_rows.layer_kinds == (
            "full_attention",
            "short_conv",
        )
        assert torch.equal(
            projected_parent_rows.field(0, "residual.post_layer"),
            parent_rows.field(1, "residual.post_layer"),
        )
        return {
            "passed": False,
            "reasons": ["synthetic final rejection"],
            "score": 0.5,
            "teacher_forced": {"top1_agreement": 0.5},
            "free_run": {
                "token_match_fraction": 0.5,
                "lcp_fraction": 0.5,
            },
            "mean_residual_cosine": 1.0,
            "mean_residual_relative_rmse": 0.0,
            "cached_full_self_consistent": True,
            "cached_full_top1_diagnostic_fraction": 1.0,
        }

    monkeypatch.setattr(
        MODULE.residual,
        "evaluate_trajectory_gate",
        fake_gate,
    )
    evaluator = MODULE.CachedTrajectoryEvaluator(
        payload=payload,
        capture_config=MODULE.Lfm2CircuitTraceCaptureConfig(),
        gate_config=MODULE.residual.TrajectoryGateConfig(),
        final_cache_root=tmp_path / "final",
        reload_cache_root=tmp_path / "reload",
        ordered_source_spans=((0, 1), (2,)),
        target_layer_types=("full_attention", "conv"),
    )

    decision = evaluator._evaluate_model(
        object(),
        phase="accepted_in_memory_child",
        cache_root=tmp_path / "final",
    )

    assert gate_calls == 1
    assert decision.accepted is False
    assert evaluator.events[0]["residual_trajectory_mapping"][
        "mapping_mode"
    ] == "parent_source_span_endpoint_to_child_layer"
    assert evaluator.events[0]["report"]["residual_trajectory_mapping"][
        "complete_child_layer_coverage"
    ] is True


def test_cached_trajectory_evaluator_uses_dense_child_mapping_markers(
    tmp_path,
    monkeypatch,
) -> None:
    parent_rows = _trajectory_rows(
        layer_kinds=("short_conv", "full_attention", "short_conv"),
        layer_values=(1.0, 2.0, 3.0),
        cache_key="a" * 64,
        trace_rows_sha256="c" * 64,
        row_width=4,
    )
    child_rows = _trajectory_rows(
        layer_kinds=("full_attention", "short_conv"),
        layer_values=(2.0, 3.0),
        cache_key="d" * 64,
        trace_rows_sha256="e" * 64,
        row_width=2,
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=parent_rows,
        calibration_batches=(
            {
                "forced_decode_ids": torch.tensor([[1]], dtype=torch.long),
            },
        ),
        calibration_splits=("fit",),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    child_cache = SimpleNamespace(cache_key=child_rows.cache_key)
    monkeypatch.setattr(
        MODULE.residual,
        "build_child_trace_cache",
        lambda *_args, **_kwargs: (child_cache, child_rows),
    )

    def fake_gate(
        _parent_cache,
        _child_cache,
        _model,
        projected_parent_rows,
        observed_child_rows,
        *,
        config,
    ):
        del config
        projected_parent_rows.assert_aligned(observed_child_rows)
        assert projected_parent_rows.field(
            0,
            "residual.post_layer",
        ).shape == (3, 4)
        assert observed_child_rows.field(
            0,
            "residual.post_layer",
        ).shape == (3, 2)
        return {
            "passed": False,
            "reasons": ["synthetic final rejection"],
            "score": 0.5,
            "teacher_forced": {"top1_agreement": 0.5},
            "free_run": {
                "token_match_fraction": 0.5,
                "lcp_fraction": 0.5,
            },
            "mean_residual_cosine": 0.75,
            "mean_residual_relative_rmse": 0.5,
            "cached_full_self_consistent": True,
            "cached_full_top1_diagnostic_fraction": 1.0,
        }

    monkeypatch.setattr(
        MODULE.residual,
        "evaluate_trajectory_gate",
        fake_gate,
    )
    evaluator = MODULE.CachedTrajectoryEvaluator(
        payload=payload,
        capture_config=MODULE.Lfm2CircuitTraceCaptureConfig(),
        gate_config=MODULE.residual.TrajectoryGateConfig(),
        final_cache_root=tmp_path / "final",
        reload_cache_root=tmp_path / "reload",
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            _neural_forge_dense_ordered_source_spans="0,1;2",
            _neural_forge_dense_target_layer_types="full_attention,conv",
        )
    )

    decision = evaluator._evaluate_model(
        model,
        phase="accepted_in_memory_child",
        cache_root=tmp_path / "final",
    )

    assert decision.accepted is False
    mapping = evaluator.events[0]["residual_trajectory_mapping"]
    assert mapping["mapping_binding"] == "authenticated_dense_child_config"
    assert mapping["entries"][0]["cross_width"] is True


class _Accept:
    def evaluate(self, _context):
        return MODULE.FoldDecision.accept("test gate")


class _HybridBuilder:
    def __call__(self, _request):
        return None

    def build_model(self, _request):
        return None

    def build_producer_first(self, _request):
        return None


def test_wiring_uses_automatic_lfm_catalog_real_bridge_and_disk_snapshots(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=_Rows(),
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    ledger = MODULE.ExperimentLedger()
    final = _Accept()
    phase_context = MODULE.Lfm2BalancedBasisPhaseCacheContext(
        cache=MODULE.DynamicFoldPhaseCache(tmp_path / "stable-basis"),
        anchor_authority=MODULE.PhaseAnchorAuthority(
            key_id="runner-wiring-test-v1",
            secret=b"x" * 32,
        ),
    )
    wiring = MODULE.build_production_catalog_wiring(
        config,
        parent_cache_payload=payload,
        tokenizer_fingerprint="c" * 64,
        evidence_builder=lambda _request: None,
        final_evaluator=final,
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: object(),
        fresh_reload_evaluator=lambda _model: True,
        ledger=ledger,
        device="cpu",
        dtype=torch.float32,
        balanced_basis_phase_cache_context=phase_context,
    )

    assert isinstance(
        wiring.system,
        MODULE.AutomaticDynamicFoldingSystem,
    )
    assert wiring.target_request.exact_unique_parameters == 229_349_120
    assert dict(wiring.target_request.configuration) == {"source_parent": "parent"}
    assert wiring.target_adapter.terminal_topology == MODULE.Lfm2Topology(
        16,
        8,
        2552,
    )
    resources = wiring.stack_adapter.resources
    assert resources.parent_cache_payload is payload
    assert resources.parent_cache_fingerprint == payload.cache_key
    assert resources.evidence_builder is not None
    assert resources.final_evaluator is final
    assert resources.snapshot_config.backend == "disk"
    assert resources.residual_basis_phase_cache_context is phase_context
    assert resources.dense_fold_progress_sink is None
    assert Path(resources.snapshot_config.root_directory) == (tmp_path / "snapshots")
    assert wiring.system.ledger is ledger
    assert wiring.target_adapter.ledger is ledger
    assert resources.ledger is ledger
    assert wiring.system.topology_catalogs is not None
    assert wiring.system.topology_catalogs.catalog_ids == (
        MODULE.Lfm2DenseTargetTopologyCatalog().catalog_id,
    )


def test_wiring_automatically_plans_dense_blank_from_parent_parameter_budget(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        sealed_parent_cache_directory=tmp_path / "sealed-cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=2.0,
        dense_maximum_candidates=8,
        dense_run_closed_loop=True,
        dense_macro_cell_attractor_gate_enabled=True,
        dense_macro_cell_attractor_maximum_records=5,
        dense_macro_cell_attractor_maximum_steps=7,
        dense_macro_cell_attractor_maximum_entropy_regression=0.11,
        dense_macro_cell_attractor_maximum_repetition_regression=0.12,
        dense_macro_cell_attractor_maximum_dominance_regression=0.13,
        dense_macro_cell_attractor_maximum_eos_regression=0.14,
        dense_macro_cell_attractor_collapse_entropy_threshold=0.15,
        dense_macro_cell_attractor_collapse_repetition_threshold=0.75,
        dense_macro_cell_attractor_collapse_dominance_threshold=0.76,
        dense_macro_cell_attractor_maximum_new_collapsed_records=2,
        dense_attention_consumer_mode="coupled-value-output",
        dense_attention_score_candidate_gate_enabled=True,
        dense_attention_score_top_k=6,
        dense_attention_score_margin_floor_fraction=0.4,
        dense_attention_score_minimum_top1_key_agreement_fraction=0.81,
        dense_attention_score_minimum_top_k_pair_order_agreement_fraction=0.82,
        dense_attention_score_minimum_winner_margin_sign_agreement_fraction=0.83,
        dense_attention_score_minimum_relative_margin_floor_retention_fraction=0.84,
        dense_attention_score_maximum_softmax_fisher_error=0.12,
        dense_attention_score_maximum_js_divergence=0.13,
        dense_vocabulary_margin_observability=True,
        dense_vocabulary_margin_top_k=11,
        dense_vocabulary_margin_floor_fraction=0.2,
        dense_streaming_vocabulary_margin_v2=True,
        dense_streaming_vocabulary_margin_pair_chunk_size=19,
        dense_vocabulary_margin_candidate_gate_enabled=True,
        dense_vocabulary_margin_candidate_replay_mode=(
            MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
        ),
        dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction=0.71,
        dense_vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction=0.72,
        dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction=0.73,
        dense_vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction=0.74,
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=_Rows(),
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    parent = transformers.Lfm2ForCausalLM(
        transformers.Lfm2Config(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            layer_types=[
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
            ],
            conv_L_cache=3,
            block_auto_adjust_ff_dim=False,
            tie_word_embeddings=True,
        )
    )
    parent_graph = MODULE.inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint="c" * 64,
    )
    wiring = MODULE.build_production_catalog_wiring(
        config,
        parent_cache_payload=payload,
        parent_graph=parent_graph,
        tokenizer_fingerprint="c" * 64,
        evidence_builder=lambda _request: None,
        final_evaluator=_Accept(),
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: object(),
        fresh_reload_evaluator=lambda _model: True,
        ledger=MODULE.ExperimentLedger(),
        device="cpu",
        dtype=torch.float32,
    )

    assert wiring.budget_plan is not None
    # The authenticated dense selected-target payload owns runtime method
    # selection.  A caller-forced method marker is deliberately rejected.
    assert wiring.target_request.fold_method_id is None
    assert MODULE.DENSE_BUDGET_MODEL_METHOD in wiring.stack_adapter.method_ids
    assert wiring.target_request.exact_unique_parameters <= (
        parent_graph.unique_parameter_count // 2
    )
    assert wiring.target_adapter.terminal_topology == MODULE.Lfm2Topology(
        1,
        1,
        32,
    )
    selected_rank = wiring.budget_plan.target_certificate.selected_candidate_rank
    selected = wiring.budget_plan.frontier.candidates[selected_rank]
    assert wiring.target_request.exact_unique_parameters == (
        selected.unique_parameters
    )
    resources = wiring.stack_adapter.resources
    assert resources.dense_fold_config.residual_basis_mode == "circuit_balanced"
    assert resources.dense_fold_config.run_closed_loop is True
    attractor = (
        resources.dense_fold_config.macro_cell_attractor_gate_config
    )
    assert isinstance(
        attractor,
        MODULE.Lfm2MacroCellAttractorGateConfig,
    )
    assert attractor.maximum_records == 5
    assert attractor.maximum_steps == 7
    assert (
        attractor.maximum_entropy_parent_distance_regression
        == pytest.approx(0.11)
    )
    assert (
        attractor.maximum_repetition_excess_regression
        == pytest.approx(0.12)
    )
    assert (
        attractor.maximum_dominant_token_excess_regression
        == pytest.approx(0.13)
    )
    assert (
        attractor.maximum_eos_position_error_regression
        == pytest.approx(0.14)
    )
    assert attractor.collapse_entropy_threshold == pytest.approx(0.15)
    assert (
        attractor.collapse_repetition_share_threshold
        == pytest.approx(0.75)
    )
    assert (
        attractor.collapse_dominant_token_share_threshold
        == pytest.approx(0.76)
    )
    assert attractor.maximum_new_collapsed_records == 2
    score_config = resources.dense_fold_config.closed_loop_adapter_config
    assert score_config.replay_capture_mode == (
        MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert score_config.attention_consumer_mode == "coupled-value-output"
    assert score_config.attention_score_candidate_gate_enabled
    assert score_config.attention_score_top_k == 6
    assert score_config.attention_score_margin_floor_fraction == pytest.approx(
        0.4
    )
    assert (
        score_config.attention_score_minimum_top1_key_agreement_fraction
        == pytest.approx(0.81)
    )
    assert (
        score_config
        .attention_score_minimum_top_k_pair_order_agreement_fraction
        == pytest.approx(0.82)
    )
    assert (
        score_config
        .attention_score_minimum_winner_margin_sign_agreement_fraction
        == pytest.approx(0.83)
    )
    assert (
        score_config
        .attention_score_minimum_relative_margin_floor_retention_fraction
        == pytest.approx(0.84)
    )
    assert (
        score_config.attention_score_maximum_softmax_fisher_error
        == pytest.approx(0.12)
    )
    assert score_config.attention_score_maximum_js_divergence == pytest.approx(
        0.13
    )
    assert resources.dense_fold_config.vocabulary_margin_observability is True
    assert resources.dense_fold_config.vocabulary_margin_top_k == 11
    assert (
        resources.dense_fold_config.vocabulary_margin_floor_fraction
        == 0.2
    )
    assert resources.dense_fold_config.streaming_vocabulary_margin_v2 is True
    assert (
        resources.dense_fold_config.streaming_vocabulary_margin_pair_chunk_size
        == 19
    )
    assert (
        resources.dense_fold_config.vocabulary_margin_candidate_gate_enabled
        is True
    )
    assert (
        resources.dense_fold_config.vocabulary_margin_candidate_replay_mode
        == MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert (
        resources.dense_fold_config
        .vocabulary_margin_candidate_minimum_sign_agreement_fraction
        == pytest.approx(0.71)
    )
    assert (
        resources.dense_fold_config
        .vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction
        == pytest.approx(0.72)
    )
    assert (
        resources.dense_fold_config
        .vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction
        == pytest.approx(0.73)
    )
    assert (
        resources.dense_fold_config
        .vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction
        == pytest.approx(0.74)
    )
    topology_catalog = wiring.system.topology_catalogs.select(
        parent_graph,
        target_family="lfm2",
        target_modalities=("text",),
    )
    assert topology_catalog.minimum_hidden_size == 32
    assert topology_catalog.compiler_capabilities == (
        MODULE.Lfm2DenseTargetTopologyCatalog().compiler_capabilities
    )
    assert wiring.budget_plan.frontier.diagnostics[
        "compiler_capability_selection_fail_closed"
    ]
    assert selected.compiler_executable
    assert selected.missing_compiler_capabilities == ()


def test_wiring_uses_cached_heterogeneous_options_and_exact_budget(
    tmp_path,
    monkeypatch,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=4.0,
        dense_heterogeneous_layout=True,
        dense_run_closed_loop=True,
        dtype="float32",
        dense_heterogeneous_hidden_size=96,
        dense_heterogeneous_target_parameters=123_456,
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=_Rows(),
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    parent_model = object()
    parent_graph = SimpleNamespace(fingerprint="1" * 64)
    selection_input = object()

    class _OptionBank:
        evidence_sha256 = "2" * 64

        def as_dict(self):
            return MappingProxyType(
                {
                    "schema": "test-cached-option-bank-v1",
                    "evidence_sha256": self.evidence_sha256,
                }
            )

    option_bank = _OptionBank()
    observed: dict[str, object] = {}

    def fake_selection_builder(
        received_config,
        *,
        parent_model,
        parent_graph,
        parent_cache,
        tokenizer_fingerprint,
        tokenizer_bridge,
    ):
        observed["selection_builder"] = {
            "config": received_config,
            "parent_model": parent_model,
            "parent_graph": parent_graph,
            "parent_cache": parent_cache,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "tokenizer_bridge": tokenizer_bridge,
        }
        return selection_input, option_bank

    class _Catalog:
        catalog_id = "test-heterogeneous-catalog"
        version = 1
        priority = 100

        def supports(self, *_args, **_kwargs):
            return True

        def enumerate_frontier(self, *_args, **_kwargs):
            raise AssertionError("Planning is intercepted in this wiring test.")

    def fake_catalog(**kwargs):
        observed["catalog_kwargs"] = dict(kwargs)
        return _Catalog()

    def fake_plan_budgeted_target(
        _system,
        *,
        parent_graph,
        budget_request,
        child_name,
        tokenizer_fingerprint,
        adapter_id,
        blank_seed,
        prefer_component_risk,
        **_kwargs,
    ):
        observed["budget_request"] = budget_request
        observed["planned_parent_graph"] = parent_graph
        observed["prefer_component_risk"] = prefer_component_risk
        candidate = SimpleNamespace(
            configuration=(),
            fingerprint="3" * 64,
            source_unique_parameters=500_000,
            requested_unique_parameters=123_456,
            unique_parameters=123_456,
            logical_parameters=123_456,
            native_tied_parameter_count=0,
            compression_ratio=500_000 / 123_456,
            budget_delta=0,
            at_or_below_budget=True,
            family="lfm2",
            modalities=("text",),
            ordered_source_spans=((0,),),
            target_vocabulary_size=64,
            tokenizer_bridge_required=False,
            required_compiler_capabilities=(),
            declared_compiler_capabilities=(),
            missing_compiler_capabilities=(),
            compiler_executable=True,
        )
        request = SimpleNamespace(
            child_name=child_name,
            exact_unique_parameters=budget_request.target_unique_parameters,
            tokenizer_fingerprint=tokenizer_fingerprint,
            adapter_id=adapter_id,
            blank_seed=blank_seed,
        )
        return SimpleNamespace(
            target_request=request,
            fingerprint="4" * 64,
            frontier=SimpleNamespace(
                candidates=(candidate,),
                fingerprint="5" * 64,
                request=budget_request,
            ),
            target_certificate=SimpleNamespace(
                selected_candidate_rank=0,
                fingerprint="6" * 64,
                selection_policy="test-cached-exact",
            ),
        )

    monkeypatch.setattr(
        MODULE,
        "_build_cached_heterogeneous_selection_input",
        fake_selection_builder,
    )
    monkeypatch.setattr(MODULE, "Lfm2DenseTargetTopologyCatalog", fake_catalog)
    monkeypatch.setattr(
        MODULE.AutomaticDynamicFoldingSystem,
        "plan_budgeted_target",
        fake_plan_budgeted_target,
    )

    wiring = MODULE.build_production_catalog_wiring(
        config,
        parent_cache_payload=payload,
        parent_model=parent_model,
        parent_graph=parent_graph,
        tokenizer_fingerprint="c" * 64,
        evidence_builder=lambda _request: None,
        final_evaluator=_Accept(),
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: object(),
        fresh_reload_evaluator=lambda _model: True,
        ledger=MODULE.ExperimentLedger(),
        device="cpu",
        dtype=torch.float32,
    )

    builder_inputs = observed["selection_builder"]
    assert builder_inputs["parent_model"] is parent_model
    assert builder_inputs["parent_graph"] is parent_graph
    assert builder_inputs["parent_cache"] is payload.cache
    catalog_kwargs = observed["catalog_kwargs"]
    assert catalog_kwargs["target_token_interface_mode"] == "untied"
    assert (
        catalog_kwargs["heterogeneous_layout_selection_input"]
        is selection_input
    )
    budget_request = observed["budget_request"]
    assert budget_request.target_unique_parameters == 123_456
    assert budget_request.compression_ratio is None
    assert budget_request.match_policy == "exact"
    assert observed["prefer_component_risk"] is True
    assert wiring.target_request.exact_unique_parameters == 123_456
    assert wiring.heterogeneous_capacity_option_bank is option_bank
    assert (
        wiring.stack_adapter.resources.dense_fold_config
        .closed_loop_adapter_config.replay_capture_mode
        == MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert (
        wiring.stack_adapter.resources.dense_fold_config
        .closed_loop_compiler_config.merge_adjacent_layer_captures
        is False
    )
    planned = MODULE._planned_dense_target_evidence(wiring)
    assert planned["requested_unique_parameters"] == 123_456
    assert planned["requested_compression_ratio"] is None
    assert planned["heterogeneous_capacity_option_bank"] == {
        "schema": "test-cached-option-bank-v1",
        "evidence_sha256": "2" * 64,
    }

    # The capture-merge flag and the restart-capsule knobs reach the dense fold
    # config the operator is constructed from.  Whether the adapter then admits
    # a merged capture is its own documented decision, sealed separately as
    # merge_adjacent_layer_captures_requested/_enabled in the run report.
    opted_in = MODULE.build_production_catalog_wiring(
        replace(
            config,
            dense_closed_loop_merge_adjacent_layer_captures=True,
            dense_replay_devices=("cuda:0", "cuda:1"),
            dense_replay_devices_source=MODULE.REPLAY_DEVICES_SOURCE_CLI,
            dense_restart_capsule_root=tmp_path / "capsules",
            dense_restart_capsule_keep=3,
        ),
        parent_cache_payload=payload,
        parent_model=parent_model,
        parent_graph=parent_graph,
        tokenizer_fingerprint="c" * 64,
        evidence_builder=lambda _request: None,
        final_evaluator=_Accept(),
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: object(),
        fresh_reload_evaluator=lambda _model: True,
        ledger=MODULE.ExperimentLedger(),
        device="cpu",
        dtype=torch.float32,
    )
    opted_in_dense_config = opted_in.stack_adapter.resources.dense_fold_config
    assert (
        opted_in_dense_config.closed_loop_compiler_config
        .merge_adjacent_layer_captures
        is True
    )
    assert opted_in_dense_config.restart_capsule_directory == (
        (tmp_path / "capsules").absolute()
    )
    assert opted_in_dense_config.restart_capsule_rotation_keep == 3
    assert opted_in_dense_config.resume_from_restart_capsule is None
    # The shard request reaches the dense fold config and, because the adapter
    # is what performs every capture, the adapter config it carries.
    assert opted_in_dense_config.closed_loop_replay_devices == (
        "cuda:0",
        "cuda:1",
    )
    assert (
        opted_in_dense_config.closed_loop_adapter_config.replay_devices
        == ("cuda:0", "cuda:1")
    )
    assert (
        wiring.stack_adapter.resources.dense_fold_config
        .closed_loop_adapter_config.replay_devices
        is None
    )


def test_cached_heterogeneous_builder_receives_dynamic_exact_budget(
    tmp_path,
    monkeypatch,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=4.0,
        dense_heterogeneous_layout=True,
        dense_heterogeneous_factor_writer_rank_discovery=True,
        dense_heterogeneous_hidden_size=96,
        dense_heterogeneous_target_parameters=123_456,
        dense_attention_geometry_risk_enabled=True,
    )
    parent_model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size=64,
            conv_L_cache=7,
            rope_theta=1_000_000.0,
            norm_eps=1e-5,
            max_position_embeddings=8_192,
            use_cache=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    parent_graph = SimpleNamespace(fingerprint="1" * 64)
    parent_cache = SimpleNamespace(cache_key="2" * 64)
    selective_session = object()
    option_bank = SimpleNamespace(
        evidence_sha256="3" * 64,
        layer_options=(object(),),
    )
    observed: dict[str, object] = {}

    def fake_capacity_builder(graph, cache, **kwargs):
        observed["graph"] = graph
        observed["cache"] = cache
        observed["capacity_kwargs"] = dict(kwargs)
        return option_bank

    def fake_pins(**kwargs):
        observed["pins_kwargs"] = dict(kwargs)
        return object()

    def fake_selection_input(**kwargs):
        observed["selection_kwargs"] = dict(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        MODULE,
        "build_lfm2_cached_heterogeneous_capacity_option_bank",
        fake_capacity_builder,
    )
    monkeypatch.setattr(MODULE, "build_lfm2_heterogeneous_target_pins", fake_pins)
    monkeypatch.setattr(
        MODULE,
        "Lfm2HeterogeneousLayoutSelectionInput",
        fake_selection_input,
    )

    selection_input, returned_bank = MODULE._build_cached_heterogeneous_selection_input(
        config,
        parent_model=parent_model,
        parent_graph=parent_graph,
        parent_cache=parent_cache,
        selective_session=selective_session,
        tokenizer_fingerprint="4" * 64,
        tokenizer_bridge=None,
    )

    assert returned_bank is option_bank
    assert observed["graph"] is parent_graph
    assert observed["cache"] is parent_cache
    assert observed["capacity_kwargs"] == {
        "selective_session": selective_session,
        "target_hidden_size": 96,
        "target_vocab_size": 64,
        "exact_target_unique_parameters": 123_456,
        "conv_kernel_size": 7,
        "attention_geometry_risk_enabled": True,
        "discover_exact_short_conv_writer_ranks": True,
    }
    assert observed["pins_kwargs"]["hidden_size"] == 96
    assert observed["pins_kwargs"]["vocab_size"] == 64
    assert observed["pins_kwargs"]["conv_kernel_size"] == 7
    assert selection_input.layer_options == option_bank.layer_options


def test_wiring_plans_h8_at_roughly_250x_without_weakening_capability_gate(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=250.0,
        dense_maximum_candidates=16,
        dense_minimum_hidden_size=8,
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=_Rows(),
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    parent = transformers.Lfm2ForCausalLM(
        transformers.Lfm2Config(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=1024,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=16,
            layer_types=[
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
            ],
            conv_L_cache=3,
            block_auto_adjust_ff_dim=False,
            tie_word_embeddings=True,
        )
    )
    parent_graph = MODULE.inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint="c" * 64,
    )

    wiring = MODULE.build_production_catalog_wiring(
        config,
        parent_cache_payload=payload,
        parent_graph=parent_graph,
        tokenizer_fingerprint="c" * 64,
        evidence_builder=lambda _request: None,
        final_evaluator=_Accept(),
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: object(),
        fresh_reload_evaluator=lambda _model: True,
        ledger=MODULE.ExperimentLedger(),
        device="cpu",
        dtype=torch.float32,
    )

    assert wiring.budget_plan is not None
    selected_rank = wiring.budget_plan.target_certificate.selected_candidate_rank
    selected = wiring.budget_plan.frontier.candidates[selected_rank]
    selected_configuration = dict(selected.configuration)
    topology_catalog = wiring.system.topology_catalogs.select(
        parent_graph,
        target_family="lfm2",
        target_modalities=("text",),
    )

    assert config.identity()["target"]["dense_minimum_hidden_size"] == 8
    assert topology_catalog.minimum_hidden_size == 8
    assert selected_configuration["hidden_size"] == 8
    assert wiring.budget_plan.target_certificate.selection_policy == (
        "supported_pareto_balanced_executable_v1"
    )
    assert dict(wiring.target_request.configuration)["hidden_size"] == 8
    assert wiring.target_request.exact_unique_parameters <= (
        parent_graph.unique_parameter_count // 250
    )
    assert selected.compiler_executable
    assert selected.missing_compiler_capabilities == ()
    assert topology_catalog.compiler_capabilities == (
        MODULE.Lfm2DenseTargetTopologyCatalog().compiler_capabilities
    )
    assert selected.required_compiler_capabilities == (
        topology_catalog.compiler_capabilities
    )
    assert wiring.budget_plan.frontier.diagnostics[
        "minimum_hidden_size"
    ] == 8
    assert wiring.budget_plan.frontier.diagnostics[
        "compiler_capability_selection_fail_closed"
    ]


def test_wiring_emits_explicit_hybrid_method_and_ordered_span_request(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        exact_target_unique_parameters=12345,
        hybrid_model_fold=True,
        target_layer_types=("full_attention", "conv"),
        ordered_source_spans=((0, 1), (2,)),
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=_Rows(),
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    builder = _HybridBuilder()
    wiring = MODULE.build_production_catalog_wiring(
        config,
        parent_cache_payload=payload,
        tokenizer_fingerprint="c" * 64,
        evidence_builder=builder,
        final_evaluator=_Accept(),
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: object(),
        fresh_reload_evaluator=lambda _model: True,
        ledger=MODULE.ExperimentLedger(),
        device="cpu",
        dtype=torch.float32,
    )

    request = wiring.target_request
    assert request.fold_method_id == MODULE.HYBRID_MODEL_METHOD
    assert dict(request.configuration) == {
        "source_parent": "parent",
        "target_depth": 2,
        "target_layer_types": "full_attention,conv",
        "ordered_source_spans": "0,1;2",
    }
    assert wiring.stack_adapter.resources.model_evidence_builder == (
        builder.build_model
    )
    assert wiring.stack_adapter.resources.model_fold_config is not None
    assert MODULE.HYBRID_MODEL_METHOD in wiring.stack_adapter.method_ids


def test_wiring_emits_explicit_producer_first_full_depth_method(tmp_path) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        exact_target_unique_parameters=12345,
        producer_first_model_fold=True,
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=_Rows(),
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    builder = _HybridBuilder()
    wiring = MODULE.build_production_catalog_wiring(
        config,
        parent_cache_payload=payload,
        tokenizer_fingerprint="c" * 64,
        evidence_builder=builder,
        final_evaluator=_Accept(),
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: object(),
        fresh_reload_evaluator=lambda _model: True,
        ledger=MODULE.ExperimentLedger(),
        device="cpu",
        dtype=torch.float32,
    )

    assert wiring.target_request.fold_method_id == (
        MODULE.PRODUCER_FIRST_MODEL_METHOD
    )
    assert dict(wiring.target_request.configuration) == {
        "source_parent": "parent"
    }
    resources = wiring.stack_adapter.resources
    assert resources.producer_first_evidence_builder == (
        builder.build_producer_first
    )
    assert resources.producer_first_fold_config is not None
    assert resources.model_evidence_builder is None
    assert (
        MODULE.PRODUCER_FIRST_MODEL_METHOD
        in wiring.stack_adapter.method_ids
    )


def test_cached_prefix_mismatch_is_exposed_without_weakening_tolerance() -> None:
    diagnostic = MODULE.diagnose_failure(
        RuntimeError(
            "decode.000 cached logits differ from use_cache=False "
            "full-prefix logits (maximum error 1.375)."
        )
    )

    assert diagnostic["category"] == ("cached_vs_full_prefix_logit_mismatch")
    assert diagnostic["phase"] == "decode.000"
    assert diagnostic["maximum_logit_error"] == 1.375
    assert diagnostic["tolerance_weakened"] is False
    assert diagnostic["fail_closed"] is True
    assert diagnostic["next_function_to_fix"].endswith("._capture_batch")
    assert diagnostic["detector"].endswith("._compare_logits")


def test_new_ledger_attempt_evidence_exposes_only_this_invocation() -> None:
    old = SimpleNamespace(
        record_sha256="1" * 64,
        status="accepted",
        reason_class="old",
        reason="old record",
        signature=SimpleNamespace(
            method_id="old-method",
            candidate_sha256="2" * 64,
        ),
        metrics=(("score", 0.1),),
    )
    current = SimpleNamespace(
        record_sha256="3" * 64,
        status="rejected",
        reason_class="behavior_gate_reject",
        reason="candidate margin gate rejected",
        signature=SimpleNamespace(
            method_id=MODULE.DENSE_BUDGET_MODEL_METHOD,
            candidate_sha256="4" * 64,
        ),
        metrics=(
            ("vocabulary_margin_candidate_gate_enabled", True),
            ("vocabulary_margin_candidate_gate_passed", False),
            (
                "vocabulary_margin_candidate_"
                "candidate_exact_parent_top1_agreement_fraction",
                0.42,
            ),
        ),
    )
    ledger = SimpleNamespace(records=(old, current))

    assert MODULE._new_ledger_attempt_evidence(ledger, 1) == [
        {
            "record_sha256": "3" * 64,
            "status": "rejected",
            "reason_class": "behavior_gate_reject",
            "reason": "candidate margin gate rejected",
            "method_id": MODULE.DENSE_BUDGET_MODEL_METHOD,
            "candidate_sha256": "4" * 64,
            "metrics": {
                "vocabulary_margin_candidate_gate_enabled": True,
                "vocabulary_margin_candidate_gate_passed": False,
                (
                    "vocabulary_margin_candidate_"
                    "candidate_exact_parent_top1_agreement_fraction"
                ): 0.42,
            },
        }
    ]


def test_sealed_parent_cache_never_calls_builder_or_parent_forward(
    monkeypatch,
    sealed_parent_cache_fixture,
) -> None:
    fixture = sealed_parent_cache_fixture
    config = _sealed_resolution_config(
        fixture.root,
        fixture.cache.path,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("sealed cache mode must not execute this function")

    monkeypatch.setattr(
        MODULE,
        "build_lfm2_circuit_trace_cache",
        forbidden,
    )
    monkeypatch.setattr(fixture.parent, "forward", forbidden)
    monkeypatch.setattr(MODULE.Lfm2CircuitTraceCache, "iter_batches", forbidden)
    monkeypatch.setattr(MODULE.Lfm2CircuitTraceCache, "load_batch", forbidden)

    resolved = MODULE.resolve_production_parent_cache(
        config,
        fixture.parent,
        fixture.tokenizer,
        fixture.capture_config,
        input_device="cpu",
    )

    assert resolved.cache.cache_key == fixture.cache.cache_key
    assert resolved.source_mode == MODULE.SEALED_PARENT_CACHE_SOURCE_MODE
    assert resolved.evidence_split_source == "sealed_cache_deterministic"
    assert resolved.parent_cache_build_called is False
    assert resolved.parent_cache_opened_with_full_verification is False
    assert resolved.parent_trace_regeneration_forward_count == 0
    assert resolved.supplied_calibration_matches_cache is None
    assert resolved.calibration_source_sha256 is None
    assert len(resolved.evidence_split_sha256) == 64
    assert sorted(resolved.calibration_splits) == [
        "fit",
        "selection",
        "validation",
    ]
    receipt_fields = resolved.runtime_receipt_fields()
    assert receipt_fields["parent_cache_build_called"] is False
    assert receipt_fields["parent_cache_build_forward_count"] == 0
    assert receipt_fields["parent_trace_regeneration_forward_count"] == 0
    assert receipt_fields["parent_trace_regeneration_after_cache"] == 0
    assert receipt_fields["parent_cache_selective_session_reused"] is True
    assert len(receipt_fields["parent_cache_selective_session_sha256"]) == 64
    assert resolved.selective_session is not None
    resolved.selective_session.close()


def test_sealed_parent_cache_uses_one_selective_hash_authentication_pass(
    monkeypatch,
    sealed_parent_cache_fixture,
) -> None:
    fixture = sealed_parent_cache_fixture
    config = _sealed_resolution_config(
        fixture.root,
        fixture.cache.path,
    )
    hash_counts: dict[Path, int] = {}
    real_file_sha256 = trace_cache._file_sha256

    def counting_file_sha256(path: Path) -> str:
        resolved = Path(path).resolve()
        if resolved.name in {
            trace_cache.TRACE_FILENAME,
            trace_cache.BATCH_RECORD_FILENAME,
        }:
            hash_counts[resolved] = hash_counts.get(resolved, 0) + 1
        return real_file_sha256(path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("sealed cache admission must not fully load a batch")

    monkeypatch.setattr(trace_cache, "_file_sha256", counting_file_sha256)
    monkeypatch.setattr(trace_cache, "_load_verified_batch", forbidden)
    monkeypatch.setattr(trace_cache.Lfm2CircuitTraceCache, "verify", forbidden)
    monkeypatch.setattr(fixture.parent, "forward", forbidden)

    resolved = MODULE.resolve_production_parent_cache(
        config,
        fixture.parent,
        fixture.tokenizer,
        fixture.capture_config,
        input_device="cpu",
    )
    try:
        committed_payloads = {
            path.resolve()
            for path in fixture.cache.path.rglob("*")
            if path.name
            in {
                trace_cache.TRACE_FILENAME,
                trace_cache.BATCH_RECORD_FILENAME,
            }
        }
        assert resolved.parent_cache_opened_with_full_verification is False
        assert resolved.parent_trace_regeneration_forward_count == 0
        assert resolved.selective_session is not None
        contract = resolved.selective_session.as_dict()["authentication_contract"]
        assert contract["trace_file_sha256_passes_per_session"] == 1
        assert contract["record_file_sha256_passes_per_session"] == 1
        assert contract["unselected_tensor_payloads_loaded"] is False
        assert contract["parent_forward_calls"] == 0
        assert contract["training_steps"] == 0
    finally:
        assert resolved.selective_session is not None
        resolved.selective_session.close()
    assert set(hash_counts) == committed_payloads
    assert all(count == 1 for count in hash_counts.values())
    assert len(hash_counts) == 2 * len(fixture.cache)


def test_global_frontier_can_resolve_matching_calibration_metadata_only(
    tmp_path,
    monkeypatch,
    sealed_parent_cache_fixture,
) -> None:
    fixture = sealed_parent_cache_fixture
    calibration = tmp_path / "matching-calibration.json"
    calibration.write_text(
        json.dumps(
            [
                {
                    "split": split,
                    "input_ids": [1, 2, 3, 4 + index],
                    "attention_mask": [1, 1, 1, 1],
                    "forced_decode_ids": [8 + index],
                }
                for index, split in enumerate(
                    ("selection", "fit", "validation")
                )
            ]
        ),
        encoding="utf-8",
    )
    config = _sealed_resolution_config(
        tmp_path,
        fixture.cache.path,
        calibration=calibration,
    )

    def forbidden_session(_self):
        raise AssertionError(
            "metadata-only global planning must defer payload authentication"
        )

    monkeypatch.setattr(
        trace_cache.Lfm2CircuitTraceCache,
        "selective_authenticated_session",
        forbidden_session,
    )
    resolved = MODULE.resolve_production_parent_cache(
        config,
        fixture.parent,
        fixture.tokenizer,
        fixture.capture_config,
        input_device="cpu",
        defer_selective_session=True,
    )

    assert resolved.selective_session is None
    assert resolved.supplied_calibration_matches_cache is True
    assert resolved.evidence_split_source == "supplied_matching_calibration"
    assert resolved.calibration_splits == ("selection", "fit", "validation")
    assert len(resolved.calibration_batches) == len(fixture.cache)
    receipt = resolved.runtime_receipt_fields()
    assert receipt["parent_cache_selective_session_reused"] is False
    assert receipt["parent_cache_selective_session_sha256"] is None


def test_sealed_parent_cache_closes_session_once_when_binding_fails_after_calibration(
    monkeypatch,
    sealed_parent_cache_fixture,
) -> None:
    fixture = sealed_parent_cache_fixture
    config = _sealed_resolution_config(
        fixture.root,
        fixture.cache.path,
    )
    captured: dict[str, object] = {}
    close_calls: list[object] = []
    calibration_completed = False
    original_session_factory = (
        trace_cache.Lfm2CircuitTraceCache.selective_authenticated_session
    )
    original_close = (
        trace_cache.Lfm2CircuitTraceSelectiveAuthenticatedSession.close
    )
    original_calibration = MODULE.residual.calibration_batches_from_cache

    def tracked_session_factory(self):
        session = original_session_factory(self)
        captured["session"] = session
        return session

    def tracked_close(self):
        close_calls.append(self)
        return original_close(self)

    def tracked_calibration(*args, **kwargs):
        nonlocal calibration_completed
        result = original_calibration(*args, **kwargs)
        calibration_completed = True
        return result

    def fail_binding(*_args, **_kwargs):
        assert calibration_completed is True
        raise MODULE.ProductionLfm2CatalogError("forced post-calibration failure")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("sealed cache failure path must not call the parent")

    monkeypatch.setattr(
        trace_cache.Lfm2CircuitTraceCache,
        "selective_authenticated_session",
        tracked_session_factory,
    )
    monkeypatch.setattr(
        trace_cache.Lfm2CircuitTraceSelectiveAuthenticatedSession,
        "close",
        tracked_close,
    )
    monkeypatch.setattr(
        MODULE.residual,
        "calibration_batches_from_cache",
        tracked_calibration,
    )
    monkeypatch.setattr(
        MODULE,
        "_validate_sealed_parent_cache_binding",
        fail_binding,
    )
    monkeypatch.setattr(fixture.parent, "forward", forbidden)

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="forced post-calibration failure",
    ):
        MODULE.resolve_production_parent_cache(
            config,
            fixture.parent,
            fixture.tokenizer,
            fixture.capture_config,
            input_device="cpu",
        )

    session = captured["session"]
    assert calibration_completed is True
    assert close_calls == [session]
    assert session.closed is True
    assert len(session._batch_states) == len(fixture.cache)
    assert all(state.trace_guard.closed for state in session._batch_states)


def test_tokenizer_facing_parent_accepts_padded_model_vocabulary() -> None:
    parent = _sealed_test_parent(seed=703)
    tokenizer = _FixedVocabularyTokenizer(29)
    batches = (
        {
            "input_ids": torch.tensor([[1, 2, 28]], dtype=torch.long),
            "forced_decode_ids": torch.tensor([[7]], dtype=torch.long),
        },
    )

    MODULE._validate_tokenizer_facing_parent(parent, tokenizer, batches)


def test_tokenizer_facing_parent_rejects_tokenizer_beyond_model_rows() -> None:
    parent = _sealed_test_parent(seed=704)
    tokenizer = _FixedVocabularyTokenizer(32)
    batches = (
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "forced_decode_ids": torch.tensor([[7]], dtype=torch.long),
        },
    )

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="Tokenizer vocabulary exceeds",
    ):
        MODULE._validate_tokenizer_facing_parent(parent, tokenizer, batches)


def test_tokenizer_facing_parent_rejects_cache_id_beyond_model_rows() -> None:
    parent = _sealed_test_parent(seed=705)
    tokenizer = _FixedVocabularyTokenizer(31)
    batches = (
        {
            "input_ids": torch.tensor([[1, 2, 31]], dtype=torch.long),
            "forced_decode_ids": torch.tensor([[7]], dtype=torch.long),
        },
    )

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="row capacity",
    ):
        MODULE._validate_tokenizer_facing_parent(parent, tokenizer, batches)


def test_sealed_parent_cache_ignores_mismatched_calibration_for_splits(
    tmp_path,
    monkeypatch,
    sealed_parent_cache_fixture,
) -> None:
    fixture = sealed_parent_cache_fixture
    calibration = tmp_path / "different-calibration.json"
    calibration.write_text(
        json.dumps(
            [
                {
                    "split": "selection",
                    "input_ids": [1, 20, 21],
                    "forced_decode_ids": [22],
                },
                {
                    "split": "fit",
                    "input_ids": [1, 23, 24],
                    "forced_decode_ids": [25],
                },
            ]
        ),
        encoding="utf-8",
    )
    config = _sealed_resolution_config(
        tmp_path,
        fixture.cache.path,
        calibration=calibration,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("sealed cache mode must not call the builder")

    monkeypatch.setattr(
        MODULE,
        "build_lfm2_circuit_trace_cache",
        forbidden,
    )

    resolved = MODULE.resolve_production_parent_cache(
        config,
        fixture.parent,
        fixture.tokenizer,
        fixture.capture_config,
        input_device="cpu",
    )

    assert resolved.supplied_calibration_matches_cache is False
    assert resolved.supplied_calibration_sha256 == MODULE._file_sha256(
        calibration
    )
    assert resolved.evidence_split_source == "sealed_cache_deterministic"
    assert resolved.calibration_source_sha256 is None
    assert len(resolved.calibration_splits) == len(fixture.cache)


def test_sealed_parent_cache_rejects_different_live_parent(
    monkeypatch,
    sealed_parent_cache_fixture,
) -> None:
    fixture = sealed_parent_cache_fixture
    different_parent = _sealed_test_parent(seed=702)
    config = _sealed_resolution_config(
        fixture.root,
        fixture.cache.path,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("sealed cache mode must not call the builder")

    monkeypatch.setattr(
        MODULE,
        "build_lfm2_circuit_trace_cache",
        forbidden,
    )

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="fingerprint does not match",
    ):
        MODULE.resolve_production_parent_cache(
            config,
            different_parent,
            fixture.tokenizer,
            fixture.capture_config,
            input_device="cpu",
        )


def test_sealed_parent_cache_rejects_corrupt_trace_file(
    tmp_path,
    monkeypatch,
    sealed_parent_cache_fixture,
) -> None:
    fixture = sealed_parent_cache_fixture
    corrupt_cache = tmp_path / "corrupt-cache"
    shutil.copytree(fixture.cache.path, corrupt_cache)
    trace_path = next(corrupt_cache.rglob("trace.safetensors"))
    contents = bytearray(trace_path.read_bytes())
    contents[-1] ^= 1
    trace_path.write_bytes(contents)
    config = _sealed_resolution_config(tmp_path, corrupt_cache)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("sealed cache mode must not call the builder")

    monkeypatch.setattr(
        MODULE,
        "build_lfm2_circuit_trace_cache",
        forbidden,
    )

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="failed selective authentication",
    ):
        MODULE.resolve_production_parent_cache(
            config,
            fixture.parent,
            fixture.tokenizer,
            fixture.capture_config,
            input_device="cpu",
        )


def test_missing_local_parent_writes_compact_failed_receipt_and_no_artifact(
    tmp_path,
    capsys,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "missing-parent",
        calibration=tmp_path / "missing-calibration.json",
        cached_parent_outputs=tmp_path / "missing-parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        diagnostics_path=tmp_path / "diagnostics.json",
        device="cpu",
        dtype="float32",
    )

    receipt = MODULE.run_production_catalog(config)
    progress = _progress_rows(capsys.readouterr().out)
    stored = json.loads(config.receipt_path.read_text(encoding="ascii"))

    assert receipt == stored
    assert receipt["status"] == "failed_closed"
    assert receipt["publication_present"] is False
    assert receipt["training_steps"] == 0
    assert receipt["tolerance_weakened"] is False
    assert receipt["ledger"]["new_attempts"] == []
    assert len(receipt["receipt_sha256"]) == 64
    assert config.ledger_path.read_bytes() == b""
    assert config.diagnostics_path.is_file()
    assert not config.publish_destination.exists()
    assert [row["stage"] for row in progress] == [
        "run_started",
        "directories_ready",
        "failure",
        "final_receipt",
    ]
    assert progress[-2]["failure_category"] == "fold_or_infrastructure_failure"
    assert progress[-1]["receipt_status"] == "failed_closed"


def test_failed_receipt_explicitly_records_candidate_gate_policy(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "missing-parent",
        calibration=tmp_path / "missing-calibration.json",
        cached_parent_outputs=tmp_path / "missing-parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=10.0,
        dense_vocabulary_margin_observability=True,
        dense_vocabulary_margin_candidate_gate_enabled=True,
        dense_vocabulary_margin_candidate_replay_mode=(
            MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
        ),
        dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction=0.81,
        dense_vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction=0.82,
        dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction=0.83,
        dense_vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction=0.84,
        device="cpu",
        dtype="float32",
    )

    receipt = MODULE.run_production_catalog(config)

    assert receipt["status"] == "failed_closed"
    assert receipt["configuration_sha256"] == MODULE.canonical_sha256(
        config.identity()
    )
    assert receipt["vocabulary_margin_candidate_gate_policy"] == {
        "enabled": True,
        "replay_mode": MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        "minimum_sign_agreement_fraction": pytest.approx(0.81),
        "minimum_top_competitor_nonnegative_fraction": pytest.approx(0.82),
        "minimum_all_top_k_nonnegative_fraction": pytest.approx(0.83),
        "minimum_exact_top1_agreement_fraction": pytest.approx(0.84),
    }
    assert json.loads(
        config.receipt_path.read_text(encoding="ascii")
    ) == receipt


def test_failed_receipt_records_macro_cell_attractor_policy(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "missing-parent",
        calibration=tmp_path / "missing-calibration.json",
        cached_parent_outputs=tmp_path / "missing-parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=10.0,
        dense_run_closed_loop=True,
        dense_macro_cell_attractor_gate_enabled=True,
        dense_macro_cell_attractor_maximum_records=5,
        dense_macro_cell_attractor_maximum_steps=7,
        dense_macro_cell_attractor_maximum_entropy_regression=0.11,
        dense_macro_cell_attractor_maximum_repetition_regression=0.12,
        dense_macro_cell_attractor_maximum_dominance_regression=0.13,
        dense_macro_cell_attractor_maximum_eos_regression=0.14,
        dense_macro_cell_attractor_collapse_entropy_threshold=0.15,
        dense_macro_cell_attractor_collapse_repetition_threshold=0.75,
        dense_macro_cell_attractor_collapse_dominance_threshold=0.76,
        dense_macro_cell_attractor_maximum_new_collapsed_records=2,
        device="cpu",
        dtype="float32",
    )

    receipt = MODULE.run_production_catalog(config)
    policy = receipt["macro_cell_attractor_gate_policy"]

    assert receipt["status"] == "failed_closed"
    assert policy["enabled"]
    assert policy["maximum_records"] == 5
    assert policy["maximum_steps"] == 7
    assert (
        policy["maximum_entropy_parent_distance_regression"]
        == pytest.approx(0.11)
    )
    assert policy["maximum_repetition_excess_regression"] == pytest.approx(
        0.12
    )
    assert (
        policy["maximum_dominant_token_excess_regression"]
        == pytest.approx(0.13)
    )
    assert (
        policy["maximum_eos_position_error_regression"]
        == pytest.approx(0.14)
    )
    assert policy["maximum_new_collapsed_records"] == 2
    assert policy["parent_outputs_cached"]
    assert policy["child_only_free_rollout"]
    assert policy["exact_token_match_is_diagnostic_only"]
    assert policy["training_steps"] == 0


def test_failed_receipt_records_attention_score_policy(tmp_path) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "missing-parent",
        calibration=tmp_path / "missing-calibration.json",
        cached_parent_outputs=tmp_path / "missing-parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=10.0,
        dense_run_closed_loop=True,
        dense_attention_consumer_mode="coupled-value-output",
        dense_attention_score_candidate_gate_enabled=True,
        dense_attention_score_top_k=6,
        dense_attention_score_margin_floor_fraction=0.4,
        dense_attention_score_minimum_top1_key_agreement_fraction=0.81,
        dense_attention_score_minimum_top_k_pair_order_agreement_fraction=0.82,
        dense_attention_score_minimum_winner_margin_sign_agreement_fraction=0.83,
        dense_attention_score_minimum_relative_margin_floor_retention_fraction=0.84,
        dense_attention_score_maximum_softmax_fisher_error=0.12,
        dense_attention_score_maximum_js_divergence=0.13,
        device="cpu",
        dtype="float32",
    )

    receipt = MODULE.run_production_catalog(config)
    policy = receipt["attention_score_candidate_gate_policy"]

    assert receipt["status"] == "failed_closed"
    assert policy["enabled"]
    assert policy["consumer_mode"] == "coupled-value-output"
    assert policy["top_k"] == 6
    assert policy["relative_margin_floor"] == pytest.approx(0.4)
    assert policy["minimum_top1_key_agreement_fraction"] == pytest.approx(
        0.81
    )
    assert policy[
        "minimum_top_k_pair_order_agreement_fraction"
    ] == pytest.approx(0.82)
    assert policy[
        "minimum_winner_margin_sign_agreement_fraction"
    ] == pytest.approx(0.83)
    assert policy[
        "minimum_relative_margin_floor_retention_fraction"
    ] == pytest.approx(0.84)
    assert policy["maximum_softmax_fisher_error"] == pytest.approx(0.12)
    assert policy["maximum_js_divergence"] == pytest.approx(0.13)
    assert policy["masked_keys_excluded"]
    assert policy["numeric_weight_matching_used"] is False
    assert policy["parent_forward_calls"] == 0
    assert policy["training_steps"] == 0


def test_tiny_real_local_catalog_folds_publishes_and_reloads(
    tmp_path,
    capsys,
) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    parent_directory = tmp_path / "parent"
    parent_config = transformers.Lfm2Config(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    torch.manual_seed(123)
    parent = transformers.Lfm2ForCausalLM(parent_config).eval()
    MODULE.residual.atomic_save_local_checkpoint(
        parent,
        parent_directory,
        output_root=tmp_path,
    )
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<unk>": 0,
                "<pad>": 1,
                "<bos>": 2,
                "<eos>": 3,
                    "4": 4,
                "b": 5,
                "c": 6,
                "d": 7,
                "e": 8,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(parent_directory / "tokenizer.json"))
    (parent_directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    target_config = transformers.Lfm2Config(**parent_config.to_dict())
    target_config.intermediate_size = 5
    target_config.block_ff_dim = 5
    target_config.block_auto_adjust_ff_dim = False
    target_count = MODULE.residual.unique_parameter_count(
        transformers.Lfm2ForCausalLM(target_config)
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            [
                {
                    "split": "selection",
                    "input_ids": [2, 4, 5],
                    "forced_decode_ids": [6, 7],
                },
                {
                    "split": "fit",
                    "input_ids": [2, 5, 6],
                    "forced_decode_ids": [7, 8],
                },
                {
                    "split": "validation",
                    "input_ids": [2, 6, 7],
                    "forced_decode_ids": [8, 4],
                },
            ]
        ),
        encoding="utf-8",
    )
    cached_parent_output_rows = tmp_path / "parent-output-rows.json"
    cached_parent_output_rows.write_text(
        json.dumps(
            [
                {
                    "id": "tiny",
                        "prompt": "4",
                        "category": "math",
                        "expected_response": "4",
                        "parent_output_text": "4 4 4 4",
                        "parent_output_ids": [4, 4, 4, 4],
                }
            ]
        ),
        encoding="utf-8",
    )
    loaded_calibration = MODULE.residual.load_calibration_batches(calibration)
    capture_config = MODULE.Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=8,
        max_decode_steps=8,
        max_capture_bytes=128 * 1024 * 1024,
        logit_rtol=0.2,
        logit_atol=1.0,
        require_top1_agreement=False,
    )
    parent_cache = MODULE.build_lfm2_circuit_trace_cache(
        tmp_path / "cache" / "parent",
        parent,
        loaded_calibration.batches,
        config=capture_config,
        input_device="cpu",
    )
    cached_parent_outputs = tmp_path / "parent-output-envelope.json"
    envelope_receipt = seal_lfm2_cached_whole_output_envelope(
        cached_parent_output_rows,
        cached_parent_outputs,
        parent_cache_key=parent_cache.cache_key,
        parent_state_fingerprint=parent_cache.parent_fingerprint,
        tokenizer_fingerprint=MODULE.fingerprint_local_tokenizer(parent_directory),
        generation_limit=4,
    )
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=parent_directory,
        calibration=calibration,
        cached_parent_outputs=cached_parent_outputs,
        cached_parent_outputs_sha256=envelope_receipt["file_sha256"],
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        diagnostics_path=tmp_path / "diagnostics.json",
        device="cpu",
        dtype="float32",
        target_query_heads=4,
        target_key_value_heads=2,
        target_intermediate_size=5,
        exact_target_unique_parameters=target_count,
        max_selection_rows=16,
        max_fit_rows=16,
        max_validation_rows=16,
        endpoint_ridge=0.0,
        max_endpoint_relative_rmse=1e6,
        min_endpoint_cosine_similarity=-1.0,
        min_residual_cosine=-1.0,
        whole_output_max_new_tokens=4,
        snapshot_chunk_bytes=1024 * 1024,
    )

    receipt = MODULE.run_production_catalog(config)
    progress = _progress_rows(capsys.readouterr().out)
    diagnostics = json.loads(config.diagnostics_path.read_text(encoding="ascii"))

    assert receipt["status"] == "complete"
    assert receipt["target"]["exact_unique_parameters"] == target_count
    assert receipt["target"]["dense_minimum_hidden_size"] == 32
    assert receipt["target"]["parameters_removed"] > 0
    assert receipt["target"]["hub_loader_used"] is False
    assert receipt["fold"]["snapshot_backend"] == "disk"
    assert receipt["fold"]["optimizer_steps"] == 0
    assert receipt["fold"]["backward_calls"] == 0
    assert receipt["parent"]["parent_forward_count_after_cache"] == 0
    assert set(receipt["behavior"]) == {
        "accepted_in_memory_child",
        "freshly_reloaded_child",
    }
    assert receipt["artifact"]["fresh_reload_accepted"] is True
    assert (config.publish_destination / "certificate.json").is_file()
    assert (config.publish_destination / "child" / "tokenizer.json").is_file()
    assert diagnostics["parent_traces_regenerated_after_cache"] is False
    assert diagnostics["evidence_builds"]
    assert [row["stage"] for row in progress] == [
        "run_started",
        "directories_ready",
        "local_artifacts_verified",
        "parent_verified",
        "parent_cache_verified",
        "tokenizer_bridge_verified",
        "carrier_verified",
        "parent_rows_materialization_started",
        "parent_rows_materialized",
        "global_frontier_planning_started",
        "wiring_planned",
        "parent_identity_digest_started",
        "system_fold_started",
        "system_fold_completed",
        "success",
        "final_receipt",
    ]
    assert [row["sequence"] for row in progress] == list(range(len(progress)))
    materialized = next(
        row for row in progress if row["stage"] == "parent_rows_materialized"
    )
    assert int(materialized["path_count"]) > 0
    assert int(materialized["retained_bytes"]) > 0
    assert len(str(materialized["trace_rows_sha256"])) == 64
    assert progress[-1]["receipt_status"] == "complete"


def test_production_runner_source_has_no_training_or_hub_loader_path() -> None:
    source = inspect.getsource(MODULE)

    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert ".from_pretrained(" not in source
    assert "hf_hub_download" not in source
    assert "snapshot_download" not in source


def test_cached_global_heterogeneous_builder_selects_dynamic_hidden_frontier(
    tmp_path,
    monkeypatch,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        derived_global_topology_cache_root=(
            tmp_path / "shared-derived-topology-cache"
        ),
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=4.0,
        dense_minimum_hidden_size=32,
        dense_heterogeneous_layout=True,
        dense_heterogeneous_global_hidden_frontier=True,
        dense_heterogeneous_factor_writer_rank_discovery=True,
        dense_heterogeneous_global_functional_rows_per_role=96,
        dense_heterogeneous_target_parameters=123_456,
    )
    parent_model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size=64,
            conv_L_cache=7,
            rope_theta=1_000_000.0,
            norm_eps=1e-5,
            max_position_embeddings=8_192,
            use_cache=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    parent_graph = SimpleNamespace(fingerprint="1" * 64)
    parent_cache = SimpleNamespace(cache_key="2" * 64)
    selective_session = object()
    retained_rows = object()
    phase_context = MODULE.Lfm2GlobalTopologyPhaseCacheContext(
        cache=MODULE.DynamicFoldPhaseCache(tmp_path / "derived-phases"),
        evidence_split_sha256="3" * 64,
    )
    selection_input = object()
    option_bank = object()
    observed: dict[str, object] = {}
    frontier = SimpleNamespace(
        selected_input=selection_input,
        selected_capacity_option_bank=option_bank,
    )

    def fake_global_builder(graph, cache, **kwargs):
        observed["graph"] = graph
        observed["cache"] = cache
        observed["kwargs"] = dict(kwargs)
        return frontier

    monkeypatch.setattr(
        MODULE,
        "build_lfm2_cached_global_topology_frontier",
        fake_global_builder,
    )
    returned_input, returned_bank, returned_frontier = (
        MODULE._build_cached_global_heterogeneous_selection_input(
            config,
            parent_model=parent_model,
            parent_graph=parent_graph,
            parent_cache=parent_cache,
            selective_session=selective_session,
            retained_parent_rows=retained_rows,
            defer_evidence_cache_miss=True,
            tokenizer_fingerprint="4" * 64,
            tokenizer_bridge=None,
            phase_cache_context=phase_context,
        )
    )

    assert returned_input is selection_input
    assert returned_bank is option_bank
    assert returned_frontier is frontier
    assert observed["graph"] is parent_graph
    assert observed["cache"] is parent_cache
    kwargs = observed["kwargs"]
    assert kwargs["exact_target_unique_parameters"] == 123_456
    assert kwargs["required_target_hidden_size"] is None
    assert kwargs["minimum_hidden_size"] == 32
    assert kwargs["maximum_functional_rows_per_role"] == 96
    assert kwargs["discover_exact_short_conv_writer_ranks"] is True
    assert kwargs["selective_session"] is selective_session
    assert kwargs["retained_parent_rows"] is retained_rows
    assert kwargs["defer_evidence_cache_miss"] is True
    assert kwargs["phase_cache_context"] is phase_context
    runtime = kwargs["runtime"]
    assert runtime.vocab_size == 64
    assert runtime.conv_kernel_size == 7
    assert runtime.target_vocab_evidence_sha256 == "4" * 64


def test_global_wiring_forwards_materialized_rows_to_exact_cold_path(
    tmp_path,
    monkeypatch,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        derived_global_topology_cache_root=tmp_path / "derived-phases",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=4.0,
        dense_heterogeneous_layout=True,
        dense_heterogeneous_global_hidden_frontier=True,
        dense_heterogeneous_target_parameters=123_456,
    )
    retained_rows = _Rows()
    selective_session = object()
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=retained_rows,
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
        selective_session=selective_session,
    )
    phase_context = MODULE.Lfm2GlobalTopologyPhaseCacheContext(
        cache=MODULE.DynamicFoldPhaseCache(tmp_path / "derived-phases"),
        evidence_split_sha256="3" * 64,
    )
    observed: dict[str, object] = {}

    class _StopAfterGlobalSelection(RuntimeError):
        pass

    def capture_global_selection(received_config, **kwargs):
        observed["config"] = received_config
        observed.update(kwargs)
        raise _StopAfterGlobalSelection

    monkeypatch.setattr(
        MODULE,
        "_build_cached_global_heterogeneous_selection_input",
        capture_global_selection,
    )
    with pytest.raises(_StopAfterGlobalSelection):
        MODULE.build_production_catalog_wiring(
            config,
            parent_cache_payload=payload,
            parent_model=object(),
            parent_graph=object(),
            tokenizer_fingerprint="4" * 64,
            evidence_builder=lambda _request: None,
            final_evaluator=_Accept(),
            model_saver=lambda _model, _path: None,
            model_loader=lambda _path: object(),
            fresh_reload_evaluator=lambda _model: True,
            ledger=MODULE.ExperimentLedger(),
            device="cpu",
            dtype=torch.float32,
            global_topology_phase_cache_context=phase_context,
        )
    assert observed["config"] is config
    assert observed["retained_parent_rows"] is retained_rows
    assert observed["selective_session"] is selective_session
    assert observed["phase_cache_context"] is phase_context

    runner_source = inspect.getsource(MODULE.run_production_catalog)
    assert runner_source.index("defer_evidence_cache_miss=True") < (
        runner_source.index("residual.extract_trace_rows(")
    )
    assert "except Lfm2CachedGlobalTopologyEvidenceCacheMiss" in runner_source


def test_carrier_h840_frontier_is_rejected_before_target_synthesis(
    tmp_path,
    monkeypatch,
) -> None:
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(_carrier_arguments(tmp_path))
    )
    parent_model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size=4_096,
            conv_L_cache=3,
            rope_theta=1_000_000.0,
            norm_eps=1e-5,
            max_position_embeddings=8_192,
            use_cache=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    observed: dict[str, object] = {}
    h840 = SimpleNamespace(
        required_target_hidden_size=816,
        candidate_hidden_sizes=(840,),
        selected_input=SimpleNamespace(
            target_pins=SimpleNamespace(hidden_size=840, vocab_size=4_096)
        ),
        selected_capacity_option_bank=object(),
    )
    target_tokenizer = SimpleNamespace(
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    tokenizer_bridge = SimpleNamespace(
        artifact=SimpleNamespace(
            artifact_sha256="5" * 64,
            transport_profile=SimpleNamespace(target_row_capacity=4_096),
        ),
        target_tokenizer_artifact_fingerprint="6" * 64,
        target_tokenizer=target_tokenizer,
    )

    def fake_global_builder(_graph, _cache, **kwargs):
        observed.update(kwargs)
        return h840

    monkeypatch.setattr(
        MODULE,
        "build_lfm2_cached_global_topology_frontier",
        fake_global_builder,
    )

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match=(
            "required H816, selected H840, exact body budget 53552368"
        ),
    ):
        MODULE._build_cached_global_heterogeneous_selection_input(
            config,
            parent_model=parent_model,
            parent_graph=object(),
            parent_cache=object(),
            tokenizer_fingerprint="4" * 64,
            tokenizer_bridge=tokenizer_bridge,
        )

    assert observed["required_target_hidden_size"] == 816
    runner_source = inspect.getsource(MODULE.run_production_catalog)
    assert runner_source.index(
        "_build_cached_global_heterogeneous_selection_input("
    ) < runner_source.index('"system_fold_started"')


def test_carrier_rejects_token_bridge_row_capacity_before_frontier(
    tmp_path,
    monkeypatch,
) -> None:
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(_carrier_arguments(tmp_path))
    )
    parent_model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size=65_536,
            conv_L_cache=3,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    tokenizer_bridge = SimpleNamespace(
        artifact=SimpleNamespace(
            artifact_sha256="5" * 64,
            transport_profile=SimpleNamespace(target_row_capacity=4_095),
        ),
        target_tokenizer_artifact_fingerprint="6" * 64,
        target_tokenizer=SimpleNamespace(
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        ),
    )

    def forbidden_builder(*_args, **_kwargs):
        raise AssertionError("frontier must not run after a vocabulary ABI mismatch")

    monkeypatch.setattr(
        MODULE,
        "build_lfm2_cached_global_topology_frontier",
        forbidden_builder,
    )

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="target-row capacity to equal 4096",
    ):
        MODULE._build_cached_global_heterogeneous_selection_input(
            config,
            parent_model=parent_model,
            parent_graph=object(),
            parent_cache=object(),
            tokenizer_fingerprint="4" * 64,
            tokenizer_bridge=tokenizer_bridge,
        )


def test_planned_and_materialized_required_target_abi_are_authenticated(
    tmp_path,
) -> None:
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "4",
                "--dense-heterogeneous-layout",
                "--dense-heterogeneous-global-hidden-frontier",
                "--derived-global-topology-cache-root",
                str(tmp_path / "derived-phases"),
                "--required-target-hidden-size",
                "8",
            ]
        )
    )
    pins = SimpleNamespace(
        hidden_size=8,
        vocab_size=16,
        pins_sha256="1" * 64,
    )
    frontier = SimpleNamespace(
        required_target_hidden_size=8,
        candidate_hidden_sizes=(8,),
        selected_input=SimpleNamespace(target_pins=pins),
        frontier_sha256="2" * 64,
    )
    request = SimpleNamespace(
        configuration=(
            ("hidden_size", 8),
            ("vocab_size", 16),
            ("tie_word_embeddings", False),
        )
    )
    planned = MODULE._verify_planned_target_abi(config, request, frontier)
    child = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=8,
            vocab_size=16,
            tie_word_embeddings=False,
        )
    )
    input_table = torch.nn.Embedding(16, 8)
    output_table = torch.nn.Linear(8, 16, bias=False)
    child.get_input_embeddings = lambda: input_table
    child.get_output_embeddings = lambda: output_table

    materialized = MODULE._verify_materialized_target_abi(
        config,
        child,
        planned,
    )

    assert planned["checked_before_system_fold_started"] is True
    assert planned["planned_hidden_size"] == 8
    assert planned["planned_vocab_size"] == 16
    assert planned["tie_word_embeddings"] is False
    assert materialized["input_table_shape"] == [16, 8]
    assert materialized["output_table_shape"] == [16, 8]
    assert materialized["input_output_tables_untied"] is True

    tied_request = SimpleNamespace(
        configuration=(
            ("hidden_size", 8),
            ("vocab_size", 16),
            ("tie_word_embeddings", True),
        )
    )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="required H/V/untied ABI",
    ):
        MODULE._verify_planned_target_abi(config, tied_request, frontier)


def test_global_topology_phase_context_is_durable_and_relays_progress(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        derived_global_topology_cache_root=(
            tmp_path / "shared-derived-topology-cache"
        ),
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
    )
    journal_stream = _FlushRecordingStream()
    journal = MODULE._ProgressJournal("a" * 64, stream=journal_stream)

    context = MODULE._global_topology_phase_cache_context(
        config,
        evidence_split_sha256="b" * 64,
        progress=journal,
    )
    assert context.cache.root == config.global_topology_phase_cache_root.absolute()
    assert context.evidence_split_sha256 == "b" * 64
    assert context.selective_session_factory is None

    assert context.progress is not None
    context.progress("global_topology_phase_cache_hit", {"resumed": True})
    rows = _progress_rows(journal_stream.getvalue())
    assert rows[-1]["stage"] == "global_topology_phase_cache_hit"
    assert rows[-1]["resumed"] is True


def test_distinct_run_caches_share_explicit_global_topology_phase_root(
    tmp_path,
) -> None:
    stable_root = tmp_path / "stable" / "derived-global-topology-phases"

    def build_config(run_name: str):
        run_root = tmp_path / run_name
        return MODULE.ProductionLfm2CatalogConfig(
            parent_model=tmp_path / "parent",
            calibration=tmp_path / "calibration.json",
            cached_parent_outputs=tmp_path / "parent-outputs.json",
            cached_parent_outputs_sha256="d" * 64,
            cache_root=run_root / "cache",
            derived_global_topology_cache_root=stable_root,
            snapshot_root=run_root / "snapshots",
            publish_destination=run_root / "published",
            ledger_path=run_root / "ledger.jsonl",
            receipt_path=run_root / "receipt.json",
            compression_ratio=4.0,
            dense_heterogeneous_layout=True,
            dense_heterogeneous_global_hidden_frontier=True,
        )

    first_config = build_config("run-a")
    second_config = build_config("run-b")
    assert first_config.cache_root != second_config.cache_root
    assert first_config.evidence_cache_root != second_config.evidence_cache_root
    assert first_config.final_cache_root != second_config.final_cache_root
    assert first_config.reload_cache_root != second_config.reload_cache_root
    assert first_config.global_topology_phase_cache_root == stable_root.absolute()
    assert second_config.global_topology_phase_cache_root == stable_root.absolute()
    assert first_config.identity()["derived_global_topology_cache_root"] == str(
        stable_root.absolute()
    )

    parent, graph, parent_cache, frontier_kwargs = _phase_cache_case(
        tmp_path / "fold-fixture"
    )
    del parent
    cold_stream = _FlushRecordingStream()
    cold_context = MODULE._global_topology_phase_cache_context(
        first_config,
        evidence_split_sha256="b" * 64,
        progress=MODULE._ProgressJournal("d" * 64, stream=cold_stream),
    )
    cold = MODULE.build_lfm2_cached_global_topology_frontier(
        graph,
        parent_cache,
        phase_cache_context=cold_context,
        **frontier_kwargs,
    )

    session_factory_calls = []

    def forbidden_session_factory():
        session_factory_calls.append(True)
        raise AssertionError("phase-cache hit must not open a parent session")

    hot_stream = _FlushRecordingStream()
    hot_context = MODULE._global_topology_phase_cache_context(
        second_config,
        evidence_split_sha256="b" * 64,
        progress=MODULE._ProgressJournal("e" * 64, stream=hot_stream),
        selective_session_factory=forbidden_session_factory,
    )
    hot = MODULE.build_lfm2_cached_global_topology_frontier(
        graph,
        parent_cache,
        phase_cache_context=hot_context,
        **frontier_kwargs,
    )
    assert dict(hot.as_dict()) == dict(cold.as_dict())
    hot_rows = _progress_rows(hot_stream.getvalue())
    hit_rows = [
        row
        for row in hot_rows
        if row["stage"] == "global_topology_phase_cache_hit"
    ]
    assert [row["phase_name"] for row in hit_rows] == [
        GLOBAL_TOPOLOGY_CAPACITY_PHASE,
        GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
        GLOBAL_TOPOLOGY_SELECTION_PHASE,
    ]
    assert session_factory_calls == []
    for phase_name in (
        GLOBAL_TOPOLOGY_CAPACITY_PHASE,
        GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
        GLOBAL_TOPOLOGY_SELECTION_PHASE,
    ):
        entry = next((stable_root / phase_name).iterdir())
        assert {
            path.relative_to(entry).as_posix()
            for path in entry.rglob("*")
            if path.is_file()
        } == {"complete.json", "payload/evidence.json"}


def test_global_hidden_frontier_requires_explicit_stable_phase_cache_root(
    tmp_path,
) -> None:
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="explicit stable derived_global_topology_cache_root",
    ):
        MODULE.ProductionLfm2CatalogConfig(
            parent_model=tmp_path / "parent",
            calibration=tmp_path / "calibration.json",
            cached_parent_outputs=tmp_path / "parent-outputs.json",
            cached_parent_outputs_sha256="d" * 64,
            cache_root=tmp_path / "cache",
            snapshot_root=tmp_path / "snapshots",
            publish_destination=tmp_path / "published",
            ledger_path=tmp_path / "ledger.jsonl",
            receipt_path=tmp_path / "receipt.json",
            compression_ratio=4.0,
            dense_heterogeneous_layout=True,
            dense_heterogeneous_global_hidden_frontier=True,
        )


def test_global_hidden_frontier_requires_cached_heterogeneous_layout(tmp_path) -> None:
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="global_hidden_frontier requires",
    ):
        MODULE.ProductionLfm2CatalogConfig(
            parent_model=tmp_path / "parent",
            calibration=tmp_path / "calibration.json",
            cached_parent_outputs=tmp_path / "parent-outputs.json",
            cached_parent_outputs_sha256="d" * 64,
            cache_root=tmp_path / "cache",
            snapshot_root=tmp_path / "snapshots",
            publish_destination=tmp_path / "published",
            ledger_path=tmp_path / "ledger.jsonl",
            receipt_path=tmp_path / "receipt.json",
            compression_ratio=4.0,
            dense_heterogeneous_global_hidden_frontier=True,
        )


def test_factor_writer_rank_discovery_requires_cached_heterogeneous_layout(
    tmp_path,
) -> None:
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="factor_writer_rank_discovery requires",
    ):
        MODULE.ProductionLfm2CatalogConfig(
            parent_model=tmp_path / "parent",
            calibration=tmp_path / "calibration.json",
            cached_parent_outputs=tmp_path / "parent-outputs.json",
            cached_parent_outputs_sha256="d" * 64,
            cache_root=tmp_path / "cache",
            snapshot_root=tmp_path / "snapshots",
            publish_destination=tmp_path / "published",
            ledger_path=tmp_path / "ledger.jsonl",
            receipt_path=tmp_path / "receipt.json",
            compression_ratio=4.0,
            dense_heterogeneous_factor_writer_rank_discovery=True,
        )


# ---------------------------------------------------------------------------
# Accepted-layer restart capsules, replay-mode default, and on-policy rounds.
#
# The dense closed-loop body fold is the only place capsules exist, so these
# tests drive the real dense budget lane at toy scale through the same
# run_production_catalog entrypoint main() uses.
# ---------------------------------------------------------------------------


_DENSE_LANE_VOCABULARY = {
    "<unk>": 0,
    "<pad>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "4": 4,
    "b": 5,
    "c": 6,
    "d": 7,
    "e": 8,
}


def _build_tiny_dense_lane_pack(root: Path) -> SimpleNamespace:
    """Build the smallest real parent the dense budget catalog can fold.

    ``Lfm2DenseTargetTopologyCatalog`` refuses head dimensions below 8, so the
    toy parent keeps a production-legal head geometry and shrinks only depth,
    vocabulary, residual width, and MLP width.
    """

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    parent_directory = root / "parent"
    parent_config = transformers.Lfm2Config(
        vocab_size=17,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["conv", "conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4242)
        parent = transformers.Lfm2ForCausalLM(parent_config).eval()
    MODULE.residual.atomic_save_local_checkpoint(
        parent,
        parent_directory,
        output_root=root,
    )
    tokenizer = Tokenizer(
        WordLevel(dict(_DENSE_LANE_VOCABULARY), unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(parent_directory / "tokenizer.json"))
    (parent_directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    calibration = root / "calibration.json"
    calibration.write_text(
        json.dumps(
            [
                {
                    "split": "selection",
                    "input_ids": [2, 4, 5],
                    "forced_decode_ids": [6, 7],
                },
                {
                    "split": "fit",
                    "input_ids": [2, 5, 6],
                    "forced_decode_ids": [7, 8],
                },
                {
                    "split": "validation",
                    "input_ids": [2, 6, 7],
                    "forced_decode_ids": [8, 4],
                },
                {
                    "split": "fit",
                    "input_ids": [2, 7, 8],
                    "forced_decode_ids": [4, 5],
                },
            ]
        ),
        encoding="utf-8",
    )
    behavior_rows = root / "parent-output-rows.json"
    behavior_rows.write_text(
        json.dumps(
            [
                {
                    "id": "tiny",
                    "prompt": "4",
                    "category": "math",
                    "expected_response": "4",
                    "parent_output_text": "4 4 4 4",
                    "parent_output_ids": [4, 4, 4, 4],
                }
            ]
        ),
        encoding="utf-8",
    )
    loaded_calibration = MODULE.residual.load_calibration_batches(calibration)
    capture_config = MODULE.Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=8,
        max_decode_steps=8,
        max_capture_bytes=128 * 1024 * 1024,
        logit_rtol=0.2,
        logit_atol=1.0,
        require_top1_agreement=False,
        decode_trace_mode=MODULE.CACHED_VERIFIED_TRACE_MODE,
    )
    parent_cache = MODULE.build_lfm2_circuit_trace_cache(
        root / "sealed-cache",
        parent,
        loaded_calibration.batches,
        config=capture_config,
        input_device="cpu",
    )
    envelope = root / "parent-output-envelope.json"
    envelope_receipt = seal_lfm2_cached_whole_output_envelope(
        behavior_rows,
        envelope,
        parent_cache_key=parent_cache.cache_key,
        parent_state_fingerprint=parent_cache.parent_fingerprint,
        tokenizer_fingerprint=MODULE.fingerprint_local_tokenizer(
            parent_directory
        ),
        generation_limit=4,
    )
    return SimpleNamespace(
        root=root,
        parent_directory=parent_directory,
        calibration=calibration,
        sealed_cache=parent_cache.path,
        cached_parent_outputs=envelope,
        cached_parent_outputs_sha256=envelope_receipt["file_sha256"],
        layer_count=parent_config.num_hidden_layers,
    )


@pytest.fixture(scope="module")
def tiny_dense_lane_pack(tmp_path_factory):
    return _build_tiny_dense_lane_pack(tmp_path_factory.mktemp("tiny-dense-lane"))


def _tiny_dense_lane_arguments(
    pack: SimpleNamespace,
    run: Path,
    extra: list[str],
) -> list[str]:
    return [
        "--parent-model",
        str(pack.parent_directory),
        "--calibration",
        str(pack.calibration),
        "--sealed-parent-cache-directory",
        str(pack.sealed_cache),
        "--cached-parent-outputs",
        str(pack.cached_parent_outputs),
        "--cached-parent-outputs-sha256",
        pack.cached_parent_outputs_sha256,
        "--cache-root",
        str(run / "cache"),
        "--snapshot-root",
        str(run / "snapshots"),
        "--publish-destination",
        str(run / "published"),
        "--ledger",
        str(run / "ledger.jsonl"),
        "--receipt",
        str(run / "receipt.json"),
        "--diagnostics",
        str(run / "diagnostics.json"),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--child-name",
        "tiny-dense-restartable",
        "--compression-ratio",
        "2",
        "--dense-minimum-hidden-size",
        "8",
        "--dense-residual-basis-mode",
        "weight_complete",
        "--dense-run-closed-loop",
        "--dense-closed-loop-performance-metrics-diagnostic-only",
        "--sample-last-tokens",
        "8",
        "--max-decode-steps",
        "8",
        "--max-capture-bytes",
        str(128 * 1024 * 1024),
        "--capture-logit-rtol",
        "0.2",
        "--capture-logit-atol",
        "1",
        "--max-selection-rows",
        "16",
        "--max-fit-rows",
        "16",
        "--max-validation-rows",
        "16",
        "--endpoint-ridge",
        "0",
        "--max-endpoint-relative-rmse",
        "1000000",
        "--min-endpoint-cosine-similarity",
        "-1",
        "--min-residual-cosine",
        "-1",
        "--min-teacher-top1",
        "0",
        "--min-free-run-match",
        "0",
        "--whole-output-max-new-tokens",
        "4",
        "--whole-output-fragment-minimum-characters",
        "1",
        "--whole-output-maximum-new-empty-responses",
        "8",
        "--whole-output-maximum-new-fragment-responses",
        "8",
        "--whole-output-minimum-parent-nonempty-retention",
        "0",
        "--whole-output-minimum-task-score-retention",
        "0",
        "--whole-output-maximum-parent-task-score-regressions",
        "8",
        "--whole-output-maximum-new-generation-limit-saturations",
        "8",
        "--whole-output-maximum-repeated-ngram-share-regressions",
        "8",
        "--whole-output-maximum-repeated-bigram-share-delta",
        "1",
        "--snapshot-chunk-bytes",
        str(1024 * 1024),
        *extra,
    ]


def test_dense_closed_loop_replay_and_capsule_cli_defaults(tmp_path) -> None:
    parser = MODULE.build_parser()
    default_config = MODULE.config_from_args(
        parser.parse_args(_required_args(tmp_path))
    )

    assert default_config.dense_closed_loop_merge_adjacent_layer_captures is False
    assert default_config.dense_restart_capsule_root is None
    assert default_config.dense_restart_capsule_keep == 2
    assert default_config.dense_resume_from_capsule is None
    # Capsules only exist inside the dense closed-loop body fold.
    assert default_config.dense_restart_capsule_directory is None
    assert default_config.on_policy_evidence_rounds == 0
    assert default_config.uses_on_policy_evidence is False

    closed_loop = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "4",
                "--dense-run-closed-loop",
                "--dense-closed-loop-merge-adjacent-layer-captures",
            ]
        )
    )
    derived = closed_loop.dense_restart_capsule_directory
    assert derived is not None
    assert derived.parent == (
        (tmp_path / "cache").absolute() / "closed-loop-restart-capsules"
    )
    assert derived.name.startswith(f"{closed_loop.child_name}-")
    assert closed_loop.dense_closed_loop_merge_adjacent_layer_captures is True
    # The derived root must not move when the same run is resumed, otherwise a
    # killed run could never find its own capsules.
    resumed = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "4",
                "--dense-run-closed-loop",
                "--dense-closed-loop-merge-adjacent-layer-captures",
                "--dense-resume-from-capsule",
                str(tmp_path / "capsules"),
            ]
        )
    )
    assert resumed.dense_restart_capsule_directory == derived
    assert resumed.identity() != closed_loop.identity()

    explicit = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "4",
                "--dense-run-closed-loop",
                "--dense-restart-capsule-root",
                str(tmp_path / "capsules"),
                "--dense-restart-capsule-keep",
                "3",
            ]
        )
    )
    assert explicit.dense_restart_capsule_directory == (
        (tmp_path / "capsules").absolute()
    )
    assert explicit.dense_restart_capsule_keep == 3


def test_restart_capsule_options_require_the_dense_closed_loop(tmp_path) -> None:
    parser = MODULE.build_parser()
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="only exist inside the dense",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--dense-restart-capsule-root",
                    str(tmp_path / "capsules"),
                ]
            )
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="Merged adjacent layer captures require",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--dense-closed-loop-merge-adjacent-layer-captures",
                ]
            )
        )


def _closed_loop_args(tmp_path, *extra: str) -> list[str]:
    return [
        *_required_args(tmp_path),
        "--compression-ratio",
        "4",
        "--dense-run-closed-loop",
        *extra,
    ]


def test_dense_replay_device_cli_wins_over_the_environment(
    tmp_path,
    monkeypatch,
) -> None:
    parser = MODULE.build_parser()
    monkeypatch.delenv(MODULE.REPLAY_DEVICES_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv(
        MODULE.LEGACY_REPLAY_DEVICES_ENVIRONMENT_VARIABLE,
        raising=False,
    )

    default_config = MODULE.config_from_args(
        parser.parse_args(_required_args(tmp_path))
    )
    assert default_config.dense_replay_devices == ()
    assert default_config.dense_replay_devices_source == (
        MODULE.REPLAY_DEVICES_SOURCE_UNSET
    )

    monkeypatch.setenv(
        MODULE.REPLAY_DEVICES_ENVIRONMENT_VARIABLE,
        "cuda:4,cuda:5",
    )
    environment_only = MODULE.config_from_args(
        parser.parse_args(_closed_loop_args(tmp_path))
    )
    assert environment_only.dense_replay_devices == ("cuda:4", "cuda:5")
    assert environment_only.dense_replay_devices_source == (
        MODULE.REPLAY_DEVICES_SOURCE_ENVIRONMENT
    )

    overridden = MODULE.config_from_args(
        parser.parse_args(
            _closed_loop_args(
                tmp_path,
                "--dense-replay-devices",
                " cuda:0 , cuda:1 ,cuda:2 ",
            )
        )
    )
    assert overridden.dense_replay_devices == ("cuda:0", "cuda:1", "cuda:2")
    assert overridden.dense_replay_devices_source == (
        MODULE.REPLAY_DEVICES_SOURCE_CLI
    )

    # An exported device list must never change, or fail, a run that has no
    # closed-loop captures to shard.
    ignored = MODULE.config_from_args(
        parser.parse_args(_required_args(tmp_path))
    )
    assert ignored.dense_replay_devices == ()
    assert ignored.dense_replay_devices_source == (
        MODULE.REPLAY_DEVICES_SOURCE_ENVIRONMENT_IGNORED
    )

    # The device list is execution-only, exactly like portable_runtime_device:
    # a sharded capture seals the same bytes as the serial one, so the fold
    # identity, and therefore every derived cache key, must not move.
    serial = MODULE.config_from_args(
        parser.parse_args(_closed_loop_args(tmp_path))
    )
    monkeypatch.delenv(MODULE.REPLAY_DEVICES_ENVIRONMENT_VARIABLE)
    assert overridden.identity() == replace(
        serial,
        dense_replay_devices=(),
        dense_replay_devices_source=MODULE.REPLAY_DEVICES_SOURCE_UNSET,
    ).identity()
    assert "cuda:1" not in json.dumps(overridden.identity(), sort_keys=True)


def test_dense_replay_devices_fall_back_to_the_legacy_environment_name(
    tmp_path,
    monkeypatch,
) -> None:
    parser = MODULE.build_parser()
    monkeypatch.delenv(MODULE.REPLAY_DEVICES_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setenv(
        MODULE.LEGACY_REPLAY_DEVICES_ENVIRONMENT_VARIABLE,
        "cuda:0,cuda:1",
    )
    config = MODULE.config_from_args(
        parser.parse_args(_closed_loop_args(tmp_path))
    )
    assert config.dense_replay_devices == ("cuda:0", "cuda:1")
    assert config.dense_replay_devices_source == (
        MODULE.REPLAY_DEVICES_SOURCE_ENVIRONMENT
    )

    monkeypatch.setenv(MODULE.REPLAY_DEVICES_ENVIRONMENT_VARIABLE, "cpu,cpu")
    preferred = MODULE.config_from_args(
        parser.parse_args(_closed_loop_args(tmp_path))
    )
    assert preferred.dense_replay_devices == ("cpu", "cpu")


def test_dense_replay_device_requests_fail_closed_on_unusable_shapes(
    tmp_path,
    monkeypatch,
) -> None:
    parser = MODULE.build_parser()
    monkeypatch.delenv(MODULE.REPLAY_DEVICES_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv(
        MODULE.LEGACY_REPLAY_DEVICES_ENVIRONMENT_VARIABLE,
        raising=False,
    )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="only applies to dense closed-loop captures",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--dense-replay-devices",
                    "cuda:0,cuda:1",
                ]
            )
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="must be 'cpu', 'cuda'",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                _closed_loop_args(tmp_path, "--dense-replay-devices", "gpu:0")
            )
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="cannot be combined with explicit devices",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                _closed_loop_args(
                    tmp_path,
                    "--dense-replay-devices",
                    "all,cuda:1",
                )
            )
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="named no replay devices",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                _closed_loop_args(tmp_path, "--dense-replay-devices", " , ")
            )
        )
    every_device = MODULE.config_from_args(
        parser.parse_args(
            _closed_loop_args(tmp_path, "--dense-replay-devices", "all")
        )
    )
    assert every_device.dense_replay_devices == (
        MODULE.REPLAY_DEVICES_ALL_TOKEN,
    )


def test_progress_journal_summarizes_observed_replay_sharding(tmp_path) -> None:
    stream = io.StringIO()
    journal = MODULE._ProgressJournal("a" * 64, stream=stream)
    assert journal.replay_sharding_summary() is None

    journal.emit(
        MODULE.LFM2_REPLAY_SHARDING_RESOLVED_STAGE,
        replay_sharding_enabled=True,
        replay_sharding_resolved_devices="cuda:0,cuda:1",
        replay_sharding_worker_count=2,
        replay_sharding_fallback_reason=None,
    )
    journal.emit(
        MODULE.LFM2_REPLAY_SHARDING_COMPLETED_STAGE,
        replay_sharding_replay_wall_seconds=2.0,
        replay_sharding_worker_replay_seconds_total=3.5,
        replay_sharding_worker_peak_capture_bytes=4096,
    )
    journal.emit(
        MODULE.LFM2_REPLAY_SHARDING_RESOLVED_STAGE,
        replay_sharding_enabled=False,
        replay_sharding_resolved_devices=None,
        replay_sharding_worker_count=1,
        replay_sharding_fallback_reason="child_carries_hooks",
    )

    assert journal.merged_capture_policy() is None
    journal.emit(
        MODULE.DENSE_CLOSED_LOOP_MERGE_CAPTURE_POLICY_STAGE,
        merge_adjacent_layer_captures_requested=True,
        merge_adjacent_layer_captures_admitted=False,
        replay_capture_mode=MODULE.CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        selective_session_bound=True,
        heterogeneous_child=True,
    )
    # A requested-but-inapplicable merge is reported as such, never as
    # silently effective.
    assert journal.merged_capture_policy() == {
        "requested": True,
        "admitted_by_adapter": False,
        "replay_capture_mode": MODULE.CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        "selective_session_bound": True,
        "heterogeneous_child": True,
    }

    summary = journal.replay_sharding_summary()
    assert summary["resolved_capture_count"] == 2
    assert summary["sharded_capture_count"] == 1
    assert summary["fallback_capture_count"] == 1
    assert summary["first_fallback_reason"] == "child_carries_hooks"
    assert summary["resolved_devices"] == "cuda:0,cuda:1"
    assert summary["worker_count"] == 2
    assert summary["completed_capture_count"] == 1
    assert summary["maximum_worker_peak_capture_bytes"] == 4096
    # Speedup is measured, never assumed.
    assert summary["observed_speedup"] == pytest.approx(1.75)


def test_failed_receipt_records_replay_capsule_and_on_policy_policies(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "missing-parent",
        calibration=tmp_path / "missing-calibration.json",
        cached_parent_outputs=tmp_path / "missing-parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=4.0,
        dense_run_closed_loop=True,
        device="cpu",
        dtype="float32",
    )

    receipt = MODULE.run_production_catalog(config)
    replay = receipt["closed_loop_replay_policy"]
    capsule = receipt["restart_capsule_policy"]
    on_policy = receipt["on_policy_evidence_policy"]

    assert receipt["status"] == "failed_closed"
    assert replay["policy"] == MODULE.DENSE_CLOSED_LOOP_REPLAY_AUTO_MODE
    assert replay["resolved_mode"] == MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    assert replay["in_loop_single_pass_certification_replay"] is False
    assert replay["single_pass_certification_is_explicit_opt_in"] is True
    assert replay["merge_adjacent_layer_captures"] is False
    # No fold ran, so there is no adapter decision to report.
    assert replay["merge_adjacent_layer_captures_observed"] is None
    sharding = replay["replay_sharding"]
    assert sharding["requested_devices"] is None
    assert sharding["requested_device_count"] == 0
    assert sharding["request_source"] == MODULE.REPLAY_DEVICES_SOURCE_UNSET
    assert sharding["serial_default_seals_identical_bytes"] is True
    # No capture ran, so there is no observed runtime to claim.
    assert sharding["runtime"] is None
    assert capsule["enabled"] is True
    assert capsule["rotation_keep"] == 2
    assert capsule["resume_requested"] is False
    assert on_policy["requested_rounds"] == 0
    assert on_policy["max_rounds"] == MODULE.DEFAULT_ON_POLICY_MAX_ROUNDS
    assert on_policy["all_rows_are_fit_rows"] is True


def test_on_policy_rounds_fail_closed_without_their_sealed_inputs(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="require an already sealed parent cache",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--on-policy-evidence-rounds",
                    "1",
                ]
            )
        )
    sealed = [
        "--sealed-parent-cache-directory",
        str(tmp_path / "sealed-cache"),
    ]
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="require --on-policy-fit-prompts",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    *sealed,
                    "--on-policy-evidence-rounds",
                    "1",
                ]
            )
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="on-policy-manual-prompt-manifest",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    *sealed,
                    "--on-policy-evidence-rounds",
                    "1",
                    "--on-policy-fit-prompts",
                    str(tmp_path / "fit-prompts.json"),
                ]
            )
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="exceeds the bounded on_policy_max_rounds",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    *sealed,
                    "--on-policy-evidence-rounds",
                    "3",
                    "--on-policy-max-rounds",
                    "2",
                    "--on-policy-fit-prompts",
                    str(tmp_path / "fit-prompts.json"),
                    "--on-policy-manual-prompt-manifest",
                    str(tmp_path / "manual-prompts.json"),
                ]
            )
        )


def test_on_policy_extension_evidence_changes_every_derived_phase_key(
    sealed_parent_cache_fixture,
) -> None:
    """A base-only solve must never be reused for a base+extension request."""

    cache = sealed_parent_cache_fixture.cache
    base_splits, base_digest = MODULE._derive_sealed_cache_splits(cache)
    extension_identity = MODULE._ResolvedOnPolicyEvidence(
        extension_cache_paths=(Path("extension-cache-0"),),
        extension_cache_keys=("a" * 64,),
        extension_batch_counts=(2,),
        round_manifest_sha256=("b" * 64,),
        fit_prompts_disjoint=True,
    ).identity_sha256(cache.cache_key)
    extended_splits, extended_digest = MODULE._derive_sealed_cache_splits(
        cache,
        on_policy_evidence_identity_sha256=extension_identity,
    )

    assert extension_identity is not None
    assert extended_splits == base_splits
    assert extended_digest != base_digest
    # No extension evidence must leave the pre-extension digest untouched.
    assert MODULE._derive_sealed_cache_splits(
        cache,
        on_policy_evidence_identity_sha256=None,
    ) == (base_splits, base_digest)
    assert (
        MODULE._ResolvedOnPolicyEvidence().identity_sha256(cache.cache_key)
        is None
    )

    # Every row a child visited is a fit row; selection would let the child
    # pick its own frontier and validation would stop being held out.
    all_fit_splits, all_fit_digest = MODULE._derive_sealed_cache_splits(
        cache,
        all_rows_are_fit=True,
        on_policy_evidence_identity_sha256=extension_identity,
    )
    assert set(all_fit_splits) == {"fit"}
    assert len(all_fit_splits) == len(cache)
    assert all_fit_digest not in {base_digest, extended_digest}


def test_mid_body_capsule_resume_is_refused_under_moving_gauge_reclosure(
    tmp_path,
) -> None:
    """``compile_body_reclosure`` claims this process installed every layer."""

    from neural_forge.hf_lfm2_dense_budget_model_fold import (
        Lfm2DenseBudgetFoldConfig,
        Lfm2DenseBudgetFoldError,
    )

    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="requires this process to fold every layer",
    ):
        Lfm2DenseBudgetFoldConfig(
            body_reclosure_mode="moving_gauge_v3",
            run_closed_loop=True,
            restart_capsule_directory=tmp_path / "capsules",
            resume_from_restart_capsule=tmp_path / "capsules",
        )
    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="requires the restart-capsule directory",
    ):
        Lfm2DenseBudgetFoldConfig(
            run_closed_loop=True,
            resume_from_restart_capsule=tmp_path / "capsules",
        )


def test_on_policy_phase_dependency_is_added_only_when_evidence_exists() -> None:
    """The balanced-basis phase key must move with extension evidence."""

    def carrier_dependencies():
        return {"carrier_route": "1" * 64}

    assert MODULE._compose_on_policy_phase_dependencies(None, None) is None
    assert (
        MODULE._compose_on_policy_phase_dependencies(carrier_dependencies, None)
        is carrier_dependencies
    )

    composed = MODULE._compose_on_policy_phase_dependencies(
        carrier_dependencies,
        "e" * 64,
    )
    assert composed is not None
    assert dict(composed()) == {
        "carrier_route": "1" * 64,
        MODULE.ON_POLICY_PHASE_DEPENDENCY_NAME: "e" * 64,
    }

    standalone = MODULE._compose_on_policy_phase_dependencies(None, "f" * 64)
    assert standalone is not None
    assert dict(standalone()) == {
        MODULE.ON_POLICY_PHASE_DEPENDENCY_NAME: "f" * 64
    }

    def colliding():
        return {MODULE.ON_POLICY_PHASE_DEPENDENCY_NAME: "0" * 64}

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="collides with an existing external phase dependency",
    ):
        MODULE._compose_on_policy_phase_dependencies(colliding, "e" * 64)()


def test_tiny_dense_catalog_writes_and_resumes_accepted_layer_capsules(
    tiny_dense_lane_pack,
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    """Kill the body fold at a layer boundary, then resume from the capsule."""

    from neural_forge import hf_lfm2_closed_loop_restart_capsule as capsule_module

    pack = tiny_dense_lane_pack
    run = tmp_path / "run"
    capsule_root = tmp_path / "capsules"
    arguments = _tiny_dense_lane_arguments(
        pack,
        run,
        ["--dense-restart-capsule-root", str(capsule_root)],
    )
    killed_config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(arguments)
    )
    assert killed_config.dense_restart_capsule_directory == (
        capsule_root.absolute()
    )

    original_record = (
        capsule_module.Lfm2ClosedLoopRestartCapsuleWriter.record_accepted_layer
    )

    class _SimulatedProcessKill(RuntimeError):
        pass

    def kill_after_first_accepted_layer(self, **kwargs):
        record = original_record(self, **kwargs)
        assert record["layer_index"] == 0
        raise _SimulatedProcessKill("simulated kill at a layer boundary")

    monkeypatch.setattr(
        capsule_module.Lfm2ClosedLoopRestartCapsuleWriter,
        "record_accepted_layer",
        kill_after_first_accepted_layer,
    )
    killed_receipt = MODULE.run_production_catalog(killed_config)
    killed_stages = [
        row["stage"] for row in _progress_rows(capsys.readouterr().out)
    ]
    monkeypatch.undo()

    assert killed_receipt["status"] == "failed_closed"
    assert killed_config.publish_destination.exists() is False
    assert "dense_restart_capsule_writer_bound" in killed_stages
    assert "dense_closed_loop_completed" not in killed_stages
    # The kill lands inside ``record_accepted_layer`` after the capsule is
    # published and before the orchestrator can report it, which is exactly the
    # window a real process death occupies.
    references = capsule_module.discover_lfm2_closed_loop_restart_capsules(
        capsule_root
    )
    assert [value.capsule_sequence for value in references] == [0]
    assert [value.layer_index for value in references] == [0]

    resumed_config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [*arguments, "--dense-resume-from-capsule", str(capsule_root)]
        )
    )
    resumed_receipt = MODULE.run_production_catalog(resumed_config)
    resumed_rows = _progress_rows(capsys.readouterr().out)
    resumed_stages = [row["stage"] for row in resumed_rows]

    assert resumed_receipt["status"] == "complete", resumed_receipt.get(
        "failure"
    )
    assert resumed_receipt["restart_capsule_policy"]["resume_requested"] is True
    assert resumed_receipt["restart_capsule_policy"]["root"] == str(
        capsule_root.absolute()
    )
    assert "dense_restart_capsule_resume_loaded" in resumed_stages
    assert "child_closed_loop_restart_capsule_resume_admitted" in resumed_stages
    assert "child_closed_loop_layer_skipped_restart_resume" in resumed_stages
    assert "child_closed_loop_restart_capsule_published" in resumed_stages
    assert "dense_restart_capsule_run_completed" in resumed_stages

    resume_loaded = next(
        row
        for row in resumed_rows
        if row["stage"] == "dense_restart_capsule_resume_loaded"
    )
    assert resume_loaded["capsule_layer_index"] == 0
    assert resume_loaded["completed_layer_count"] == 1
    assert resume_loaded["resume_layer_index"] == 1
    assert resume_loaded["skipped_layers_are_not_replayed"] is True

    skipped = [
        row
        for row in resumed_rows
        if row["stage"] == "child_closed_loop_layer_skipped_restart_resume"
    ]
    executed = [
        row
        for row in resumed_rows
        if row["stage"] == "child_closed_loop_layer_started"
    ]
    assert [row["layer_index"] for row in skipped] == [0]
    assert [row["layer_index"] for row in executed] == [1, 2]
    assert [row["sequence"] for row in resumed_rows] == list(
        range(len(resumed_rows))
    )

    # The published run receipt records the replay contract that actually ran.
    assert resumed_receipt["closed_loop_replay_policy"]["resolved_mode"] == (
        MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert (
        resumed_receipt["closed_loop_replay_policy"][
            "in_loop_single_pass_certification_replay"
        ]
        is False
    )
    assert resumed_receipt["parent"]["parent_forward_count_after_cache"] == 0
    assert resumed_receipt["on_policy_evidence"]["round_count"] == 0
    assert (
        resumed_receipt["on_policy_evidence"]["parent_capture_invocations"] == 0
    )
    assert (resumed_config.publish_destination / "certificate.json").is_file()


def test_tiny_dense_catalog_accounts_on_policy_rounds_after_publication(
    tiny_dense_lane_pack,
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    """The extension round runs after publication and is counted separately."""

    pack = tiny_dense_lane_pack
    run = tmp_path / "run"
    fit_prompts = tmp_path / "on-policy-fit-prompts.json"
    fit_prompts.write_text(
        json.dumps(["4 b c", "b c d", "c d e"]),
        encoding="utf-8",
    )
    manual_manifest = tmp_path / "manual-review-prompts.json"
    manual_manifest.write_text(json.dumps(["e d c"]), encoding="utf-8")
    round_root = tmp_path / "on-policy-rounds"
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            _tiny_dense_lane_arguments(
                pack,
                run,
                [
                    "--on-policy-evidence-rounds",
                    "1",
                    "--on-policy-fit-prompts",
                    str(fit_prompts),
                    "--on-policy-manual-prompt-manifest",
                    str(manual_manifest),
                    "--on-policy-round-root",
                    str(round_root),
                    "--on-policy-max-new-tokens",
                    "2",
                    "--on-policy-max-context-tokens",
                    "16",
                    "--on-policy-max-prompts",
                    "3",
                ],
            )
        )
    )

    assert config.uses_on_policy_evidence is True
    assert config.on_policy_round_root_directory == round_root.absolute()

    observed: dict[str, object] = {}
    sealed_manifest = round_root / "round-000" / "round-manifest.json"

    def fake_round(**kwargs):
        observed.update(kwargs)
        sealed_manifest.parent.mkdir(parents=True, exist_ok=True)
        sealed_manifest.write_text("{}", encoding="ascii")
        return SimpleNamespace(
            round_index=kwargs["round_index"],
            max_rounds=kwargs["max_rounds"],
            manifest_path=sealed_manifest,
            manifest_sha256="a" * 64,
            request_sha256="b" * 64,
            extension_cache_path=round_root / "round-000" / "extension-cache",
            extension_cache_key="c" * 64,
            batch_count=3,
            parent_fingerprint=kwargs["parent_fingerprint"],
            manifest=MappingProxyType(
                {
                    "disjointness": {"verified_disjoint": True},
                    "counters": {
                        "parent_capture_invocations": 1,
                        "optimizer_steps": 0,
                        "backward_calls": 0,
                        "training_steps": 0,
                    },
                }
            ),
        )

    monkeypatch.setattr(MODULE, "run_lfm2_on_policy_evidence_round", fake_round)
    receipt = MODULE.run_production_catalog(config)
    rows = _progress_rows(capsys.readouterr().out)
    stages = [row["stage"] for row in rows]

    assert receipt["status"] == "complete", receipt.get("failure")
    # The round reloaded the child this run just published.
    assert observed["child_artifact_directory"] == (
        Path(receipt["artifact"]["path"]).resolve(strict=True) / "child"
    )
    assert observed["round_index"] == 0
    assert observed["max_rounds"] == MODULE.DEFAULT_ON_POLICY_MAX_ROUNDS
    assert tuple(observed["previous_round_manifest_paths"]) == ()
    assert tuple(observed["fit_prompts"]) == ("4 b c", "b c d", "c d e")
    assert observed["rollout"].max_new_tokens == 2
    assert observed["rollout"].max_context_tokens == 16
    assert Path(observed["base_cache_directory"]).resolve() == (
        Path(pack.sealed_cache).resolve()
    )

    # ``parent_forward_count_after_cache`` keeps its exact pre-extension
    # meaning; on-policy capture is accounted on its own counter.
    evidence = receipt["on_policy_evidence"]
    assert receipt["parent"]["parent_forward_count_after_cache"] == 0
    assert receipt["parent"]["parent_trace_regeneration_after_cache"] == 0
    assert receipt["parent"]["on_policy_parent_capture_invocations"] == 1
    assert evidence["parent_capture_invocations"] == 1
    assert evidence["parent_trace_regeneration_forward_count"] == 0
    assert evidence["postpublication_round_count"] == 1
    assert evidence["round_count"] == 0
    assert evidence["consumed_by_this_solve_pass"] is False
    assert evidence["all_rows_are_fit_rows"] is True
    assert evidence["fit_prompts_disjoint_from_manual_review"] is True
    assert (
        evidence["postpublication_rounds_executed"][0]["extension_cache_key"]
        == "c" * 64
    )

    started = stages.index("on_policy_evidence_round_started")
    sealed = stages.index("on_policy_evidence_round_sealed")
    assert stages.index("system_fold_completed") < started < sealed
    assert sealed < stages.index("success")
    assert [row["sequence"] for row in rows] == list(range(len(rows)))


def test_extension_evidence_consumption_fails_closed_on_unready_lanes(
    tmp_path,
) -> None:
    """No lane may silently solve base-only under a base+extension request."""

    parser = MODULE.build_parser()
    manifest = [
        "--sealed-parent-cache-directory",
        str(tmp_path / "sealed-cache"),
        "--on-policy-round-manifest",
        str(tmp_path / "rounds" / "round-000" / "round-manifest.json"),
    ]
    for reason, extra in (
        (
            "producer-first model fold binds one sealed parent cache key",
            ["--producer-first-model-fold"],
        ),
        (
            "macro-cell attractor gate binds one sealed cache handle",
            [
                "--compression-ratio",
                "4",
                "--dense-macro-cell-attractor-gate",
            ],
        ),
        (
            "vocabulary-margin candidate gate binds one sealed cache handle",
            [
                "--compression-ratio",
                "4",
                "--dense-vocabulary-margin-candidate-gate",
            ],
        ),
    ):
        with pytest.raises(
            MODULE.ProductionLfm2CatalogError,
            match=reason,
        ):
            MODULE.config_from_args(
                parser.parse_args(
                    [*_required_args(tmp_path), *manifest, *extra]
                )
            )
    # The dense budget lane consumes base+extension evidence now: its evidence
    # provider, structural basis, and closed-loop adapter all bind the merged
    # read-only evidence-set views.
    dense_consumer = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                *manifest,
                "--compression-ratio",
                "4",
                "--dense-heterogeneous-layout",
                "--dense-heterogeneous-target-parameters",
                "123456",
            ]
        )
    )
    assert (
        MODULE._unconsumed_extension_evidence_reason(dense_consumer) is None
    )
    dense_policy = MODULE._on_policy_evidence_policy(dense_consumer)
    assert dense_policy["consumes_extension_evidence"] is True
    assert dense_policy["extension_evidence_consumer_gap"] is None
    # The bridged carrier lane is still gated, and for a stated reason: the
    # sealed bridge trace adapter has no target-token preimage for a batch an
    # on-policy round tokenized with the parent tokenizer.
    bridged = SimpleNamespace(
        hybrid_model_fold=False,
        producer_first_model_fold=False,
        uses_tokenizer_bridge=True,
        uses_carrier_routes=False,
        dense_macro_cell_attractor_gate_enabled=False,
        dense_vocabulary_margin_candidate_gate_enabled=False,
    )
    assert "no target-token preimage" in str(
        MODULE._unconsumed_extension_evidence_reason(bridged)
    )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="no target-token preimage",
    ):
        MODULE._assert_extension_evidence_is_consumable(bridged)
    # Producing rounds stays available on every lane, including the gated ones.
    bridged_rounds = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--sealed-parent-cache-directory",
                str(tmp_path / "sealed-cache"),
                "--producer-first-model-fold",
                "--on-policy-evidence-rounds",
                "1",
                "--on-policy-fit-prompts",
                str(tmp_path / "fit-prompts.json"),
                "--on-policy-manual-prompt-manifest",
                str(tmp_path / "manual-prompts.json"),
            ]
        )
    )
    assert bridged_rounds.on_policy_evidence_rounds == 1
    assert bridged_rounds.on_policy_refold_after_rounds == 0
    assert (
        MODULE._unconsumed_extension_evidence_reason(bridged_rounds)
        is not None
    )
    receipt_policy = MODULE._on_policy_evidence_policy(bridged_rounds)
    assert receipt_policy["consumes_extension_evidence"] is False
    assert receipt_policy["extension_evidence_consumer_gap"] is not None
    assert receipt_policy["refold_after_rounds"] == 0


def test_refold_alternation_configuration_is_bounded_and_on_policy(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    sealed = [
        "--sealed-parent-cache-directory",
        str(tmp_path / "sealed-cache"),
        "--on-policy-fit-prompts",
        str(tmp_path / "fit-prompts.json"),
        "--on-policy-manual-prompt-manifest",
        str(tmp_path / "manual-prompts.json"),
    ]
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="needs at least one new round per pass",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    *sealed,
                    "--on-policy-refold-after-rounds",
                    "1",
                ]
            )
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="always reloads the child each pass published",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    *sealed,
                    "--on-policy-evidence-rounds",
                    "1",
                    "--on-policy-refold-after-rounds",
                    "1",
                    "--on-policy-child-artifact-directory",
                    str(tmp_path / "foreign-child"),
                ]
            )
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="alternation exceeds the bounded on_policy_max_rounds",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    *sealed,
                    "--on-policy-evidence-rounds",
                    "1",
                    "--on-policy-refold-after-rounds",
                    "3",
                ]
            )
        )
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                *sealed,
                "--on-policy-evidence-rounds",
                "1",
                "--on-policy-refold-after-rounds",
                "2",
            ]
        )
    )

    assert config.on_policy_refold_after_rounds == 2
    # The alternation knob never changes what one pass solves, so it stays out
    # of the canonical fold identity and out of the derived capsule scope.
    plain = replace(config, on_policy_refold_after_rounds=0)
    assert MODULE.canonical_sha256(config.identity()) == (
        MODULE.canonical_sha256(plain.identity())
    )
    assert "refold" not in set(
        config.identity()["target"]["on_policy_evidence"]
    )
    assert config.dense_restart_capsule_directory == (
        plain.dense_restart_capsule_directory
    )
    assert (
        MODULE._on_policy_evidence_policy(config)["refold_after_rounds"] == 2
    )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="must be driven by run_production_catalog_on_policy_alternation",
    ):
        MODULE.run_production_catalog(config)
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires --on-policy-refold-after-rounds",
    ):
        MODULE.run_production_catalog_on_policy_alternation(plain)


def test_alternation_pass_configs_change_only_evidence_and_destinations(
    tmp_path,
) -> None:
    """Every pass reuses the same blank, topology, budget, ledger, and rounds."""

    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_required_args(tmp_path),
                "--diagnostics",
                str(tmp_path / "diagnostics.json"),
                "--sealed-parent-cache-directory",
                str(tmp_path / "sealed-cache"),
                "--on-policy-fit-prompts",
                str(tmp_path / "fit-prompts.json"),
                "--on-policy-manual-prompt-manifest",
                str(tmp_path / "manual-prompts.json"),
                "--on-policy-evidence-rounds",
                "1",
                "--on-policy-refold-after-rounds",
                "2",
                "--on-policy-round-root",
                str(tmp_path / "rounds"),
            ]
        )
    )
    first_manifest = tmp_path / "rounds" / "round-000" / "round-manifest.json"
    second_manifest = tmp_path / "rounds" / "round-001" / "round-manifest.json"

    base_pass = MODULE._on_policy_alternation_pass_config(
        config,
        pass_index=0,
        consumed_round_manifests=(),
        rounds=1,
    )
    first_refold = MODULE._on_policy_alternation_pass_config(
        config,
        pass_index=1,
        consumed_round_manifests=(first_manifest,),
        rounds=1,
    )
    final_refold = MODULE._on_policy_alternation_pass_config(
        config,
        pass_index=2,
        consumed_round_manifests=(first_manifest, second_manifest),
        rounds=0,
    )

    assert base_pass.publish_destination == config.publish_destination
    assert base_pass.receipt_path == config.receipt_path
    assert base_pass.on_policy_round_manifests == ()
    assert base_pass.on_policy_refold_after_rounds == 0
    assert first_refold.publish_destination == (
        tmp_path / "published-round-0-refold"
    )
    assert first_refold.receipt_path == (
        tmp_path / "receipt-round-0-refold.json"
    )
    assert first_refold.diagnostics_path == (
        tmp_path / "diagnostics-round-0-refold.json"
    )
    assert first_refold.on_policy_round_manifests == (first_manifest,)
    assert first_refold.on_policy_evidence_rounds == 1
    assert final_refold.publish_destination == (
        tmp_path / "published-round-1-refold"
    )
    assert final_refold.on_policy_round_manifests == (
        first_manifest,
        second_manifest,
    )
    # The last pass consumes everything and runs no further round.
    assert final_refold.on_policy_evidence_rounds == 0
    for pass_config in (base_pass, first_refold, final_refold):
        assert pass_config.ledger_path == config.ledger_path
        assert pass_config.cache_root == config.cache_root
        assert pass_config.snapshot_root == config.snapshot_root
        assert pass_config.on_policy_round_root == config.on_policy_round_root
        assert pass_config.blank_seed == config.blank_seed
        assert pass_config.target_topology == config.target_topology
        assert pass_config.exact_target_unique_parameters == (
            config.exact_target_unique_parameters
        )
        assert pass_config.sealed_parent_cache_directory == (
            config.sealed_parent_cache_directory
        )
    # Distinct destinations mean distinct fold identities and distinct derived
    # restart-capsule chains.
    identities = {
        MODULE.canonical_sha256(value.identity())
        for value in (base_pass, first_refold, final_refold)
    }
    assert len(identities) == 3


_ALTERNATION_VOCABULARY = {
    "<unk>": 0,
    "<pad>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "4": 4,
    "b": 5,
    "c": 6,
    "d": 7,
    "e": 8,
}


def _build_tiny_alternation_lane(root: Path) -> SimpleNamespace:
    """The smallest real lane whose closed-form writers read merged rows.

    The layer lane fits every closed-form writer from the merged evidence rows,
    so an extension corpus is genuinely consumed here.  Lanes that bind exactly
    one sealed cache handle refuse extension evidence at configuration time.
    """

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    parent_directory = root / "parent"
    parent_config = transformers.Lfm2Config(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        parent = transformers.Lfm2ForCausalLM(parent_config).eval()
    MODULE.residual.atomic_save_local_checkpoint(
        parent,
        parent_directory,
        output_root=root,
    )
    tokenizer = Tokenizer(
        WordLevel(dict(_ALTERNATION_VOCABULARY), unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(parent_directory / "tokenizer.json"))
    (parent_directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    target_config = transformers.Lfm2Config(**parent_config.to_dict())
    target_config.intermediate_size = 5
    target_config.block_ff_dim = 5
    target_config.block_auto_adjust_ff_dim = False
    target_count = MODULE.residual.unique_parameter_count(
        transformers.Lfm2ForCausalLM(target_config)
    )
    calibration = root / "calibration.json"
    calibration.write_text(
        json.dumps(
            [
                {
                    "split": "selection",
                    "input_ids": [2, 4, 5],
                    "forced_decode_ids": [6, 7],
                },
                {
                    "split": "fit",
                    "input_ids": [2, 5, 6],
                    "forced_decode_ids": [7, 8],
                },
                {
                    "split": "validation",
                    "input_ids": [2, 6, 7],
                    "forced_decode_ids": [8, 4],
                },
            ]
        ),
        encoding="utf-8",
    )
    behavior_rows = root / "parent-output-rows.json"
    behavior_rows.write_text(
        json.dumps(
            [
                {
                    "id": "tiny",
                    "prompt": "4",
                    "category": "math",
                    "expected_response": "4",
                    "parent_output_text": "4 4 4 4",
                    "parent_output_ids": [4, 4, 4, 4],
                }
            ]
        ),
        encoding="utf-8",
    )
    loaded_calibration = MODULE.residual.load_calibration_batches(calibration)
    # Exactly the capture configuration the runner derives, so a round can
    # extend this corpus under the same sealed selection policy.
    capture_config = MODULE.Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=8,
        max_decode_steps=8,
        max_capture_bytes=128 * 1024 * 1024,
        logit_rtol=0.2,
        logit_atol=1.0,
        require_top1_agreement=False,
        decode_trace_mode=MODULE.CACHED_VERIFIED_TRACE_MODE,
    )
    parent_cache = MODULE.build_lfm2_circuit_trace_cache(
        root / "sealed-cache",
        parent,
        loaded_calibration.batches,
        config=capture_config,
        input_device="cpu",
    )
    envelope = root / "parent-output-envelope.json"
    envelope_receipt = seal_lfm2_cached_whole_output_envelope(
        behavior_rows,
        envelope,
        parent_cache_key=parent_cache.cache_key,
        parent_state_fingerprint=parent_cache.parent_fingerprint,
        tokenizer_fingerprint=MODULE.fingerprint_local_tokenizer(
            parent_directory
        ),
        generation_limit=4,
    )
    fit_prompts = root / "on-policy-fit-prompts.json"
    fit_prompts.write_text(
        json.dumps(["4 b c", "b c d", "c d e"]),
        encoding="utf-8",
    )
    manual_manifest = root / "manual-review-prompts.json"
    manual_manifest.write_text(
        json.dumps(
            [
                hashlib.sha256(value.encode("utf-8")).hexdigest()
                for value in ("e d 4", "d 4 b")
            ]
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        parent_directory=parent_directory,
        calibration=calibration,
        sealed_cache=parent_cache.path,
        sealed_cache_key=parent_cache.cache_key,
        cached_parent_outputs=envelope,
        cached_parent_outputs_sha256=envelope_receipt["file_sha256"],
        target_count=target_count,
        fit_prompts=fit_prompts,
        manual_manifest=manual_manifest,
    )


@pytest.fixture(scope="module")
def tiny_alternation_lane(tmp_path_factory):
    return _build_tiny_alternation_lane(
        tmp_path_factory.mktemp("tiny-alternation-lane")
    )


def test_tiny_real_catalog_alternates_fold_round_and_refold(
    tiny_alternation_lane,
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    """publish -> real round -> refold on base+extension -> publish again."""

    lane = tiny_alternation_lane
    run = tmp_path / "run"
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=lane.parent_directory,
        calibration=lane.calibration,
        sealed_parent_cache_directory=lane.sealed_cache,
        cached_parent_outputs=lane.cached_parent_outputs,
        cached_parent_outputs_sha256=lane.cached_parent_outputs_sha256,
        cache_root=run / "cache",
        snapshot_root=run / "snapshots",
        publish_destination=run / "published",
        ledger_path=run / "ledger.jsonl",
        receipt_path=run / "receipt.json",
        diagnostics_path=run / "diagnostics.json",
        device="cpu",
        dtype="float32",
        target_query_heads=4,
        target_key_value_heads=2,
        target_intermediate_size=5,
        exact_target_unique_parameters=lane.target_count,
        max_selection_rows=64,
        max_fit_rows=64,
        max_validation_rows=64,
        endpoint_ridge=0.0,
        max_endpoint_relative_rmse=1e6,
        min_endpoint_cosine_similarity=-1.0,
        min_residual_cosine=-1.0,
        whole_output_max_new_tokens=4,
        snapshot_chunk_bytes=1024 * 1024,
        on_policy_evidence_rounds=1,
        on_policy_refold_after_rounds=1,
        on_policy_fit_prompts=lane.fit_prompts,
        on_policy_manual_prompt_manifest=lane.manual_manifest,
        on_policy_round_root=run / "on-policy-rounds",
        on_policy_max_prompts=3,
        on_policy_max_new_tokens=2,
        on_policy_max_context_tokens=16,
    )

    # Observers only: the real loader and the real round still run.  They count
    # every parent forward and record how many happened before the sanctioned
    # capture inside the round.
    observed: list[dict[str, int | None]] = []
    real_loader = MODULE._load_cache_bound_parent
    real_round = MODULE.run_lfm2_on_policy_evidence_round

    def counting_loader(*args, **kwargs):
        model, dtype, name = real_loader(*args, **kwargs)
        record: dict[str, int | None] = {
            "forwards": 0,
            "before_round": None,
            "after_round": None,
        }
        observed.append(record)

        def count(_module, _inputs, _output):
            record["forwards"] = int(record["forwards"]) + 1

        model.register_forward_hook(count)
        return model, dtype, name

    def observing_round(**kwargs):
        record = observed[-1]
        record["before_round"] = record["forwards"]
        result = real_round(**kwargs)
        record["after_round"] = record["forwards"]
        return result

    monkeypatch.setattr(MODULE, "_load_cache_bound_parent", counting_loader)
    monkeypatch.setattr(
        MODULE,
        "run_lfm2_on_policy_evidence_round",
        observing_round,
    )
    summary = MODULE.run_production_catalog_on_policy_alternation(config)
    rows = _progress_rows(capsys.readouterr().out)

    assert summary["status"] == "complete", summary.get("failure")
    assert summary["schema"] == MODULE.ON_POLICY_REFOLD_ALTERNATION_SCHEMA
    assert summary["executed_pass_count"] == summary["expected_pass_count"] == 2
    assert len(summary["alternation_sha256"]) == 64
    assert (run / "receipt-on-policy-alternation.json").is_file()
    base_pass, refold_pass = summary["passes"]
    assert base_pass["role"] == "base_fold"
    assert refold_pass["role"] == "on_policy_refold"
    assert base_pass["status"] == refold_pass["status"] == "complete"

    base_receipt = json.loads(
        (run / "receipt.json").read_text(encoding="ascii")
    )
    refold_receipt = json.loads(
        (run / "receipt-round-0-refold.json").read_text(encoding="ascii")
    )
    round_manifest = json.loads(
        Path(base_pass["sealed_round_manifests"][0]).read_text(encoding="ascii")
    )

    # 1. The round ran against the child the first pass published, and the
    #    refold consumed exactly that sealed round.
    assert base_receipt["on_policy_evidence"]["postpublication_round_count"] == 1
    assert base_receipt["on_policy_evidence"]["consumed_by_this_solve_pass"] is False
    assert refold_pass["consumed_round_manifests"] == (
        base_pass["sealed_round_manifests"]
    )
    evidence = refold_receipt["on_policy_evidence"]
    assert evidence["consumed_by_this_solve_pass"] is True
    assert evidence["round_count"] == 1
    assert evidence["postpublication_round_count"] == 0
    extension_batches = int(round_manifest["extension_cache"]["batch_count"])
    assert evidence["extension_batch_counts"] == [extension_batches]
    evidence_set = evidence["evidence_set"]
    assert evidence_set["member_count"] == 2
    assert evidence_set["batch_count"] == 3 + extension_batches
    assert evidence_set["extension_fit_batch_count"] == extension_batches
    # 2. Extension rows are fit rows only: never selection, never validation.
    assert evidence_set["extension_selection_or_validation_batch_count"] == 0
    assert evidence_set["all_extension_rows_are_fit_rows"] is True
    assert evidence_set["parent_forward_calls"] == 0

    # 3. Every derived key moved with the extension evidence.
    assert base_receipt["on_policy_evidence"]["evidence_identity_sha256"] is None
    assert len(evidence["evidence_identity_sha256"]) == 64
    assert (
        refold_receipt["parent"]["evidence_split_sha256"]
        != base_receipt["parent"]["evidence_split_sha256"]
    )
    assert (
        refold_receipt["parent"]["parent_trace_rows_sha256"]
        != base_receipt["parent"]["parent_trace_rows_sha256"]
    )
    assert refold_receipt["parent"]["on_policy_extension_evidence_consumed"] is True
    assert "on_policy_extension_evidence_consumed" not in base_receipt["parent"]
    # The sealed base cache itself is unchanged and shared by both passes.
    assert (
        refold_receipt["parent"]["parent_cache_key"]
        == base_receipt["parent"]["parent_cache_key"]
        == lane.sealed_cache_key
    )
    base_certificate = json.loads(
        (run / "published" / "certificate.json").read_text(encoding="ascii")
    )
    refold_certificate = json.loads(
        (run / "published-round-0-refold" / "certificate.json").read_text(
            encoding="ascii"
        )
    )
    assert base_certificate["evidence_fingerprint"] == lane.sealed_cache_key
    assert (
        refold_certificate["evidence_fingerprint"]
        == evidence_set["evidence_set_cache_key"]
        != lane.sealed_cache_key
    )

    # 4. The extension rows really reached the closed-form solves: every layer
    #    gained exactly the extension rows as fit rows, and no validation row
    #    moved.
    materialized = next(
        row
        for row in rows
        if row["stage"] == "on_policy_extension_evidence_materialized"
    )
    assert materialized["member_count"] == 2
    assert int(materialized["extension_path_count"]) > 0
    assert int(materialized["merged_path_count"]) == (
        int(materialized["base_path_count"])
        + int(materialized["extension_path_count"])
    )
    assert materialized["parent_forwards"] == 0
    base_builds = json.loads(
        (run / "diagnostics.json").read_text(encoding="ascii")
    )["evidence_builds"]
    refold_builds = json.loads(
        (run / "diagnostics-round-0-refold.json").read_text(encoding="ascii")
    )["evidence_builds"]
    assert base_builds and len(base_builds) == len(refold_builds)
    for before, after in zip(base_builds, refold_builds):
        assert after["layer_index"] == before["layer_index"]
        assert after["fit_rows"] == before["fit_rows"] + int(
            materialized["extension_path_count"]
        )
        assert after["validation_rows"] == before["validation_rows"]
        assert after["parent_forward_count"] == before["parent_forward_count"] == 0

    # 5. Zero parent forwards outside the round's single sanctioned capture.
    assert len(observed) == 2
    assert observed[0]["before_round"] == 0
    assert observed[0]["after_round"] == observed[0]["forwards"]
    assert int(observed[0]["forwards"]) > 0
    assert observed[1]["before_round"] is None
    assert observed[1]["forwards"] == 0
    assert round_manifest["counters"]["parent_capture_invocations"] == 1
    assert round_manifest["counters"]["optimizer_steps"] == 0
    assert round_manifest["counters"]["backward_calls"] == 0
    assert round_manifest["counters"][
        "parent_forward_calls_during_child_rollout"
    ] == 0
    for receipt in (base_receipt, refold_receipt):
        assert receipt["parent"]["parent_forward_count_after_cache"] == 0
        assert receipt["parent"]["parent_trace_regeneration_after_cache"] == 0
        assert receipt["fold"]["optimizer_steps"] == 0
        assert receipt["fold"]["backward_calls"] == 0
    assert base_receipt["parent"]["on_policy_parent_capture_invocations"] == 1
    assert refold_receipt["parent"]["on_policy_parent_capture_invocations"] == 0

    # 6. Both artifacts are real, distinct, and freshly reloadable.
    assert base_receipt["artifact"]["fresh_reload_accepted"] is True
    assert refold_receipt["artifact"]["fresh_reload_accepted"] is True
    assert (
        base_receipt["target"]["exact_unique_parameters"]
        == refold_receipt["target"]["exact_unique_parameters"]
        == lane.target_count
    )
    for directory in ("published", "published-round-0-refold"):
        child = run / directory / "child"
        assert (child / "tokenizer.json").is_file()
        reloaded = MODULE.load_lfm2_direct_local_model(
            child,
            device="cpu",
            dtype=torch.float32,
        )
        assert int(reloaded.config.intermediate_size) == 5
        assert MODULE.residual.unique_parameter_count(reloaded) == (
            lane.target_count
        )

    # 7. Both fold attempts landed in one ledger, and each pass journals its
    #    own ordered progress.
    ledger_rows = [
        json.loads(line)
        for line in (run / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(ledger_rows) > len(base_builds)
    assert base_receipt["ledger"]["record_count"] < (
        refold_receipt["ledger"]["record_count"]
    )
    for pass_record in (base_pass, refold_pass):
        pass_rows = [
            row
            for row in rows
            if row["configuration_sha256"] == pass_record["configuration_sha256"]
        ]
        assert [row["sequence"] for row in pass_rows] == list(
            range(len(pass_rows))
        )
        assert pass_rows[-1]["stage"] == "final_receipt"
    assert base_pass["configuration_sha256"] != (
        refold_pass["configuration_sha256"]
    )
