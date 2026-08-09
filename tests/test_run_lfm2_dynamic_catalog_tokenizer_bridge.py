from __future__ import annotations

import io
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from experiments import run_lfm2_dynamic_catalog as MODULE  # noqa: E402
from neural_forge.dynamic_behavior_gate import WholeOutputCase  # noqa: E402
from neural_forge.dynamic_lfm2_behavior import Lfm2TaskContract  # noqa: E402
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (  # noqa: E402
    Lfm2StrictTokenizerBridgeTraceAdapter,
    Lfm2StrictTokenizerBridgeTraceError,
    Lfm2TargetTokenizedRecord,
    authenticate_lfm2_bridge_input_pair,
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    STRICT_TOKENIZER_BRIDGE_FILENAME,
    seal_strict_tokenizer_bridge_artifact,
)


def _write_wordlevel_tokenizer(
    directory: Path,
    vocabulary: dict[str, int],
) -> Any:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    directory.mkdir(parents=True)
    tokenizer = Tokenizer(
        WordLevel(
            vocabulary,
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
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
    return MODULE.residual.load_local_tokenizer(directory)


def _bridge_fixture(
    tmp_path: Path,
) -> tuple[Any, Any, Any, MODULE.ResolvedProductionTokenizerBridge]:
    donor_directory = tmp_path / "donor-tokenizer"
    target_directory = tmp_path / "target-tokenizer"
    donor = _write_wordlevel_tokenizer(
        donor_directory,
        {
            "<unk>": 0,
            "<bos>": 1,
            "B": 2,
            "C": 3,
            "<eos>": 4,
            "<pad>": 5,
        },
    )
    target = _write_wordlevel_tokenizer(
        target_directory,
        {
            "<bos>": 0,
            "B": 1,
            "<eos>": 2,
            "<pad>": 3,
            "<unk>": 4,
        },
    )
    artifact = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        target_directory,
        donor_row_capacity=8,
        target_row_capacity=5,
    )
    record = Lfm2TargetTokenizedRecord(
        record_id="batch-000000",
        split="selection",
        input_ids=torch.tensor([[0, 1, 2]]),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        forced_decode_ids=torch.tensor([[1, 2]]),
    )
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        artifact,
        (record,),
        expected_artifact_sha256=artifact.artifact_sha256,
    )
    calibration = SimpleNamespace(
        batches=(record.as_child_replay_batch(),),
        splits=("selection",),
        source_sha256="6" * 64,
    )
    paired = authenticate_lfm2_bridge_input_pair(
        adapter,
        record_id=record.record_id,
        parent_cache_batch=adapter.records[0].as_parent_cache_batch(),
        child_replay_batch=adapter.records[0].as_child_replay_batch(),
    )
    runtime = MODULE.ResolvedProductionTokenizerBridge(
        artifact=artifact,
        target_tokenizer=target,
        target_tokenizer_artifact_fingerprint=(
            MODULE.fingerprint_local_tokenizer(target_directory)
        ),
        trace_adapter=adapter,
        target_calibration=calibration,
        paired_input_evidence=(MappingProxyType(paired.to_dict()),),
        evidence_sha256=adapter.evidence_sha256,
        canonical_tokenization_evidence_sha256="d" * 64,
    )
    return donor, target, artifact, runtime


class _CacheBatch:
    def __init__(self, record: Any, *, missing_decode: bool = False) -> None:
        self.input_sha256 = record.donor_cache_input_sha256
        self.input_ids = record.donor_input_ids
        self.attention_mask = record.attention_mask
        self.forced_decode_ids = record.donor_forced_decode_ids
        tensors = {
            "input.input_ids": self.input_ids,
            "input.attention_mask": self.attention_mask,
            "input.forced_decode_ids": self.forced_decode_ids,
            "cached_prefill.logits": torch.arange(
                8,
                dtype=torch.float32,
            ).reshape(1, 1, 8),
            "decode.000.logits": torch.arange(
                8,
                dtype=torch.float32,
            ).reshape(1, 1, 8),
            "decode.001.logits": torch.arange(
                8,
                dtype=torch.float32,
            ).reshape(1, 1, 8),
        }
        if missing_decode:
            tensors.pop("decode.001.logits")
        self.tensors = MappingProxyType(tensors)


class _Cache:
    def __init__(self, batch: _CacheBatch) -> None:
        self.cache_key = "a" * 64
        self.parent_fingerprint = "b" * 64
        self.corpus_fingerprint = "c" * 64
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def load_batch(self, index: int) -> _CacheBatch:
        assert index == 0
        return self.batch


def _base_config(tmp_path: Path, **overrides: Any) -> Any:
    values = {
        "parent_model": tmp_path / "parent",
        "calibration": None,
        "cached_parent_outputs": tmp_path / "parent-output.json",
        "cached_parent_outputs_sha256": "0" * 64,
        "cache_root": tmp_path / "cache",
        "snapshot_root": tmp_path / "snapshots",
        "publish_destination": tmp_path / "published",
        "ledger_path": tmp_path / "ledger.jsonl",
        "receipt_path": tmp_path / "receipt.json",
        "sealed_parent_cache_directory": tmp_path / "sealed-cache",
        "compression_ratio": 4.0,
        "device": "cpu",
        "dtype": "float32",
    }
    values.update(overrides)
    return MODULE.ProductionLfm2CatalogConfig(**values)


def test_bridge_config_requires_complete_explicit_seals_and_structural_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="explicit target tokenizer",
    ):
        _base_config(
            tmp_path,
            target_tokenizer_directory=tmp_path / "target",
        )

    complete = {
        "target_tokenizer_directory": tmp_path / "target",
        "target_tokenizer_calibration": tmp_path / "target-calibration.json",
        "tokenizer_bridge_artifact_sha256": "1" * 64,
        "tokenizer_bridge_profile_sha256": "2" * 64,
        "tokenizer_bridge_mapped_parent_outputs": (
            tmp_path / "mapped-parent.json"
        ),
        "tokenizer_bridge_mapped_parent_outputs_sha256": "3" * 64,
    }
    config = _base_config(tmp_path, **complete)
    assert config.uses_tokenizer_bridge
    assert config.identity()["tokenizer_bridge"] == {
        "enabled": True,
        "artifact_sha256": "1" * 64,
        "transport_profile_sha256": "2" * 64,
        "mapped_parent_outputs": str(tmp_path / "mapped-parent.json"),
        "mapped_parent_outputs_sha256": "3" * 64,
        "minimum_parent_task_retention": 0.9,
        "trace_parent_input_mode": (
            "target-tokenizer-canonical-sequence-mapped-one-row-to-donor-ids"
        ),
        "trace_donor_canonical_sequence_equivalence_claimed": False,
        "whole_output_original_parent_mode": (
            "donor-tokenizer-canonical-prompt-and-generation"
        ),
        "whole_output_mapped_parent_mode": (
            "target-tokenizer-canonical-prompt-mapped-to-donor-ids"
        ),
        "explicit_verified_artifact_required": True,
        "numeric_token_id_equality_required": False,
    }
    closed_loop = _base_config(
        tmp_path,
        dense_run_closed_loop=True,
        **complete,
    )
    assert closed_loop.uses_tokenizer_bridge
    assert closed_loop.dense_run_closed_loop is True

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="macro-attractor",
    ):
        _base_config(
            tmp_path,
            dense_run_closed_loop=True,
            dense_macro_cell_attractor_gate_enabled=True,
            **complete,
        )

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="streaming v2",
    ):
        _base_config(
            tmp_path,
            dense_residual_basis_mode="circuit_balanced",
            dense_vocabulary_margin_observability=True,
            **complete,
        )

    target_margin = _base_config(
        tmp_path,
        dense_residual_basis_mode="circuit_balanced",
        dense_vocabulary_margin_observability=True,
        dense_streaming_vocabulary_margin_v2=True,
        **complete,
    )
    assert target_margin.dense_streaming_vocabulary_margin_v2 is True

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="candidate gating remains disabled",
    ):
        _base_config(
            tmp_path,
            dense_run_closed_loop=True,
            dense_residual_basis_mode="circuit_balanced",
            dense_vocabulary_margin_observability=True,
            dense_streaming_vocabulary_margin_v2=True,
            dense_vocabulary_margin_candidate_gate_enabled=True,
            **complete,
        )


def test_bridge_projection_uses_target_columns_and_fails_on_missing_logits(
    tmp_path: Path,
) -> None:
    _donor, _target, artifact, runtime = _bridge_fixture(tmp_path)
    record = runtime.trace_adapter.records[0]
    parent_cache = _Cache(_CacheBatch(record))

    projected = MODULE._project_bridge_parent_cache(
        parent_cache,
        runtime,
        top_k=5,
    )

    class _SelectedPayload:
        def __init__(self, tensors):
            self.tensors = MappingProxyType(dict(tensors))
            self.released = False

    class _SelectiveSession:
        def __init__(self):
            self.cache_key = parent_cache.cache_key
            self.path = tmp_path / "sealed-cache"
            self.batch_references = (
                SimpleNamespace(input_sha256=record.donor_cache_input_sha256),
            )
            self.payloads = []

        def __len__(self):
            return 1

        def load_selected_tensors(self, index, names):
            assert index == 0
            assert tuple(names) == tuple(sorted(set(names)))
            payload = _SelectedPayload(
                {name: parent_cache.batch.tensors[name] for name in names}
            )
            self.payloads.append(payload)
            return payload

        @staticmethod
        def release_selected_tensors(payload):
            payload.released = True

    parent_cache.path = tmp_path / "sealed-cache"
    session = _SelectiveSession()

    def forbidden_full_load(_index):
        raise AssertionError("selective bridge projection used load_batch")

    parent_cache.load_batch = forbidden_full_load
    selective_projected = MODULE._project_bridge_parent_cache(
        parent_cache,
        runtime,
        top_k=5,
        selective_session=session,
    )

    batch = projected.load_batch(0)
    assert not torch.equal(record.donor_input_ids, batch.input_ids)
    assert torch.equal(batch.input_ids, record.target_input_ids)
    assert tuple(batch.tensors["cached_prefill.logits"].shape) == (1, 1, 5)
    assert torch.equal(
        batch.tensors["cached_prefill.logits"],
        torch.arange(8, dtype=torch.float32)
        .reshape(1, 1, 8)
        .index_select(
            -1,
            torch.tensor(
                artifact.transport_profile.target_to_donor_row_indices
            ),
        ),
    )
    assert len(projected.projection_evidence) == 3
    assert projected.projection_evidence_sha256
    assert (
        selective_projected.projection_evidence_sha256
        == projected.projection_evidence_sha256
    )
    selective_batch = selective_projected.load_batch(0)
    assert selective_batch.input_sha256 == batch.input_sha256
    assert selective_batch.tensors.keys() == batch.tensors.keys()
    for name in batch.tensors:
        torch.testing.assert_close(
            selective_batch.tensors[name],
            batch.tensors[name],
        )
    assert len(session.payloads) == 1
    assert session.payloads[0].released is True

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="lacks full donor-vocabulary logits",
    ):
        MODULE._project_bridge_parent_cache(
            _Cache(_CacheBatch(record, missing_decode=True)),
            runtime,
            top_k=5,
        )


def _passing_trajectory_report() -> dict[str, Any]:
    return {
        "passed": True,
        "reasons": [],
        "score": 1.0,
        "teacher_forced": {"top1_agreement": 1.0},
        "free_run": {"token_match_fraction": 1.0, "lcp_fraction": 1.0},
        "mean_residual_cosine": 1.0,
        "mean_residual_relative_rmse": 0.0,
        "cached_full_self_consistent": True,
        "cached_full_top1_diagnostic_fraction": 1.0,
    }


def test_cached_trajectory_evaluator_builds_bridge_projection_once_for_three_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _donor, _target, _artifact, runtime = _bridge_fixture(tmp_path)
    record = runtime.trace_adapter.records[0]
    parent_cache = _Cache(_CacheBatch(record))
    parent_rows = SimpleNamespace(
        cache_key=parent_cache.cache_key,
        corpus_fingerprint=parent_cache.corpus_fingerprint,
        paths=("row",),
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=parent_cache,
        rows=parent_rows,
        calibration_batches=(record.as_child_replay_batch(),),
        calibration_splits=(record.split,),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(),
            validation_indices=(),
        ),
        tokenizer_bridge=runtime,
    )
    child_cache = SimpleNamespace(cache_key="d" * 64)
    child_rows = SimpleNamespace(trace_rows_sha256="e" * 64)
    monkeypatch.setattr(
        MODULE.residual,
        "build_child_trace_cache",
        lambda *_args, **_kwargs: (child_cache, child_rows),
    )
    monkeypatch.setattr(
        MODULE,
        "_bridge_aligned_parent_rows",
        lambda rows, _child, _bridge: (
            rows,
            MappingProxyType({"alignment": "test"}),
        ),
    )
    monkeypatch.setattr(
        MODULE.residual,
        "evaluate_trajectory_gate",
        lambda *_args, **_kwargs: _passing_trajectory_report(),
    )
    builds = 0
    original_project = MODULE._project_bridge_parent_cache

    def counted_project(*args, **kwargs):
        nonlocal builds
        builds += 1
        return original_project(*args, **kwargs)

    monkeypatch.setattr(
        MODULE,
        "_project_bridge_parent_cache",
        counted_project,
    )
    progress_stream = io.StringIO()
    evaluator = MODULE.CachedTrajectoryEvaluator(
        payload=payload,
        capture_config=MODULE.Lfm2CircuitTraceCaptureConfig(),
        gate_config=MODULE.residual.TrajectoryGateConfig(),
        final_cache_root=tmp_path / "child-final",
        reload_cache_root=tmp_path / "child-reload",
        progress=MODULE._ProgressJournal(
            "f" * 64,
            stream=progress_stream,
        ),
    )

    assert evaluator._evaluate_model(
        SimpleNamespace(config=SimpleNamespace()),
        phase="accepted_in_memory_child",
        cache_root=tmp_path / "child-final",
    ).accepted
    for _ in range(2):
        assert evaluator._evaluate_model(
            SimpleNamespace(config=SimpleNamespace()),
            phase="freshly_reloaded_child",
            cache_root=tmp_path / "child-reload",
        ).accepted

    assert builds == 1
    projections = [
        event["tokenizer_bridge_projection"] for event in evaluator.events
    ]
    assert [value["projection_cache_reused"] for value in projections] == [
        False,
        True,
        True,
    ]
    assert [value["projection_build_count"] for value in projections] == [
        1,
        1,
        1,
    ]
    assert len(
        {
            value["projection_evidence_sha256"]
            for value in projections
        }
    ) == 1
    progress_stages = [
        json.loads(line)["stage"]
        for line in progress_stream.getvalue().splitlines()
        if line.strip()
    ]
    assert progress_stages.count(
        "tokenizer_bridge_parent_projection_started"
    ) == 1
    assert progress_stages.count(
        "tokenizer_bridge_parent_projection_record_completed"
    ) == 1
    assert progress_stages.count(
        "tokenizer_bridge_parent_projection_completed"
    ) == 1
    assert progress_stages.count(
        "tokenizer_bridge_parent_projection_reused"
    ) == 2


def test_memoized_bridge_projection_reauthenticates_and_invalidates_bindings(
    tmp_path: Path,
) -> None:
    _donor, _target, _artifact, runtime = _bridge_fixture(tmp_path)
    record = runtime.trace_adapter.records[0]
    parent_cache = _Cache(_CacheBatch(record))
    binding = MODULE._bridge_projection_reuse_binding(
        parent_cache,
        runtime,
        top_k=5,
        selective_session=None,
    )
    projected = MODULE._project_bridge_parent_cache(
        parent_cache,
        runtime,
        top_k=5,
        reuse_binding=binding,
    )

    assert MODULE._authenticate_bridge_projected_parent_cache(
        projected,
        parent_cache,
        runtime,
        reuse_binding=binding,
    ) is projected
    changed_top_k = MODULE._bridge_projection_reuse_binding(
        parent_cache,
        runtime,
        top_k=4,
        selective_session=None,
    )
    assert changed_top_k["binding_sha256"] != binding["binding_sha256"]
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="authenticated binding",
    ):
        MODULE._authenticate_bridge_projected_parent_cache(
            projected,
            parent_cache,
            runtime,
            reuse_binding=changed_top_k,
        )

    projected._projections[0]._target_logits[0, 0, 0] += 1.0
    with pytest.raises(
        Lfm2StrictTokenizerBridgeTraceError,
        match="seal is invalid",
    ):
        MODULE._authenticate_bridge_projected_parent_cache(
            projected,
            parent_cache,
            runtime,
            reuse_binding=binding,
        )


def test_target_calibration_proves_target_canonical_segmentation(
    tmp_path: Path,
) -> None:
    _donor, target, _artifact, _runtime = _bridge_fixture(tmp_path)
    calibration_path = tmp_path / "target-calibration.json"
    rows = [
        {
            "split": "selection",
            "target_tokenization_mode": "tokenizer-call-add-special-tokens",
            "target_forced_decode_tokenization_mode": (
                "tokenizer-call-no-special-tokens"
            ),
            "target_input_texts": ["B"],
            "target_forced_decode_texts": ["B"],
            "input_ids": [1],
            "attention_mask": [1],
            "forced_decode_ids": [1],
        },
        {
            "split": "fit",
            "target_tokenization_mode": "tokenizer-call-add-special-tokens",
            "target_forced_decode_tokenization_mode": (
                "tokenizer-call-no-special-tokens"
            ),
            "target_input_texts": ["B"],
            "target_forced_decode_texts": ["B"],
            "input_ids": [1],
            "attention_mask": [1],
            "forced_decode_ids": [1],
        },
    ]
    calibration_path.write_text(json.dumps(rows), encoding="utf-8")
    calibration = MODULE.residual.load_calibration_batches(calibration_path)

    evidence_sha = MODULE._verify_target_calibration_canonical_tokenization(
        calibration_path,
        calibration,
        target,
    )

    assert len(evidence_sha) == 64

    rows[1]["input_ids"] = [0]
    calibration_path.write_text(json.dumps(rows), encoding="utf-8")
    tampered = MODULE.residual.load_calibration_batches(calibration_path)
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="not the sealed target tokenizer's canonical sequence",
    ):
        MODULE._verify_target_calibration_canonical_tokenization(
            calibration_path,
            tampered,
            target,
        )

    rows[1]["input_ids"] = [1]
    rows[1]["forced_decode_ids"] = [2]
    calibration_path.write_text(json.dumps(rows), encoding="utf-8")
    tampered = MODULE.residual.load_calibration_batches(calibration_path)
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="forced-decode IDs are not",
    ):
        MODULE._verify_target_calibration_canonical_tokenization(
            calibration_path,
            tampered,
            target,
        )


def test_bridge_resolution_uses_two_full_verifies_and_reports_pair_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor, _target, artifact, _runtime = _bridge_fixture(tmp_path)
    calibration_path = tmp_path / "target-resolution-calibration.json"
    rows = [
        {
            "split": "selection" if _index % 2 == 0 else "fit",
            "target_tokenization_mode": "tokenizer-call-add-special-tokens",
            "target_forced_decode_tokenization_mode": (
                "tokenizer-call-no-special-tokens"
            ),
            "target_input_texts": ["B"],
            "target_forced_decode_texts": ["B"],
            "input_ids": [1],
            "attention_mask": [1],
            "forced_decode_ids": [1],
        }
        for _index in range(17)
    ]
    calibration_path.write_text(json.dumps(rows), encoding="utf-8")
    calibration = MODULE.residual.load_calibration_batches(calibration_path)
    target_records = MODULE._target_tokenized_records(calibration)
    expected_adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        artifact,
        target_records,
        expected_artifact_sha256=artifact.artifact_sha256,
    )

    class _ManifestCache:
        manifest = {
            "corpus": {
                "ordered_batch_input_sha256": [
                    record.donor_cache_input_sha256
                    for record in expected_adapter.records
                ]
            }
        }

        def __len__(self) -> int:
            return len(expected_adapter.records)

    config = _base_config(
        tmp_path,
        target_tokenizer_directory=artifact.root_dir,
        target_tokenizer_calibration=calibration_path,
        tokenizer_bridge_artifact_sha256=artifact.artifact_sha256,
        tokenizer_bridge_profile_sha256=artifact.profile_fingerprint,
        tokenizer_bridge_mapped_parent_outputs=tmp_path / "mapped-parent.json",
        tokenizer_bridge_mapped_parent_outputs_sha256="3" * 64,
    )
    full_verify_calls = 0
    original_verify = Lfm2StrictTokenizerBridgeTraceAdapter.verify

    def counted_verify(self):
        nonlocal full_verify_calls
        full_verify_calls += 1
        return original_verify(self)

    monkeypatch.setattr(
        Lfm2StrictTokenizerBridgeTraceAdapter,
        "verify",
        counted_verify,
    )
    stream = io.StringIO()
    progress = MODULE._ProgressJournal("a" * 64, stream=stream)
    resolved = MODULE.resolve_production_tokenizer_bridge(
        config,
        donor_tokenizer=donor,
        parent_cache=_ManifestCache(),
        calibration_splits=calibration.splits,
        calibration_batches=tuple(
            record.as_parent_cache_batch()
            for record in expected_adapter.records
        ),
        progress=progress,
    )

    assert resolved is not None
    assert resolved.evidence_sha256 == expected_adapter.evidence_sha256
    assert len(resolved.paired_input_evidence) == 17
    assert full_verify_calls == 2
    events = [
        json.loads(line)
        for line in stream.getvalue().splitlines()
        if line.strip()
    ]
    assert [event["stage"] for event in events] == [
        "tokenizer_bridge_started",
        "tokenizer_bridge_live_tokenizers_authenticated",
        "tokenizer_bridge_pair_progress",
        "tokenizer_bridge_pair_progress",
        "tokenizer_bridge_pair_progress",
        "tokenizer_bridge_complete",
    ]
    assert [
        event["authenticated_pair_count"]
        for event in events
        if event["stage"] == "tokenizer_bridge_pair_progress"
    ] == [1, 16, 17]


def test_bridge_row_alignment_authenticates_geometry_not_numeric_ids(
    tmp_path: Path,
) -> None:
    _donor, _target, _artifact, runtime = _bridge_fixture(tmp_path)
    parent_rows = MODULE.residual.TraceRowIndex(
        paths=("batch.000000/prefill/row.0000",),
        path_splits=("selection",),
        layer_kinds=("short_conv",),
        layer_fields=MappingProxyType(
            {
                0: MappingProxyType(
                    {"residual.post_layer": torch.ones((1, 8))}
                )
            }
        ),
        cache_key="1" * 64,
        corpus_fingerprint="2" * 64,
        trace_rows_sha256="3" * 64,
    )
    child_rows = MODULE.residual.TraceRowIndex(
        paths=parent_rows.paths,
        path_splits=parent_rows.path_splits,
        layer_kinds=parent_rows.layer_kinds,
        layer_fields=MappingProxyType(
            {
                0: MappingProxyType(
                    {"residual.post_layer": torch.ones((1, 4))}
                )
            }
        ),
        cache_key="4" * 64,
        corpus_fingerprint="5" * 64,
        trace_rows_sha256="6" * 64,
    )

    aligned, evidence = MODULE._bridge_aligned_parent_rows(
        parent_rows,
        child_rows,
        runtime,
    )

    assert aligned.corpus_fingerprint == child_rows.corpus_fingerprint
    assert aligned.field(0, "residual.post_layer").shape == (1, 8)
    assert evidence["numeric_token_id_equality_required"] is False
    assert evidence["sequence_geometry_authenticated"] is True


def test_artifact_io_packages_target_tokenizer_and_bridge_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor, _target, artifact, runtime = _bridge_fixture(tmp_path)
    destination = tmp_path / "child-package"

    def fake_save(_model: Any, target: Path, *, output_root: Path) -> None:
        assert output_root == target.parent
        target.mkdir()

    monkeypatch.setattr(
        MODULE.residual,
        "atomic_save_local_checkpoint",
        fake_save,
    )
    io = MODULE.DirectLocalArtifactIO(
        tokenizer_directory=artifact.root_dir,
        tokenizer_fingerprint=runtime.target_tokenizer_artifact_fingerprint,
        device="cpu",
        dtype=torch.float32,
        tokenizer_bridge=runtime,
        donor_tokenizer=donor,
    )

    io.save(object(), destination)

    assert (destination / "tokenizer.json").is_file()
    # The sealed bridge manifest stays external provenance: its donor-named
    # seal keys are forbidden inside a standalone child package, so the
    # package carries a sanitized reference instead of the manifest.
    assert not (destination / STRICT_TOKENIZER_BRIDGE_FILENAME).exists()
    reference_path = destination / MODULE.BRIDGE_REFERENCE_FILENAME
    assert reference_path.is_file()
    reference = json.loads(reference_path.read_bytes())
    assert reference["schema"] == MODULE.TOKENIZER_BRIDGE_REFERENCE_SCHEMA
    assert reference["bridge_artifact_sha256"] == artifact.artifact_sha256
    assert reference["bridge_profile_sha256"] == artifact.profile_fingerprint
    assert reference["strict_bridge_manifest_is_external_provenance"] is True
    assert (
        MODULE.fingerprint_local_tokenizer(destination)
        == runtime.target_tokenizer_artifact_fingerprint
    )


def _original_behavior_baseline() -> Any:
    return MODULE.Lfm2CachedWholeOutputBaseline(
        source_path="original.json",
        source_sha256="1" * 64,
        envelope_sha256="2" * 64,
        parent_cache_key="a" * 64,
        parent_state_fingerprint="b" * 64,
        tokenizer_fingerprint="3" * 64,
        cases=(
            WholeOutputCase(
                case_id="tiny",
                prompt="What is two plus two?",
                parent_output="4",
                category="math",
                parent_fragment_like=False,
                parent_metadata=MODULE.WholeOutputMetadata(
                    generated_token_count=1,
                    generation_limit=4,
                    maximum_repeated_ngram_share=0.0,
                ),
                split="heldout",
            ),
        ),
        contracts=(
            (
                "tiny",
                Lfm2TaskContract(
                    case_id="tiny",
                    kind="numeric",
                    expected_numeric="4",
                ),
            ),
        ),
        generation_limit=4,
    )


def _mapped_parent_envelope(
    tmp_path: Path,
    runtime: MODULE.ResolvedProductionTokenizerBridge,
    parent_cache: _Cache,
    *,
    output: str,
) -> tuple[Path, str]:
    rows = tmp_path / f"mapped-rows-{output}.json"
    rows.write_text(
        json.dumps(
            [
                {
                    "id": "tiny",
                    "prompt": "What is two plus two?",
                    "mapped_parent_output_text": output,
                    "mapped_parent_output_ids": [2],
                }
            ]
        ),
        encoding="utf-8",
    )
    envelope = tmp_path / f"mapped-envelope-{output}.json"
    receipt = MODULE.seal_lfm2_tokenizer_bridge_mapped_parent_envelope(
        rows,
        envelope,
        original_parent_baseline=_original_behavior_baseline(),
        tokenizer_bridge=runtime,
        parent_cache_key=parent_cache.cache_key,
        parent_state_fingerprint=parent_cache.parent_fingerprint,
    )
    return envelope, receipt["file_sha256"]


def test_mapped_parent_normalization_is_sealed_distinct_and_fail_closed(
    tmp_path: Path,
) -> None:
    _donor, _target, artifact, runtime = _bridge_fixture(tmp_path)
    parent_cache = _Cache(_CacheBatch(runtime.trace_adapter.records[0]))
    envelope, envelope_sha = _mapped_parent_envelope(
        tmp_path,
        runtime,
        parent_cache,
        output="4",
    )
    config = _base_config(
        tmp_path,
        target_tokenizer_directory=artifact.root_dir,
        target_tokenizer_calibration=tmp_path / "target-calibration.json",
        tokenizer_bridge_artifact_sha256=artifact.artifact_sha256,
        tokenizer_bridge_profile_sha256=artifact.profile_fingerprint,
        tokenizer_bridge_mapped_parent_outputs=envelope,
        tokenizer_bridge_mapped_parent_outputs_sha256=envelope_sha,
    )

    mapped, control = MODULE._load_bridge_mapped_parent_baseline(
        config,
        original_baseline=_original_behavior_baseline(),
        tokenizer_bridge=runtime,
        parent_cache=parent_cache,
    )

    assert control["passed"]
    assert control["task_score_retention"] == 1.0
    assert control["correctness_retention"] == 1.0
    assert control["original_parent_input_mode"] == "donor-tokenizer-canonical"
    assert control["mapped_parent_input_mode"] == (
        "target-tokenizer-canonical-then-one-row-mapped-to-donor"
    )
    assert control["donor_canonical_sequence_equivalence_claimed"] is False
    assert control["canonical_behavior_input_evidence_sha256"]
    assert control["generation_protocol"] == {
        **MODULE.GENERATION_PROTOCOL,
        "max_new_tokens": 4,
    }
    assert control["generation_protocol_matches_original_parent"] is True
    assert mapped.generation_limit == 4
    assert mapped.cases[0].parent_metadata.generation_limit == 4
    assert mapped.cases[0].parent_output == "4"
    sealed = json.loads(envelope.read_text(encoding="ascii"))
    target_ids = sealed["rows"][0]["target_prompt_input_ids"]
    mapped_ids = sealed["rows"][0]["mapped_parent_input_ids"]
    assert target_ids
    assert mapped_ids == [
        runtime.artifact.transport_profile.target_to_donor_row_indices[
            token_id
        ]
        for token_id in target_ids
    ]

    protocol_tampered = dict(sealed)
    protocol_tampered["generation"] = {
        **MODULE.GENERATION_PROTOCOL,
        "max_new_tokens": 8,
    }
    protocol_tampered_body = dict(protocol_tampered)
    protocol_tampered_body.pop("envelope_sha256")
    protocol_tampered["envelope_sha256"] = MODULE.canonical_sha256(
        protocol_tampered_body
    )
    protocol_tampered_path = tmp_path / "mapped-envelope-tampered-protocol.json"
    MODULE._write_atomic_json(protocol_tampered_path, protocol_tampered)
    protocol_tampered_config = _base_config(
        tmp_path,
        target_tokenizer_directory=artifact.root_dir,
        target_tokenizer_calibration=tmp_path / "target-calibration.json",
        tokenizer_bridge_artifact_sha256=artifact.artifact_sha256,
        tokenizer_bridge_profile_sha256=artifact.profile_fingerprint,
        tokenizer_bridge_mapped_parent_outputs=protocol_tampered_path,
        tokenizer_bridge_mapped_parent_outputs_sha256=MODULE._file_sha256(
            protocol_tampered_path
        ),
    )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="execution contract is incomplete",
    ):
        MODULE._load_bridge_mapped_parent_baseline(
            protocol_tampered_config,
            original_baseline=_original_behavior_baseline(),
            tokenizer_bridge=runtime,
            parent_cache=parent_cache,
        )

    rejected_envelope, rejected_sha = _mapped_parent_envelope(
        tmp_path,
        runtime,
        parent_cache,
        output="5",
    )
    rejected_config = _base_config(
        tmp_path,
        target_tokenizer_directory=artifact.root_dir,
        target_tokenizer_calibration=tmp_path / "target-calibration.json",
        tokenizer_bridge_artifact_sha256=artifact.artifact_sha256,
        tokenizer_bridge_profile_sha256=artifact.profile_fingerprint,
        tokenizer_bridge_mapped_parent_outputs=rejected_envelope,
        tokenizer_bridge_mapped_parent_outputs_sha256=rejected_sha,
    )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="normalization control rejected before folding",
    ):
        MODULE._load_bridge_mapped_parent_baseline(
            rejected_config,
            original_baseline=_original_behavior_baseline(),
            tokenizer_bridge=runtime,
            parent_cache=parent_cache,
        )

    tampered = json.loads(envelope.read_text(encoding="ascii"))
    tampered["rows"][0]["mapped_parent_input_ids"][0] = 1
    tampered_body = dict(tampered)
    tampered_body.pop("envelope_sha256")
    tampered["envelope_sha256"] = MODULE.canonical_sha256(tampered_body)
    tampered_path = tmp_path / "mapped-envelope-tampered-input.json"
    MODULE._write_atomic_json(tampered_path, tampered)
    tampered_config = _base_config(
        tmp_path,
        target_tokenizer_directory=artifact.root_dir,
        target_tokenizer_calibration=tmp_path / "target-calibration.json",
        tokenizer_bridge_artifact_sha256=artifact.artifact_sha256,
        tokenizer_bridge_profile_sha256=artifact.profile_fingerprint,
        tokenizer_bridge_mapped_parent_outputs=tampered_path,
        tokenizer_bridge_mapped_parent_outputs_sha256=MODULE._file_sha256(
            tampered_path
        ),
    )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="canonical target-tokenized one-row mapped parent input",
    ):
        MODULE._load_bridge_mapped_parent_baseline(
            tampered_config,
            original_baseline=_original_behavior_baseline(),
            tokenizer_bridge=runtime,
            parent_cache=parent_cache,
        )


def test_bridge_normalization_does_not_offset_lost_parent_capability_with_gain(
    tmp_path: Path,
) -> None:
    _donor, _target, artifact, runtime = _bridge_fixture(tmp_path)
    parent_cache = _Cache(_CacheBatch(runtime.trace_adapter.records[0]))
    metadata = MODULE.WholeOutputMetadata(
        generated_token_count=1,
        generation_limit=4,
        maximum_repeated_ngram_share=0.0,
    )
    baseline = MODULE.Lfm2CachedWholeOutputBaseline(
        source_path="original-offset.json",
        source_sha256="1" * 64,
        envelope_sha256="2" * 64,
        parent_cache_key=parent_cache.cache_key,
        parent_state_fingerprint=parent_cache.parent_fingerprint,
        tokenizer_fingerprint="3" * 64,
        cases=(
            WholeOutputCase(
                case_id="original-pass",
                prompt="What is two plus two?",
                parent_output="4",
                category="math",
                parent_fragment_like=False,
                parent_metadata=metadata,
                split="heldout",
            ),
            WholeOutputCase(
                case_id="original-fail",
                prompt="What is three plus four?",
                parent_output="wrong",
                category="math",
                parent_fragment_like=False,
                parent_metadata=metadata,
                split="heldout",
            ),
        ),
        contracts=(
            (
                "original-pass",
                Lfm2TaskContract(
                    case_id="original-pass",
                    kind="numeric",
                    expected_numeric="4",
                ),
            ),
            (
                "original-fail",
                Lfm2TaskContract(
                    case_id="original-fail",
                    kind="numeric",
                    expected_numeric="7",
                ),
            ),
        ),
        generation_limit=4,
    )
    source = tmp_path / "mapped-offset-rows.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "original-pass",
                    "prompt": "What is two plus two?",
                    "mapped_parent_output_text": "5",
                    "mapped_parent_output_ids": [2],
                },
                {
                    "id": "original-fail",
                    "prompt": "What is three plus four?",
                    "mapped_parent_output_text": "7",
                    "mapped_parent_output_ids": [2],
                },
            ]
        ),
        encoding="utf-8",
    )
    envelope = tmp_path / "mapped-offset-envelope.json"
    receipt = MODULE.seal_lfm2_tokenizer_bridge_mapped_parent_envelope(
        source,
        envelope,
        original_parent_baseline=baseline,
        tokenizer_bridge=runtime,
        parent_cache_key=parent_cache.cache_key,
        parent_state_fingerprint=parent_cache.parent_fingerprint,
    )
    config = _base_config(
        tmp_path,
        target_tokenizer_directory=artifact.root_dir,
        target_tokenizer_calibration=tmp_path / "target-calibration.json",
        tokenizer_bridge_artifact_sha256=artifact.artifact_sha256,
        tokenizer_bridge_profile_sha256=artifact.profile_fingerprint,
        tokenizer_bridge_mapped_parent_outputs=envelope,
        tokenizer_bridge_mapped_parent_outputs_sha256=receipt["file_sha256"],
    )

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="correctness_retention=0.000000",
    ):
        MODULE._load_bridge_mapped_parent_baseline(
            config,
            original_baseline=baseline,
            tokenizer_bridge=runtime,
            parent_cache=parent_cache,
        )


def _behavior_binding(name: str, baseline: str) -> Any:
    body = {
        "schema": "test-behavior-binding-v1",
        "gate_id": name,
        "gate_version": 1,
        "baseline_fingerprint": baseline,
    }
    fingerprint = MODULE.canonical_sha256(body)
    return MODULE.BehaviorGateBinding(
        gate_id=name,
        gate_version=1,
        baseline_fingerprint=baseline,
        manifest_fingerprint=fingerprint,
        manifest={**body, "manifest_fingerprint": fingerprint},
    )


class _BehaviorEvaluatorStub:
    def __init__(self, name: str, *, accepted: bool, baseline: str) -> None:
        self.accepted = accepted
        self.events: list[dict[str, Any]] = []
        MODULE.expose_behavior_gate_binding(
            self,
            _behavior_binding(name, baseline),
        )

    def evaluate_model(self, _model: Any, *, phase: str) -> Any:
        self.events.append({"phase": phase})
        if self.accepted:
            return MODULE.FoldDecision.accept("accepted", {"score": 1.0})
        return MODULE.FoldDecision.reject("rejected", {"score": 0.0})


def test_dual_child_gate_requires_mapped_fidelity_and_original_retention() -> None:
    mapped = _BehaviorEvaluatorStub(
        "mapped",
        accepted=True,
        baseline="1" * 64,
    )
    original = _BehaviorEvaluatorStub(
        "original",
        accepted=False,
        baseline="2" * 64,
    )
    normalization = {
        "passed": True,
        "control_sha256": "3" * 64,
    }
    gate = MODULE.BridgeDualWholeOutputBehaviorEvaluator(
        original_evaluator=original,
        mapped_evaluator=mapped,
        normalization_control=normalization,
    )

    rejected = gate.evaluate_model(object(), phase="in-memory")

    assert not rejected.accepted
    assert len(mapped.events) == 1
    assert len(original.events) == 1
    assert gate.events[0]["exact_token_equality_required"] is False
    MODULE.resolve_behavior_gate_binding(gate)

    accepted_gate = MODULE.BridgeDualWholeOutputBehaviorEvaluator(
        original_evaluator=_BehaviorEvaluatorStub(
            "original-accepted",
            accepted=True,
            baseline="4" * 64,
        ),
        mapped_evaluator=_BehaviorEvaluatorStub(
            "mapped-accepted",
            accepted=True,
            baseline="5" * 64,
        ),
        normalization_control=normalization,
    )
    assert accepted_gate.evaluate_model(object(), phase="reload").accepted


def test_production_wiring_selects_verified_bridge_target_semantics(
    tmp_path: Path,
) -> None:
    import transformers

    donor, _target, artifact, runtime = _bridge_fixture(tmp_path)
    parent = transformers.Lfm2ForCausalLM(
        transformers.Lfm2Config(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            layer_types=["conv", "full_attention"],
            conv_L_cache=3,
            conv_bias=False,
            block_auto_adjust_ff_dim=False,
            tie_word_embeddings=True,
        )
    ).eval()
    parent_graph = MODULE.inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=artifact.donor_tokenizer_semantics_sha256,
    )
    config = _base_config(
        tmp_path,
        target_tokenizer_directory=artifact.root_dir,
        target_tokenizer_calibration=tmp_path / "target-calibration.json",
        tokenizer_bridge_artifact_sha256=artifact.artifact_sha256,
        tokenizer_bridge_profile_sha256=artifact.profile_fingerprint,
        tokenizer_bridge_mapped_parent_outputs=tmp_path / "mapped.json",
        tokenizer_bridge_mapped_parent_outputs_sha256="0" * 64,
        compression_ratio=1.01,
        dense_minimum_hidden_size=8,
        dense_maximum_candidates=16,
        dense_residual_basis_mode="circuit_balanced",
        dense_vocabulary_margin_observability=True,
        dense_streaming_vocabulary_margin_v2=True,
    )

    class _Evaluator:
        def evaluate(self, _context: Any) -> bool:
            return True

    wiring = MODULE.build_production_catalog_wiring(
        config,
        parent_cache_payload=SimpleNamespace(cache_key="f" * 64),
        parent_graph=parent_graph,
        tokenizer_fingerprint=artifact.donor_tokenizer_semantics_sha256,
        evidence_builder=lambda _request: None,
        final_evaluator=_Evaluator(),
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: parent,
        fresh_reload_evaluator=lambda _model: True,
        ledger=MODULE.ExperimentLedger(),
        device="cpu",
        dtype=torch.float32,
        tokenizer_bridge=runtime,
    )

    assert wiring.budget_plan is not None
    rank = wiring.budget_plan.target_certificate.selected_candidate_rank
    selected = wiring.budget_plan.frontier.candidates[rank]
    assert selected.tokenizer_bridge_required
    assert selected.target_vocabulary_size == 5
    assert selected.missing_compiler_capabilities == ()
    assert (
        wiring.target_request.tokenizer_fingerprint
        == artifact.target_tokenizer_semantics_sha256
    )
    margin_binding = (
        wiring.stack_adapter.resources
        .target_tokenizer_vocabulary_margin_binding
    )
    assert margin_binding is not None
    assert margin_binding.trace_adapter is runtime.trace_adapter
    assert margin_binding.as_dict()["target_calibration_sha256"] == "6" * 64


def test_target_vocabulary_margin_binding_rejects_tampered_runtime_receipt(
    tmp_path: Path,
) -> None:
    _donor, _target, _artifact, runtime = _bridge_fixture(tmp_path)
    binding = MODULE.Lfm2TargetTokenizerVocabularyMarginBinding(
        trace_adapter=runtime.trace_adapter,
        runtime_receipt=runtime.receipt_fields(),
    )
    assert binding.as_dict()["target_to_donor_row_indices_sha256"] == (
        MODULE.canonical_sha256(list(runtime.trace_adapter.mapping))
    )

    tampered = runtime.receipt_fields()
    tampered["target_to_donor_row_indices_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="runtime.*seal"):
        MODULE.Lfm2TargetTokenizerVocabularyMarginBinding(
            trace_adapter=runtime.trace_adapter,
            runtime_receipt=tampered,
        )

    resealed_mapping_tamper = runtime.receipt_fields()
    resealed_mapping_tamper.pop("runtime_receipt_sha256")
    resealed_mapping_tamper["target_to_donor_row_indices_sha256"] = "e" * 64
    resealed_mapping_tamper["runtime_receipt_sha256"] = (
        MODULE.canonical_sha256(resealed_mapping_tamper)
    )
    with pytest.raises(ValueError, match="differs from its artifact, row map"):
        MODULE.Lfm2TargetTokenizerVocabularyMarginBinding(
            trace_adapter=runtime.trace_adapter,
            runtime_receipt=resealed_mapping_tamper,
        )

    incomplete = runtime.receipt_fields()
    incomplete.pop("runtime_receipt_sha256")
    incomplete.pop("canonical_target_tokenization_evidence_sha256")
    incomplete["runtime_receipt_sha256"] = MODULE.canonical_sha256(incomplete)
    with pytest.raises(ValueError, match="canonical_target_tokenization"):
        MODULE.Lfm2TargetTokenizerVocabularyMarginBinding(
            trace_adapter=runtime.trace_adapter,
            runtime_receipt=incomplete,
        )


def test_failed_receipt_planned_target_helper_preserves_parameter_evidence() -> None:
    configuration = (
        ("hidden_size", 64),
        (MODULE.TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY, "7" * 64),
        (MODULE.TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY, "8" * 64),
        (MODULE.PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY, "9" * 64),
        (MODULE.CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY, "a" * 64),
    )
    candidate = SimpleNamespace(
        fingerprint="b" * 64,
        source_unique_parameters=1_000,
        requested_unique_parameters=100,
        unique_parameters=96,
        logical_parameters=102,
        native_tied_parameter_count=6,
        compression_ratio=1_000 / 96,
        budget_delta=-4,
        at_or_below_budget=True,
        family="lfm2",
        modalities=("text",),
        ordered_source_spans=((0,), (1,)),
        configuration=configuration,
        target_vocabulary_size=5,
        tokenizer_bridge_required=True,
        required_compiler_capabilities=("tokenizer_bridge",),
        declared_compiler_capabilities=("tokenizer_bridge",),
        missing_compiler_capabilities=(),
        compiler_executable=True,
    )
    plan = SimpleNamespace(
        fingerprint="c" * 64,
        frontier=SimpleNamespace(
            fingerprint="d" * 64,
            candidates=(candidate,),
            request=SimpleNamespace(compression_ratio="10"),
        ),
        target_certificate=SimpleNamespace(
            selected_candidate_rank=0,
            fingerprint="e" * 64,
            selection_policy="supported_pareto_balanced_executable_v1",
        ),
    )
    wiring = MODULE.ProductionCatalogWiring(
        system=object(),
        target_request=object(),
        target_adapter=object(),
        stack_adapter=object(),
        budget_plan=plan,
    )

    evidence = MODULE._planned_dense_target_evidence(wiring)

    assert evidence is not None
    assert evidence["source_unique_parameters"] == 1_000
    assert evidence["planned_unique_parameters"] == 96
    assert evidence["planned_parameters_removed"] == 904
    assert evidence["planned_compression_ratio"] == pytest.approx(1_000 / 96)
    assert evidence["tokenizer_bridge_artifact_sha256"] == "7" * 64
    assert evidence["tokenizer_transport_profile_sha256"] == "8" * 64
