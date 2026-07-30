from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments import run_lfm2_functional_output_refit as runner
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_sparse_semantic_codebook import (
    compile_lfm2_sparse_semantic_codebook,
)


def _case():
    anchors = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    donor = torch.cat(
        (anchors, torch.tensor([[0.7, -0.2, 0.4, 0.3]], dtype=torch.float32))
    )
    codebook = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        torch.arange(4, dtype=torch.int64),
        maximum_candidates=4,
        active_k=4,
        candidate_ridge=1e-5,
        refit_ridge=1e-5,
        donor_rows_per_chunk=5,
        selection_device="cpu",
        selection_dtype=torch.float64,
        solve_device="cpu",
        solve_dtype=torch.float64,
    )
    hidden = torch.tensor(
        [
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.1, 0.0],
            [0.1, 0.0, 1.0, 0.0],
            [0.0, 0.1, 0.0, 1.0],
            [0.8, -0.1, 0.3, 0.2],
            [0.3, 0.4, 0.1, 0.2],
        ],
        dtype=torch.float32,
    )
    winners = (hidden @ donor.transpose(0, 1)).argmax(dim=-1)
    batch_ids = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.int64)
    body = {
        "schema": runner.TRACE_ROWS_SCHEMA,
        "trace_cache_key": "d" * 64,
        "trace_parent_fingerprint": "c" * 64,
        "trace_batch_count": 4,
        "selective_session_sha256": "e" * 64,
        "selected_pair_count": 6,
        "selected_logical_bytes": 1,
        "row_count": 6,
        "hidden_size": 4,
        "hidden_storage_dtype": "float32",
        "hidden_sha256": runner._tensor_sha256(hidden),
        "winner_sha256": runner._tensor_sha256(winners),
        "batch_id_sha256": runner._tensor_sha256(batch_ids),
        "operation_audit": dict(runner._OPERATION_AUDIT),
    }
    rows = runner.CachedFunctionalRows(
        hidden,
        winners,
        batch_ids,
        {**body, "receipt_sha256": canonical_sha256(body)},
    )
    return donor, anchors, codebook, rows


def _argv(tmp_path, *, codebook_sha: str, inventory_sha: str, rows_per_chunk: int):
    return [
        "--parent-model-directory",
        str(tmp_path / "parent"),
        "--expected-parent-identity-sha256",
        "c" * 64,
        "--expected-parent-source-inspection-sha256",
        "a" * 64,
        "--parent-tensor-key",
        "model.embed_tokens.weight",
        "--checkpoint-directory",
        str(tmp_path / "checkpoint"),
        "--expected-checkpoint-artifact-inventory-sha256",
        inventory_sha,
        "--target-tensor-key",
        "model.embed_tokens.weight",
        "--codebook-directory",
        str(tmp_path / "codebook"),
        "--expected-codebook-artifact-sha256",
        codebook_sha,
        "--expected-codebook-donor-row-count",
        "5",
        "--expected-codebook-target-row-count",
        "4",
        "--expected-hidden-size",
        "4",
        "--trace-cache-directory",
        str(tmp_path / "trace"),
        "--expected-trace-cache-key",
        "d" * 64,
        "--expected-trace-parent-fingerprint",
        "c" * 64,
        "--expected-trace-batch-count",
        "4",
        "--expected-trace-row-count",
        "6",
        "--output-directory",
        str(tmp_path / "output"),
        "--holdout-modulus",
        "4",
        "--holdout-remainder",
        "3",
        "--functional-weight",
        "0.001",
        "--solve-ridge",
        "0.0001",
        "--metric-device",
        "cpu",
        "--metric-dtype",
        "fp64",
        "--solve-device",
        "cpu",
        "--solve-dtype",
        "fp64",
        "--rows-per-chunk",
        str(rows_per_chunk),
        "--score-device",
        "cpu",
        "--score-dtype",
        "fp32",
        "--score-rows-per-chunk",
        "2",
    ]


def _install_local_artifact_stubs(monkeypatch, tmp_path):
    donor, anchors, codebook, rows = _case()
    for name in ("parent", "checkpoint", "codebook", "trace"):
        (tmp_path / name).mkdir()
    inventory = [{"relative_path": "sealed", "size_bytes": 1, "sha256": "b" * 64}]
    inventory_sha = canonical_sha256(inventory)
    monkeypatch.setattr(
        runner,
        "load_lfm2_sparse_semantic_codebook",
        lambda directory, expected_artifact_sha256: codebook,
    )
    monkeypatch.setattr(
        runner,
        "inspect_parent_tied_table_source",
        lambda directory, tensor_key: {"inspection_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        runner, "_checkpoint_artifact_inventory", lambda directory: inventory
    )

    def load_table(directory, **kwargs):
        table = donor if directory == (tmp_path / "parent").resolve() else anchors
        return SimpleNamespace(
            table=table.clone(),
            receipt={
                "schema": "authenticated-selective-local-table-test",
                "published_table_sha256": runner._tensor_sha256(table),
                "parent_model_objects_constructed": 0,
            },
        )

    monkeypatch.setattr(
        runner, "load_parent_tied_rows_from_local_safetensors", load_table
    )
    monkeypatch.setattr(
        runner,
        "load_cached_functional_rows",
        lambda **kwargs: rows,
    )
    return codebook, inventory_sha


def test_runner_reuses_math_cache_across_chunking_and_never_builds_a_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    codebook, inventory_sha = _install_local_artifact_stubs(monkeypatch, tmp_path)
    parser = runner.build_parser()
    first_events: list[dict[str, object]] = []
    first = runner.run(
        parser.parse_args(
            _argv(
                tmp_path,
                codebook_sha=codebook.artifact_sha256,
                inventory_sha=inventory_sha,
                rows_per_chunk=1,
            )
        ),
        progress=lambda event: first_events.append(dict(event)),
    )
    second_events: list[dict[str, object]] = []
    second = runner.run(
        parser.parse_args(
            _argv(
                tmp_path,
                codebook_sha=codebook.artifact_sha256,
                inventory_sha=inventory_sha,
                rows_per_chunk=17,
            )
        ),
        progress=lambda event: second_events.append(dict(event)),
    )

    assert first["math_cache_key"] == second["math_cache_key"]
    assert first["math_cache_source"] == "computed-and-atomically-published"
    assert second["math_cache_source"] == "authenticated-cache-hit"
    assert first["operation_audit"] == runner._OPERATION_AUDIT
    assert first["cached_score"]["all"]["row_count"] == 6
    assert first["cached_score"]["fit"]["row_count"] == 5
    assert first["cached_score"]["holdout"]["row_count"] == 1
    assert any(event["event"] == "math-cache-miss" for event in first_events)
    assert any(event["event"] == "math-cache-hit" for event in second_events)
    assert not any(event["event"] == "refit-start" for event in second_events)
    cache_directory = tmp_path / "output" / "math-cache" / first["math_cache_key"]
    assert {path.name for path in cache_directory.iterdir()} == {
        runner.COEFFICIENT_FILENAME,
        runner.RESULT_FILENAME,
        runner.PUBLICATION_FILENAME,
    }
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "AutoModel" not in source
    assert "load_lfm2_direct_local_model" not in source
    assert ".forward(" not in source
    assert ".generate(" not in source


def test_parser_requires_all_geometry_and_numeric_placement_controls() -> None:
    actions = {action.dest: action for action in runner.build_parser()._actions}
    for name in (
        "expected_codebook_donor_row_count",
        "expected_codebook_target_row_count",
        "expected_hidden_size",
        "expected_trace_batch_count",
        "expected_trace_row_count",
        "metric_device",
        "metric_dtype",
        "solve_device",
        "solve_dtype",
        "score_device",
        "score_dtype",
    ):
        assert actions[name].required is True


def test_trace_loader_uses_selective_authenticated_terminal_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    trace_path = tmp_path / "trace"
    trace_path.mkdir()
    hidden_by_batch = (
        torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
        torch.tensor([[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]),
    )
    logits_by_batch = (
        torch.tensor([[[5.0, 1.0, 0.0, -1.0, 2.0], [0.0, 4.0, 1.0, 0.0, 2.0]]]),
        torch.tensor([[[0.0, 1.0, 6.0, 0.0, 2.0], [0.0, 1.0, 2.0, 7.0, 3.0]]]),
    )
    references = (
        SimpleNamespace(
            index=0,
            ordered_tensor_names=(
                "cached_prefill.logits",
                "cached_prefill.output.lm_head_input",
            ),
        ),
        SimpleNamespace(
            index=1,
            ordered_tensor_names=(
                "decode.000.full_prefix_logits",
                "decode.000.output.lm_head_input",
            ),
        ),
    )

    class Session:
        batch_references = references

        def __init__(self):
            self.released = 0
            self.verified = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def load_selected_tensors(self, index, names):
            phase = "cached_prefill" if index == 0 else "decode.000"
            logit_name = (
                f"{phase}.logits"
                if index == 0
                else f"{phase}.full_prefix_logits"
            )
            tensors = {
                f"{phase}.output.lm_head_input": hidden_by_batch[index],
                logit_name: logits_by_batch[index],
            }
            assert tuple(sorted(tensors)) == names
            return SimpleNamespace(tensors=tensors, logical_bytes=72)

        def release_selected_tensors(self, payload):
            self.released += 1

        def verify(self):
            self.verified += 1
            return self

        def as_dict(self):
            return {"session_sha256": "e" * 64}

    session = Session()
    cache = SimpleNamespace(
        cache_key="d" * 64,
        parent_fingerprint="c" * 64,
        batch_count=2,
        selective_authenticated_session=lambda: session,
    )
    monkeypatch.setattr(
        runner,
        "Lfm2CircuitTraceCache",
        SimpleNamespace(open=lambda path, verify: cache),
    )
    events: list[dict[str, object]] = []

    rows = runner.load_cached_functional_rows(
        trace_cache_directory=trace_path,
        expected_cache_key="d" * 64,
        expected_parent_fingerprint="c" * 64,
        expected_batch_count=2,
        expected_row_count=4,
        donor_row_count=5,
        hidden_size=4,
        progress=lambda event: events.append(dict(event)),
    )

    assert rows.winners.tolist() == [0, 1, 2, 3]
    assert rows.batch_ids.tolist() == [0, 0, 1, 1]
    assert rows.receipt["selected_pair_count"] == 2
    assert rows.receipt["selected_logical_bytes"] == 144
    assert session.released == 2
    assert session.verified == 1
    assert [event["event"] for event in events] == [
        "trace-session-start",
        "trace-batch-complete",
        "trace-batch-complete",
        "trace-session-complete",
    ]


def test_publication_tampering_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    codebook, inventory_sha = _install_local_artifact_stubs(monkeypatch, tmp_path)
    args = runner.build_parser().parse_args(
        _argv(
            tmp_path,
            codebook_sha=codebook.artifact_sha256,
            inventory_sha=inventory_sha,
            rows_per_chunk=2,
        )
    )
    result = runner.run(args, progress=lambda event: None)
    cache_directory = tmp_path / "output" / "math-cache" / result["math_cache_key"]
    publication = cache_directory / runner.PUBLICATION_FILENAME
    publication.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        runner.Lfm2FunctionalOutputRefitRunnerError,
        match="publication authentication failed",
    ):
        runner.run(args, progress=lambda event: None)
