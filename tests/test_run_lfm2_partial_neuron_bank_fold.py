from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest


torch = pytest.importorskip("torch")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_lfm2_partial_neuron_bank_fold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_partial_neuron_bank_fold",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _paths(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "source-child"
    cache = tmp_path / "parent-cache"
    tokenizer = tmp_path / "tokenizer"
    for path in (source, cache, tokenizer):
        path.mkdir(parents=True)
    outputs = tmp_path / "parent-outputs.json"
    outputs.write_text("{}\n", encoding="ascii")
    return {
        "source_child_model": source,
        "parent_cache": cache,
        "cached_parent_outputs": outputs,
        "cached_parent_outputs_sha256": "a" * 64,
        "tokenizer_directory": tokenizer,
        "work_root": tmp_path / "work",
        "checkpoint_destination": tmp_path / "candidate-checkpoint",
        "report_path": tmp_path / "report.json",
    }


def test_default_is_exact_twenty_percent_frontier_not_a_toy(tmp_path) -> None:
    config = MODULE.PartialNeuronBankRunConfig(**_paths(tmp_path))
    target = config.target_accounting

    assert target["unit_pair_count"] == 7_477
    assert target["unit_pair_storage_parameters"] == 3_072
    assert target["additional_parameters_saved"] == 22_969_344
    assert target["actual_total_unique_parameters"] == 183_757_568
    assert target["total_parameters_removed_from_original_parent"] == 45_935_616
    assert target["total_compression_fraction_of_original_parent"] == pytest.approx(
        0.1999868485431418
    )
    assert config.layer_pairs == (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
    )
    assert config.max_shared_units_per_layer == 1_069
    serialized = config.as_dict()
    assert serialized["original_parent_model_loaded"] is False
    assert serialized["new_parent_forwards_allowed"] is False
    assert serialized["training_allowed"] is False
    assert serialized["standalone_checkpoint_available"] is False


def test_parameterized_fifteen_percent_fallback_rounds_to_real_units(
    tmp_path,
) -> None:
    config = MODULE.PartialNeuronBankRunConfig(
        **_paths(tmp_path),
        target_total_unique_parameters=195_239_206,
    )
    target = config.target_accounting

    assert target["unit_pair_count"] == 3_740
    assert target["additional_parameters_saved"] == 11_489_280
    assert target["actual_total_unique_parameters"] == 195_237_632
    assert target["budget_rounding_parameters"] == 1_574
    assert target["total_compression_fraction_of_original_parent"] >= 0.15


def test_tiny_target_and_non_unit_savings_fail_closed(tmp_path) -> None:
    with pytest.raises(
        MODULE.Lfm2PartialNeuronBankRunnerError,
        match="tiny experiment",
    ):
        MODULE.PartialNeuronBankRunConfig(
            **_paths(tmp_path / "tiny"),
            target_total_unique_parameters=205_000_000,
        )

    with pytest.raises(
        MODULE.Lfm2PartialNeuronBankRunnerError,
        match="exact multiple",
    ):
        MODULE.PartialNeuronBankRunConfig(
            **_paths(tmp_path / "unaligned"),
            target_total_unique_parameters=None,
            additional_parameters_saved=11_489_281,
        )


class _FakeCache:
    def __init__(
        self,
        *,
        identity: str,
        corpus: str,
        cache_key: str,
        batches: int = 20,
    ) -> None:
        self.parent_fingerprint = identity
        self.corpus_fingerprint = corpus
        self.cache_key = cache_key
        self.batch_count = batches


def test_source_child_cache_binding_is_exact_and_parent_never_executes() -> None:
    child = _FakeCache(
        identity="a" * 64,
        corpus="b" * 64,
        cache_key="c" * 64,
    )
    parent = _FakeCache(
        identity="d" * 64,
        corpus="b" * 64,
        cache_key="e" * 64,
    )

    audit = MODULE.authenticate_source_child_trace(
        source_child_identity_sha256="a" * 64,
        source_child_cache=child,
        parent_cache=parent,
    )

    assert audit["exact_source_child_bound"] is True
    assert audit["parent_model_loaded"] is False
    assert audit["new_parent_forward_calls"] == 0
    assert audit["source_child_cache_key"] == "c" * 64
    assert audit["parent_cache_key"] == "e" * 64


def test_source_trace_builder_runs_only_the_supplied_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_child = torch.nn.Linear(3, 2)
    parent_cache = object()
    child_cache = SimpleNamespace(cache_key="c" * 64)
    rows = SimpleNamespace(trace_rows_sha256="d" * 64)
    observed: dict[str, object] = {}

    def fake_batches(value):
        observed["token_source"] = value
        return ("cached-batch",)

    def fake_device(model):
        observed["device_model"] = model
        return torch.device("cpu")

    monkeypatch.setattr(
        MODULE.fold_runtime,
        "calibration_batches_from_cache",
        fake_batches,
    )
    monkeypatch.setattr(
        MODULE.fold_runtime,
        "_model_device",
        fake_device,
    )

    def fake_build(cache_root, model, batches, *, config, input_device):
        observed.update(
            {
                "cache_root": cache_root,
                "executed_model": model,
                "batches": batches,
                "capture_config": config,
                "input_device": input_device,
            }
        )
        return child_cache

    monkeypatch.setattr(
        MODULE,
        "build_lfm2_circuit_trace_cache",
        fake_build,
    )
    monkeypatch.setattr(
        MODULE.fold_runtime,
        "extract_trace_rows",
        lambda value: rows if value is child_cache else None,
    )
    monkeypatch.setattr(
        MODULE.shared_runner,
        "_expected_capture_forwards",
        lambda value: 360 if value is parent_cache else -1,
    )
    capture_config = MODULE.Lfm2CircuitTraceCaptureConfig()

    cache, observed_rows, audit = MODULE.build_source_child_trace_cache(
        parent_cache=parent_cache,
        source_child=source_child,
        cache_root=tmp_path / "source-trace",
        capture_config=capture_config,
    )

    assert cache is child_cache
    assert observed_rows is rows
    assert observed["token_source"] is parent_cache
    assert observed["executed_model"] is source_child
    assert observed["device_model"] is source_child
    assert audit["source_child_forward_calls"] == 360
    assert audit["parent_forward_calls"] == 0


@pytest.mark.parametrize("tamper", ("identity", "corpus", "batch_count"))
def test_source_child_cache_tamper_is_rejected(tamper: str) -> None:
    child = _FakeCache(
        identity="a" * 64,
        corpus="b" * 64,
        cache_key="c" * 64,
    )
    parent = _FakeCache(
        identity="d" * 64,
        corpus="b" * 64,
        cache_key="e" * 64,
    )
    identity = "a" * 64
    if tamper == "identity":
        identity = "f" * 64
    elif tamper == "corpus":
        child.corpus_fingerprint = "f" * 64
    else:
        child.batch_count = 19

    with pytest.raises(
        MODULE.Lfm2PartialNeuronBankRunnerError,
        match="exact live child|exact token batches|batch counts",
    ):
        MODULE.authenticate_source_child_trace(
            source_child_identity_sha256=identity,
            source_child_cache=child,
            parent_cache=parent,
        )


def test_candidate_acceptance_requires_parent_incumbent_and_whole_output() -> None:
    accepted = {"decision": {"accepted": True}}
    rejected = {"decision": {"accepted": False}}

    assert MODULE.candidate_accepted(accepted, accepted, accepted) is True
    assert MODULE.candidate_accepted(rejected, accepted, accepted) is False
    assert MODULE.candidate_accepted(accepted, rejected, accepted) is False
    assert MODULE.candidate_accepted(accepted, accepted, rejected) is False


class _RollbackInstallation:
    def __init__(self) -> None:
        self.active = True
        self.calls = 0

    def rollback(self) -> bool:
        self.calls += 1
        self.active = False
        return True


def test_rejected_candidate_rolls_back_with_independent_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installation = _RollbackInstallation()
    model = torch.nn.Linear(3, 2)
    counts = {
        "logical": 8,
        "unique": 8,
        "logical_parameter_objects": 2,
        "unique_parameter_objects": 2,
    }
    monkeypatch.setattr(
        MODULE.fold_runtime,
        "model_state_sha256",
        lambda _model: "a" * 64,
    )
    monkeypatch.setattr(
        MODULE.shared_runner,
        "parameter_counts",
        lambda _model: dict(counts),
    )
    monkeypatch.setattr(
        MODULE.shared_runner,
        "topology_signature",
        lambda _model: {"topology_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        MODULE.shared_runner,
        "gradient_audit",
        lambda _model: {"zero_gradient": True},
    )

    receipt = MODULE._authenticated_rollback(
        installation,
        model,
        source_state_sha256="a" * 64,
        source_counts=counts,
        source_topology_sha256="b" * 64,
    )

    assert installation.calls == 1
    assert receipt["exact"] is True
    assert receipt["restored_role"] == "incumbent_source_child"


def test_rollback_failure_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    installation = _RollbackInstallation()
    model = torch.nn.Linear(3, 2)
    monkeypatch.setattr(
        MODULE.fold_runtime,
        "model_state_sha256",
        lambda _model: "f" * 64,
    )
    monkeypatch.setattr(
        MODULE.shared_runner,
        "parameter_counts",
        lambda _model: {},
    )
    monkeypatch.setattr(
        MODULE.shared_runner,
        "topology_signature",
        lambda _model: {"topology_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        MODULE.shared_runner,
        "gradient_audit",
        lambda _model: {"zero_gradient": True},
    )

    with pytest.raises(
        MODULE.Lfm2PartialNeuronBankRunnerError,
        match="could not restore",
    ):
        MODULE._authenticated_rollback(
            installation,
            model,
            source_state_sha256="a" * 64,
            source_counts={"unique": 8},
            source_topology_sha256="b" * 64,
        )


def test_reports_are_canonical_sealed_and_never_overwritten(tmp_path) -> None:
    config = MODULE.PartialNeuronBankRunConfig(**_paths(tmp_path))
    report = MODULE._sealed_report(
        {
            "schema": MODULE.RUN_SCHEMA,
            "status": "rejected",
            "checkpoint_published": False,
        }
    )

    MODULE.shared_runner._atomic_write_new_json(config.report_path, report)

    loaded = json.loads(config.report_path.read_text(encoding="ascii"))
    body = dict(loaded)
    digest = body.pop("run_sha256")
    assert digest == MODULE._json_sha256(body)
    with pytest.raises(Exception, match="replace existing"):
        MODULE.shared_runner._atomic_write_new_json(
            config.report_path,
            report,
        )


def test_parser_exposes_large_total_target_and_no_parent_model_argument() -> None:
    parser = MODULE.build_parser()
    actions = {action.dest: action for action in parser._actions}

    assert {
        "source_child_model",
        "parent_cache",
        "target_total_unique_parameters",
        "additional_parameters_saved",
        "checkpoint_destination",
        "report_path",
        "minimum_parent_residual_cosine",
        "minimum_incumbent_residual_cosine",
        "whole_output_minimum_task_score_retention",
    }.issubset(actions)
    assert "parent_model" not in actions
    assert actions["checkpoint_destination"].required is True


def test_runner_has_no_training_network_parent_loader_or_checkpoint_fraud() -> None:
    source = inspect.getsource(MODULE)

    assert "AutoModel" not in source
    assert "AutoTokenizer" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "save_pretrained" not in source
    assert "pickle" not in source
    assert "parent_model:" not in source
    assert "except BaseException" in source
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in source
    assert MODULE.PUBLICATION_BLOCK_STATUS in source
