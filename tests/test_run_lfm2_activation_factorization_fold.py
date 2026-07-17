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
    / "run_lfm2_activation_factorization_fold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_activation_factorization_fold",
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


def _paths_for_batches(
    *,
    batches: int = 4,
    prefill_rows: int = 4,
    decode_steps: int = 8,
) -> tuple[str, ...]:
    result = []
    for batch in range(batches):
        result.extend(
            f"batch.{batch:06d}/prefill/row.{row:04d}"
            for row in range(prefill_rows)
        )
        result.extend(
            f"batch.{batch:06d}/decode.{step:03d}/row.0000"
            for step in range(decode_steps)
        )
    return tuple(result)


def test_default_is_real_twenty_percent_frontier_with_fifteen_fallback(
    tmp_path: Path,
) -> None:
    config = MODULE.ActivationFactorizationRunConfig(**_paths(tmp_path))
    primary = MODULE.resolve_target_accounting(
        original_parent_unique_parameters=229_693_184,
        source_child_unique_parameters=206_726_912,
        target_total_unique_parameters=config.target_total_unique_parameters,
    )
    fallback = MODULE.resolve_target_accounting(
        original_parent_unique_parameters=229_693_184,
        source_child_unique_parameters=206_726_912,
        target_total_unique_parameters=(
            config.fallback_target_total_unique_parameters
        ),
    )

    assert config.target_total_unique_parameters == 183_760_640
    assert config.fallback_target_total_unique_parameters == 195_239_206
    assert config.roles == ("w2",)
    assert primary[
        "requested_total_compression_fraction_of_original_parent"
    ] == pytest.approx(0.19997347678317714)
    assert fallback[
        "requested_total_compression_fraction_of_original_parent"
    ] >= 0.149
    assert config.target_schedule(206_726_912) == (
        183_760_640,
        195_239_206,
    )
    serialized = config.as_dict()
    assert serialized["whole_output_parent_cache"] == str(
        config.parent_cache
    )
    assert serialized["original_parent_model_loaded"] is False
    assert serialized["new_parent_forwards_allowed"] is False
    assert serialized["new_parent_response_generation_allowed"] is False
    assert serialized["candidate_generation_required_for_coherence_gate"] is True
    assert serialized["training_allowed"] is False
    assert serialized["standalone_checkpoint_available"] is False


def test_explicit_target_and_disabled_fallback_are_supported(
    tmp_path: Path,
) -> None:
    config = MODULE.ActivationFactorizationRunConfig(
        **_paths(tmp_path),
        target_total_unique_parameters=195_239_206,
        fallback_target_total_unique_parameters=None,
        expected_source_child_unique_parameters=206_726_912,
    )

    assert config.target_schedule(206_726_912) == (195_239_206,)


def test_arbitrary_smaller_child_skips_an_inapplicable_default_fallback(
    tmp_path: Path,
) -> None:
    config = MODULE.ActivationFactorizationRunConfig(
        **_paths(tmp_path),
        target_total_unique_parameters=170_000_000,
        expected_source_child_unique_parameters=180_000_000,
    )

    assert config.target_schedule(180_000_000) == (170_000_000,)


def test_roles_default_to_residual_write_but_variants_are_exposed(
    tmp_path: Path,
) -> None:
    config = MODULE.ActivationFactorizationRunConfig(
        **_paths(tmp_path),
        roles=("w2", "w1"),
    )

    assert config.roles == ("w2", "w1")
    assert config.as_dict()["factorized_roles"] == ["w2", "w1"]
    with pytest.raises(
        MODULE.Lfm2ActivationFactorizationRunnerError,
        match="drawn from",
    ):
        MODULE.ActivationFactorizationRunConfig(
            **_paths(tmp_path / "bad"),
            roles=("w2", "attention"),
        )


def test_tiny_or_noncompressive_targets_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(
        MODULE.Lfm2ActivationFactorizationRunnerError,
        match="tiny experiment",
    ):
        MODULE.ActivationFactorizationRunConfig(
            **_paths(tmp_path / "tiny"),
            target_total_unique_parameters=200_000_000,
            fallback_target_total_unique_parameters=None,
            expected_source_child_unique_parameters=206_726_912,
        )

    with pytest.raises(
        MODULE.Lfm2ActivationFactorizationRunnerError,
        match="below the incumbent",
    ):
        MODULE.resolve_target_accounting(
            original_parent_unique_parameters=229_693_184,
            source_child_unique_parameters=190_000_000,
            target_total_unique_parameters=195_239_206,
        )


def test_path_split_is_exact_disjoint_deterministic_and_horizon_covered() -> None:
    paths = _paths_for_batches()
    domains = tuple(
        "math" if path.startswith(("batch.000000", "batch.000001")) else "code"
        for path in paths
    )

    first = MODULE.deterministic_factorization_split(
        paths,
        path_domains=domains,
        max_fit_rows=24,
        max_validation_rows=16,
        validation_fraction=0.25,
    )
    second = MODULE.deterministic_factorization_split(
        paths,
        path_domains=domains,
        max_fit_rows=24,
        max_validation_rows=16,
        validation_fraction=0.25,
    )

    assert first.as_dict() == second.as_dict()
    assert set(first.fit_indices).isdisjoint(first.validation_indices)
    assert first.fit_row_ids == tuple(paths[index] for index in first.fit_indices)
    assert first.validation_row_ids == tuple(
        paths[index] for index in first.validation_indices
    )
    assert first.long_position_counts["fit"] > 0
    assert first.long_position_counts["validation"] > 0
    assert all(
        first.domain_counts[domain]["fit"] > 0
        and first.domain_counts[domain]["validation"] > 0
        for domain in ("math", "code")
    )
    body = first.as_dict()
    digest = body.pop("split_sha256")
    assert digest == MODULE._json_sha256(body)


def test_default_split_uses_exact_batch_ids_without_claiming_semantics() -> None:
    paths = _paths_for_batches(batches=3)
    split = MODULE.deterministic_factorization_split(paths)

    assert split.domain_source == "authenticated_batch_case_surrogate"
    assert set(split.domain_counts) == {
        "batch.000000",
        "batch.000001",
        "batch.000002",
    }
    assert set(split.fit_row_ids).isdisjoint(split.validation_row_ids)


def test_invalid_or_duplicate_path_ids_are_rejected() -> None:
    with pytest.raises(
        MODULE.Lfm2ActivationFactorizationRunnerError,
        match="unique",
    ):
        MODULE.deterministic_factorization_split(
            (
                "batch.000000/prefill/row.0000",
                "batch.000000/prefill/row.0000",
            )
        )
    with pytest.raises(
        MODULE.Lfm2ActivationFactorizationRunnerError,
        match="Unsupported exact",
    ):
        MODULE.deterministic_factorization_split(("bad", "also-bad"))


class _FakeCache:
    def __init__(
        self,
        *,
        identity: str,
        corpus: str,
        cache_key: str,
        capture: dict[str, object] | None = None,
    ) -> None:
        self.parent_fingerprint = identity
        self.corpus_fingerprint = corpus
        self.cache_key = cache_key
        self.batch_count = 2
        self.manifest = {"capture": capture or {"sample_last_tokens": 8}}


def test_cache_binding_requires_exact_model_corpus_protocol_and_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths_for_batches(batches=2)
    source = _FakeCache(
        identity="a" * 64,
        corpus="b" * 64,
        cache_key="c" * 64,
    )
    parent = _FakeCache(
        identity="d" * 64,
        corpus="b" * 64,
        cache_key="e" * 64,
    )
    rows = MODULE.FactorizationTraceRows(
        paths=paths,
        layer_inputs={},
        post_swiglu={},
        cache_key="c" * 64,
        corpus_fingerprint="b" * 64,
        trace_sha256="f" * 64,
    )
    monkeypatch.setattr(
        MODULE.bank_runner,
        "authenticate_source_child_trace",
        lambda **_kwargs: {
            "exact_source_child_bound": True,
            "parent_model_loaded": False,
            "new_parent_forward_calls": 0,
        },
    )
    monkeypatch.setattr(
        MODULE.fold_runtime,
        "extract_trace_rows",
        lambda cache: SimpleNamespace(paths=paths),
    )

    audit = MODULE.authenticate_factorization_cache(
        source_child_identity_sha256="a" * 64,
        source_child_cache=source,
        parent_cache=parent,
        factor_rows=rows,
    )

    assert audit["exact_source_child_bound"] is True
    assert audit["capture_protocol_exact"] is True
    assert audit["trajectory_paths_exact"] is True
    assert audit["factorization_fields_exact"] is True
    assert audit["original_parent_model_loaded"] is False
    assert audit["new_parent_forward_calls"] == 0
    assert audit["new_parent_response_generation_calls"] == 0

    source.manifest["capture"] = {"sample_last_tokens": 7}
    with pytest.raises(
        MODULE.Lfm2ActivationFactorizationRunnerError,
        match="different capture protocols",
    ):
        MODULE.authenticate_factorization_cache(
            source_child_identity_sha256="a" * 64,
            source_child_cache=source,
            parent_cache=parent,
            factor_rows=rows,
        )


def test_cache_binding_rejects_factor_rows_from_another_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths_for_batches(batches=2)
    source = _FakeCache(
        identity="a" * 64,
        corpus="b" * 64,
        cache_key="c" * 64,
    )
    parent = _FakeCache(
        identity="d" * 64,
        corpus="b" * 64,
        cache_key="e" * 64,
    )
    rows = MODULE.FactorizationTraceRows(
        paths=paths,
        layer_inputs={},
        post_swiglu={},
        cache_key="f" * 64,
        corpus_fingerprint="b" * 64,
        trace_sha256="1" * 64,
    )
    monkeypatch.setattr(
        MODULE.bank_runner,
        "authenticate_source_child_trace",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        MODULE.fold_runtime,
        "extract_trace_rows",
        lambda cache: SimpleNamespace(paths=paths),
    )

    with pytest.raises(
        MODULE.Lfm2ActivationFactorizationRunnerError,
        match="exact source-child cache",
    ):
        MODULE.authenticate_factorization_cache(
            source_child_identity_sha256="a" * 64,
            source_child_cache=source,
            parent_cache=parent,
            factor_rows=rows,
        )


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


def test_rollback_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        MODULE.Lfm2ActivationFactorizationRunnerError,
        match="did not restore",
    ):
        MODULE._authenticated_rollback(
            installation,
            model,
            source_state_sha256="a" * 64,
            source_counts={"unique": 8},
            source_topology_sha256="b" * 64,
        )


def test_candidate_acceptance_requires_all_behavior_gates() -> None:
    accepted = {"decision": {"accepted": True}}
    rejected = {"decision": {"accepted": False}}

    assert MODULE._candidate_accepted(accepted, accepted, accepted) is True
    assert MODULE._candidate_accepted(rejected, accepted, accepted) is False
    assert MODULE._candidate_accepted(accepted, rejected, accepted) is False
    assert MODULE._candidate_accepted(accepted, accepted, rejected) is False


def test_reports_are_canonical_sealed_and_never_overwritten(
    tmp_path: Path,
) -> None:
    config = MODULE.ActivationFactorizationRunConfig(**_paths(tmp_path))
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


def test_parser_exposes_real_targets_and_no_parent_model_argument() -> None:
    parser = MODULE.build_parser()
    actions = {action.dest: action for action in parser._actions}

    assert {
        "source_child_model",
        "source_child_cache",
        "parent_cache",
        "whole_output_parent_cache",
        "target_total_unique_parameters",
        "fallback_target_total_unique_parameters",
        "roles",
        "checkpoint_destination",
        "report_path",
        "minimum_parent_residual_cosine",
        "minimum_incumbent_residual_cosine",
        "whole_output_minimum_task_score_retention",
    }.issubset(actions)
    assert "parent_model" not in actions
    assert (
        actions["target_total_unique_parameters"].default
        == MODULE.DEFAULT_TARGET_TOTAL_UNIQUE_PARAMETERS
    )
    assert actions["roles"].default == ("w2",)
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
