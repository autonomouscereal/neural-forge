from __future__ import annotations

import argparse
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
    / "run_lfm2_parameter_sharing_fold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_parameter_sharing_fold",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _paths(tmp_path: Path) -> dict[str, object]:
    parent = tmp_path / "parent"
    cache = tmp_path / "cache"
    tokenizer = tmp_path / "tokenizer"
    for path in (parent, cache, tokenizer):
        path.mkdir(parents=True)
    outputs = tmp_path / "parent-outputs.json"
    outputs.write_text("{}\n", encoding="ascii")
    return {
        "parent_model": parent,
        "parent_cache": cache,
        "cached_parent_outputs": outputs,
        "cached_parent_outputs_sha256": "a" * 64,
        "tokenizer_directory": tokenizer,
        "work_root": tmp_path / "work",
        "checkpoint_destination": tmp_path / "child",
        "report_path": tmp_path / "report.json",
    }


def test_default_contract_is_three_pairs_and_real_ten_percent_savings(
    tmp_path,
) -> None:
    config = MODULE.ParameterSharingRunConfig(**_paths(tmp_path))

    assert config.pair_count == 3
    assert config.expected_pair_savings == 23_592_960
    assert config.expected_child_unique_parameters == 206_100_224
    assert (
        config.expected_pair_savings / config.expected_parent_unique_parameters
        == pytest.approx(0.10271510712307423)
    )
    assert config.whole_output_minimum_task_score_retention == 0.90
    assert config.whole_output_maximum_parent_task_score_regressions == 2
    assert config.as_dict()["training_allowed"] is False


def test_strict_pair_and_weight_parsers() -> None:
    assert MODULE.parse_layer_pairs("4:1, 8:6") == ((1, 4), (6, 8))
    assert MODULE.parse_weight_pair("2, 3") == (2.0, 3.0)

    with pytest.raises(argparse.ArgumentTypeError, match="disjoint"):
        MODULE.parse_layer_pairs("0:2,2:4")
    with pytest.raises(argparse.ArgumentTypeError, match="distinct"):
        MODULE.parse_layer_pairs("3:3")
    with pytest.raises(argparse.ArgumentTypeError, match="positive total"):
        MODULE.parse_weight_pair("0,0")
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        MODULE.parse_weight_pair("nan,1")


def test_config_rejects_remote_parent_and_source_tree_outputs(tmp_path) -> None:
    values = _paths(tmp_path)
    values["parent_model"] = "https://example.invalid/model"
    with pytest.raises(
        MODULE.Lfm2ParameterSharingRunnerError,
        match="explicit local",
    ):
        MODULE.ParameterSharingRunConfig(**values)

    values = _paths(tmp_path / "second")
    parent = values["parent_model"]
    assert isinstance(parent, Path)
    values["work_root"] = parent / "work"
    with pytest.raises(
        MODULE.Lfm2ParameterSharingRunnerError,
        match="must not be inside",
    ):
        MODULE.ParameterSharingRunConfig(**values)


def test_parser_exposes_strict_large_compaction_and_gate_options() -> None:
    parser = MODULE.build_parser()
    options = {action.dest for action in parser._actions}

    assert {
        "parent_model",
        "parent_cache",
        "cached_parent_outputs",
        "cached_parent_outputs_sha256",
        "pair_count",
        "layer_pairs",
        "minimum_residual_cosine",
        "whole_output_minimum_task_score_retention",
        "whole_output_maximum_parent_task_score_regressions",
        "checkpoint_destination",
        "report_path",
    }.issubset(options)
    pair_action = next(
        action for action in parser._actions if action.dest == "pair_count"
    )
    assert pair_action.default == 3


class _Rows:
    layer_kinds = ("short_conv", "full_attention", "short_conv")

    def __init__(self) -> None:
        self.values = {
            index: torch.arange(60, dtype=torch.float32).reshape(10, 6) + index
            for index in range(3)
        }

    def field(self, layer_index, name):
        assert name == "mlp.post_swiglu"
        return self.values[layer_index]


def test_cached_post_swiglu_selection_is_bounded_and_path_aligned() -> None:
    selected, indices = MODULE.select_post_swiglu_rows(
        _Rows(),
        maximum_rows=4,
    )

    assert indices == (0, 2, 5, 7)
    assert set(selected) == {0, 1, 2}
    for layer_index, rows in selected.items():
        assert tuple(rows.shape) == (4, 6)
        torch.testing.assert_close(
            rows,
            _Rows().values[layer_index][list(indices)],
        )


def test_parent_forward_prohibition_fails_closed_then_restores() -> None:
    model = torch.nn.Linear(3, 2)
    inputs = torch.ones(1, 3)
    expected = model(inputs)

    with MODULE.ParentForwardProhibition(model) as guard:
        with pytest.raises(
            MODULE.Lfm2ParameterSharingRunnerError,
            match="source-model forward",
        ):
            model(inputs)
        assert guard.attempted_forward_calls == 1

    torch.testing.assert_close(model(inputs), expected)


class _Ffn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(4, 6, bias=False)
        self.w3 = torch.nn.Linear(4, 6, bias=False)
        self.w2 = torch.nn.Linear(6, 4, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feed_forward = _Ffn()


class _AliasModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(_Layer() for _ in range(4))


def _tie_pair(model: _AliasModel, left: int, right: int) -> None:
    source = model.model.layers[left].feed_forward
    target = model.model.layers[right].feed_forward
    for role in ("w1", "w3", "w2"):
        getattr(target, role).weight = getattr(source, role).weight


def test_alias_audit_and_parameter_counts_require_complete_disjoint_pairs() -> None:
    model = _AliasModel()
    dense = MODULE.parameter_counts(model)
    _tie_pair(model, 0, 2)

    audit = MODULE.audit_shared_aliases(model, ((0, 2),))
    shared = MODULE.parameter_counts(model)

    assert audit["exact"] is True
    assert audit["actual_shared_parameter_groups"] == 3
    assert dense["logical"] == shared["logical"]
    assert dense["unique"] - shared["unique"] == 3 * 4 * 6
    with pytest.raises(
        MODULE.Lfm2ParameterSharingRunnerError,
        match="aliases differ",
    ):
        MODULE.audit_shared_aliases(model, ((0, 3),))


def test_capture_protocol_is_reconstructed_exactly() -> None:
    config = MODULE.Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=8,
        max_decode_steps=8,
        max_capture_bytes=128 * 1024 * 1024,
        logit_rtol=0.2,
        logit_atol=1.0,
        require_top1_agreement=False,
    )
    cache = SimpleNamespace(manifest={"capture": config.as_dict()})

    observed = MODULE.capture_config_from_parent_cache(cache)

    assert observed == config
    changed = dict(config.as_dict())
    changed["training_steps"] = 1
    with pytest.raises(
        MODULE.Lfm2ParameterSharingRunnerError,
        match="cannot be reconstructed",
    ):
        MODULE.capture_config_from_parent_cache(
            SimpleNamespace(manifest={"capture": changed})
        )


def _trajectory(score: float = 0.9) -> dict[str, object]:
    report = {
        "passed": True,
        "reasons": [],
        "score": score,
        "teacher_forced": {"top1_agreement": 0.8},
        "free_run": {"token_match_fraction": 0.7},
    }
    return {
        "decision": {"accepted": True, "reasons": []},
        "cache_key": "a" * 64,
        "report": report,
    }


def _behavior(output: str = "coherent") -> dict[str, object]:
    event = {
        "decision": {"accepted": True, "reason": "ok", "metrics": {}},
        "per_case_task_regression_ids": [],
        "outputs": [{"case_id": "one", "output": output}],
    }
    return {
        "decision": {"accepted": True},
        "event": event,
    }


def test_acceptance_and_reload_equivalence_use_whole_outputs_not_exact_parent_text() -> None:
    trajectory = _trajectory()
    behavior = _behavior("a valid child phrasing")

    assert MODULE.candidate_accepted(trajectory, behavior) is True
    trajectory_comparison = MODULE.trajectory_equivalence(
        trajectory,
        _trajectory(),
    )
    assert trajectory_comparison["exact_metrics_diagnostic"] is True
    assert trajectory_comparison["gate_decision_equivalent"] is True
    assert MODULE.behavior_equivalence(
        behavior,
        _behavior("a valid child phrasing"),
    )["exact_output_and_metrics_diagnostic"] is True
    different_text = MODULE.behavior_equivalence(
        behavior,
        _behavior("different after reload"),
    )
    assert different_text["exact_output_and_metrics_diagnostic"] is False
    assert different_text["gate_decision_equivalent"] is True
    assert different_text["exact_output_required"] is False

    rejected = _behavior()
    rejected["decision"] = {"accepted": False}
    assert MODULE.candidate_accepted(trajectory, rejected) is False


def test_reports_are_canonical_sealed_and_never_overwritten(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    sealed = MODULE._sealed_report({"schema": "test", "status": "rejected"})

    MODULE._atomic_write_new_json(path, sealed)

    loaded = json.loads(path.read_text(encoding="ascii"))
    body = dict(loaded)
    digest = body.pop("run_sha256")
    assert digest == MODULE._json_sha256(body)
    with pytest.raises(
        MODULE.Lfm2ParameterSharingRunnerError,
        match="replace existing",
    ):
        MODULE._atomic_write_new_json(path, sealed)


def test_runner_source_has_no_hub_loader_or_training_execution() -> None:
    source = inspect.getsource(MODULE)

    assert "AutoModel" not in source
    assert "from_pretrained(" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "build_lfm2_circuit_trace_cache(" not in source
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in source
