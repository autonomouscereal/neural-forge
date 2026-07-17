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

from neural_forge.hf_lfm2_residual_parameter_sharing import (  # noqa: E402
    Lfm2ResidualFfnPairPlan,
    Lfm2ResidualParameterSharingPlan,
    Lfm2ResidualRolePlan,
)

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_lfm2_residual_parameter_sharing_fold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_residual_parameter_sharing_fold",
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


def test_default_is_real_four_pair_rank64_compaction(tmp_path) -> None:
    config = MODULE.ResidualParameterSharingRunConfig(**_paths(tmp_path))

    assert config.pair_count == 4
    assert config.rank == 64
    assert config.orientations == ("later",) * 4
    assert config.expected_pair_savings == 28_704_768
    assert config.expected_child_unique_parameters == 200_988_416
    assert config.expected_compression_fraction == pytest.approx(
        0.12497004699974032
    )
    assert config.whole_output_minimum_task_score_retention == 0.90
    assert config.whole_output_maximum_parent_task_score_regressions == 2
    serialized = config.as_dict()
    assert serialized["training_allowed"] is False
    assert serialized["new_parent_forwards_allowed"] is False
    assert serialized["standalone_checkpoint_available"] is False
    assert (
        serialized["whole_output_gate"]["exact_parent_text_required"]
        is False
    )
    assert (
        serialized["whole_output_gate"]["exact_token_match_required"]
        is False
    )


def test_seven_pair_rank96_target_is_supported_exactly(tmp_path) -> None:
    config = MODULE.ResidualParameterSharingRunConfig(
        **_paths(tmp_path),
        pair_count=7,
        rank=96,
        preserve="earlier",
    )

    assert config.orientations == ("earlier",) * 7
    assert config.expected_pair_savings == 47_824_896
    assert config.expected_child_unique_parameters == 181_868_288
    assert config.expected_compression_fraction == pytest.approx(
        0.20821208173073172
    )


def test_preserve_pair_and_weight_parsers_are_strict() -> None:
    assert MODULE.parse_preserve("later") == "later"
    assert MODULE.parse_preserve("later, earlier") == (
        "later",
        "earlier",
    )
    assert MODULE.parse_layer_pairs("4:1, 8:6") == ((1, 4), (6, 8))
    assert MODULE.parse_weight_pair("2, 3") == (2.0, 3.0)

    with pytest.raises(argparse.ArgumentTypeError, match="preserve"):
        MODULE.parse_preserve("newest")
    with pytest.raises(argparse.ArgumentTypeError, match="disjoint"):
        MODULE.parse_layer_pairs("0:2,2:4")
    with pytest.raises(argparse.ArgumentTypeError, match="distinct"):
        MODULE.parse_layer_pairs("3:3")
    with pytest.raises(argparse.ArgumentTypeError, match="positive total"):
        MODULE.parse_weight_pair("0,0")
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        MODULE.parse_weight_pair("nan,1")


def test_config_validates_orientation_count_and_positive_compression(
    tmp_path,
) -> None:
    with pytest.raises(
        MODULE.Lfm2ResidualParameterSharingRunnerError,
        match="one orientation per pair",
    ):
        MODULE.ResidualParameterSharingRunConfig(
            **_paths(tmp_path / "orientations"),
            pair_count=4,
            preserve=("later", "earlier"),
        )

    with pytest.raises(
        MODULE.Lfm2ResidualParameterSharingRunnerError,
        match="positive physical compression",
    ):
        MODULE.ResidualParameterSharingRunConfig(
            **_paths(tmp_path / "rank"),
            rank=1024,
        )


def test_config_rejects_remote_source_and_any_existing_checkpoint(
    tmp_path,
) -> None:
    values = _paths(tmp_path / "remote")
    values["parent_model"] = "https://example.invalid/model"
    with pytest.raises(
        MODULE.Lfm2ResidualParameterSharingRunnerError,
        match="explicit local",
    ):
        MODULE.ResidualParameterSharingRunConfig(**values)

    values = _paths(tmp_path / "checkpoint")
    checkpoint = values["checkpoint_destination"]
    assert isinstance(checkpoint, Path)
    checkpoint.mkdir()
    with pytest.raises(
        MODULE.Lfm2ResidualParameterSharingRunnerError,
        match="already exists",
    ):
        MODULE.ResidualParameterSharingRunConfig(**values)


def test_parser_exposes_large_ranked_targets_and_reserved_destination() -> None:
    parser = MODULE.build_parser()
    actions = {action.dest: action for action in parser._actions}

    assert {
        "parent_model",
        "parent_cache",
        "cached_parent_outputs",
        "cached_parent_outputs_sha256",
        "pair_count",
        "rank",
        "preserve",
        "layer_pairs",
        "minimum_residual_cosine",
        "whole_output_minimum_task_score_retention",
        "whole_output_maximum_parent_task_score_regressions",
        "checkpoint_destination",
        "report_path",
    }.issubset(actions)
    assert actions["pair_count"].default == 4
    assert actions["rank"].default == 64
    assert actions["preserve"].default == "later"
    assert actions["checkpoint_destination"].required is True


class _Rows:
    layer_kinds = ("short_conv", "full_attention", "short_conv")

    def __init__(self) -> None:
        self.values = {
            index: (
                torch.arange(60, dtype=torch.float32).reshape(10, 6)
                + index
            )
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
            MODULE.Lfm2ResidualParameterSharingRunnerError,
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


class _ResidualModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList((_Layer(), _Layer()))


def _role_plan(role: str, weight: torch.Tensor, rank: int):
    left = torch.zeros(weight.shape[0], rank)
    right = torch.zeros(rank, weight.shape[1])
    return Lfm2ResidualRolePlan(
        role=role,
        matrix_shape=tuple(weight.shape),
        rank=rank,
        residual_left=left,
        residual_right=right,
        residual_left_sha256="a" * 64,
        residual_right_sha256="b" * 64,
        residual_frobenius_energy=0.0,
        retained_singular_energy=0.0,
        truncation_error_energy=0.0,
        relative_frobenius_error=0.0,
    )


def test_residual_installation_audit_checks_base_identity_and_factor_shapes() -> None:
    model = _ResidualModel()
    rank = 2
    roles = []
    for role in ("w1", "w3", "w2"):
        preserved = getattr(model.model.layers[1].feed_forward, role)
        role_plan = _role_plan(role, preserved.weight, rank)
        roles.append(role_plan)
        setattr(
            model.model.layers[0].feed_forward,
            role,
            MODULE.Lfm2SharedBaseResidualLinear(
                preserved.weight,
                role_plan.residual_left,
                role_plan.residual_right,
            ),
        )
    pair = Lfm2ResidualFfnPairPlan(
        sharing_pair_plan=SimpleNamespace(layers=(0, 1)),
        preserve="later",
        preserved_layer=1,
        approximated_layer=0,
        rank=rank,
        role_plans=tuple(roles),
        parameters_saved=MODULE.lfm2_residual_parameter_sharing_savings(
            4,
            6,
            1,
            rank,
        ),
    )
    sharing = SimpleNamespace(
        selected_pairs=((0, 1),),
        parent_identity_sha256="a" * 64,
        evidence_sha256="b" * 64,
    )
    plan = Lfm2ResidualParameterSharingPlan(
        schema="test",
        sharing_plan=sharing,
        pair_plans=(pair,),
        rank=rank,
        orientations=("later",),
        logical_parameters_before=144,
        unique_parameters_before=144,
        projected_logical_parameters_after=204,
        projected_unique_parameters_after=84,
        projected_parameters_saved=60,
        plan_sha256="c" * 64,
    )

    audit = MODULE.audit_residual_installation(model, plan)

    assert audit["exact"] is True
    assert audit["actual_residual_modules"] == 3
    assert audit["shared_base_aliases"] == 3
    assert all(
        row["shared_base_exact_parameter_identity"]
        for row in audit["roles"]
    )


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

    assert MODULE.capture_config_from_parent_cache(cache) == config
    changed = dict(config.as_dict())
    changed["training_steps"] = 1
    with pytest.raises(
        MODULE.Lfm2ResidualParameterSharingRunnerError,
        match="cannot be reconstructed",
    ):
        MODULE.capture_config_from_parent_cache(
            SimpleNamespace(manifest={"capture": changed})
        )


def test_candidate_acceptance_uses_gates_not_exact_text_or_tokens() -> None:
    trajectory = {
        "decision": {"accepted": True},
        "report": {
            "teacher_forced": {"top1_agreement": 0.4},
            "free_run": {"token_match_fraction": 0.2},
        },
    }
    behavior = {
        "decision": {"accepted": True},
        "event": {
            "outputs": [
                {
                    "case_id": "one",
                    "output": "Different but coherent phrasing.",
                }
            ]
        },
    }

    assert MODULE.candidate_accepted(trajectory, behavior) is True
    behavior["decision"] = {"accepted": False}
    assert MODULE.candidate_accepted(trajectory, behavior) is False


def test_checkpoint_absence_is_an_authenticated_publication_invariant(
    tmp_path,
) -> None:
    config = MODULE.ResidualParameterSharingRunConfig(**_paths(tmp_path))

    audit = MODULE._checkpoint_absence_audit(config)

    assert audit["exists"] is False
    assert audit["files_written"] == 0
    assert audit["dense_fallback_used"] is False
    assert audit["ordinary_full_sharing_checkpoint_used"] is False
    config.checkpoint_destination.mkdir()
    with pytest.raises(
        MODULE.Lfm2ResidualParameterSharingRunnerError,
        match="checkpoint appeared",
    ):
        MODULE._checkpoint_absence_audit(config)


def test_reports_are_canonical_sealed_and_never_overwritten(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    sealed = MODULE._sealed_report(
        {
            "schema": MODULE.RUN_SCHEMA,
            "status": MODULE.PUBLICATION_BLOCK_STATUS,
            "checkpoint_published": False,
        }
    )

    MODULE._atomic_write_new_json(path, sealed)

    loaded = json.loads(path.read_text(encoding="ascii"))
    body = dict(loaded)
    digest = body.pop("run_sha256")
    assert digest == MODULE._json_sha256(body)
    with pytest.raises(
        MODULE.Lfm2ResidualParameterSharingRunnerError,
        match="replace existing",
    ):
        MODULE._atomic_write_new_json(path, sealed)


def test_runner_has_no_checkpoint_fraud_hub_loader_or_training_execution() -> None:
    source = inspect.getsource(MODULE)

    assert "hf_lfm2_shared_checkpoint" not in source
    assert "export_lfm2_shared" not in source
    assert "load_lfm2_shared" not in source
    assert "pickle" not in source
    assert "AutoModel" not in source
    assert "from_pretrained(" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "build_lfm2_circuit_trace_cache(" not in source
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in source
    assert MODULE.PUBLICATION_BLOCK_STATUS in source
