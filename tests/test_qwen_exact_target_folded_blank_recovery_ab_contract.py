from __future__ import annotations

import importlib
import json
import sys
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from neural_forge import hf_experimental_child_artifact as child_artifact
from neural_forge.hf_compiled_child_cache import model_state_fingerprint
from neural_forge.hf_progressive_target_bridge import _dense_model_state_sha256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

runner = importlib.import_module("run_qwen_exact_target_folded_blank_recovery_ab")
builder = importlib.import_module("prepare_qwen_exact_target_folded_blank_recovery_ab")

_TINY_STATE_KEY_SHA256 = runner.canonical_sha256(["scale", "weight"])
_TINY_CONFIG_PAYLOAD_SHA256 = runner.canonical_sha256({"width": 4})


def _digest(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return runner.stage._file_sha256(path)


def _write_canonical_json(path: Path, value: Any) -> str:
    path.write_bytes(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    return runner.stage._file_sha256(path)


def _with_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
    payload = dict(value)
    payload[field] = runner.canonical_sha256(payload)
    return payload


class _TinyFoldedModel(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.width = width
        self.weight = nn.Parameter(
            torch.arange(width * 3, dtype=torch.float32).reshape(3, width) / 13.0
        )
        self.register_buffer("scale", torch.linspace(0.25, 1.0, width))

    def save_pretrained(self, destination: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        save_file(self.state_dict(), destination / "model.safetensors")
        (destination / "config.json").write_text(
            json.dumps({"width": self.width}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, source: Path) -> _TinyFoldedModel:
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        model = cls(width=int(config["width"]))
        model.load_state_dict(
            load_file(source / "model.safetensors", device="cpu"),
            strict=True,
        )
        return model


class _TinyFoldedTokenizer:
    def save_pretrained(self, destination: Path) -> None:
        (destination / "tokenizer_config.json").write_text(
            json.dumps({"model_max_length": 128}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "tokenizer.json").write_text(
            json.dumps(
                {"model": {"type": "WordLevel", "vocab": {"<unk>": 0}}},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _contract_model_state_fingerprint(model: nn.Module) -> dict[str, Any]:
    result = dict(model_state_fingerprint(model))
    result.update(
        {
            "unique_parameter_count": runner.EXACT_UNIQUE_PARAMETERS,
            "logical_parameter_count": runner.EXACT_UNIQUE_PARAMETERS,
            "parameter_topology_sha256": _digest("parameter-topology"),
            "buffer_topology_sha256": _digest("buffer-topology"),
        }
    )
    return result


def _folded_native_certificate(_model: nn.Module) -> dict[str, Any]:
    return {
        "schema": "test-native-dense-certificate-v1",
        "exact_unique_parameters": runner.EXACT_UNIQUE_PARAMETERS,
        "logical_parameters": runner.EXACT_UNIQUE_PARAMETERS,
        "structure_sha256": _digest("folded-native-structure"),
        "state_key_sha256": _digest("folded-native-registered-keys"),
        "state_key_count": 2,
    }


def _folded_runtime_audit(_model: nn.Module, _path: Path) -> dict[str, int]:
    return {
        "adapter_or_registry_module_count": 0,
        "optimizer_parameter_count": 0,
        "source_sized_sidecar_parameter_count": 0,
        "parent_runtime_reference_count": 0,
    }


def _folded_artifact_hooks(
    *,
    model_loader: Any = None,
) -> Any:
    return runner.FoldedArtifactVerificationHooks(
        model_loader=(
            _TinyFoldedModel.from_pretrained if model_loader is None else model_loader
        ),
        native_certifier=_folded_native_certificate,
        runtime_auditor=_folded_runtime_audit,
    )


@contextmanager
def _folded_materialization_context(report_bytes: bytes) -> Iterator[bytes]:
    yield report_bytes


def _replay_payload() -> dict[str, Any]:
    return _with_hash(
        {
            "schema": runner.REPLAY_SCHEMA,
            "status": "frozen",
            "calibration_texts": ["alpha prompt", "beta prompt"],
            "precomputed_teacher_generated_texts": [
                "alpha prompt teacher answer",
                "beta prompt teacher answer",
            ],
            "on_policy_replay_bank": [
                ["event one exact text"],
                ["event two exact text"],
                ["event three exact text"],
            ],
            "behavior_eval_suite": {
                "schema": "neural-forge-recovery-eval-suite-v1",
                "split": "candidate",
                "sealed_data_accessed": False,
                "prompts": [{"id": "candidate-1", "text": "2+2="}],
            },
            "frozen_before_preregistration": True,
            "contains_sealed_data": False,
            "publication_authorized": False,
        },
        "replay_payload_sha256",
    )


def _stepwise_token_event(step: int) -> dict[str, Any]:
    prompt = [100 + step, 200 + step]
    return {
        "step": step,
        "rows": [
            {
                "prompt_input_ids": prompt,
                "rollout_input_ids": [*prompt, 300 + step, 400 + step],
            }
        ],
    }


def _plan_payload() -> dict[str, Any]:
    return _with_hash(
        {
            "schema": runner.PLAN_SCHEMA,
            "status": "preregistered",
            "recovery_kwargs": {
                "steps": 4,
                "batch_size": 1,
                "lr": 1e-5,
                "on_policy_interval": 2,
                "on_policy_anchor_kd_weight": 1.0,
                "behavior_eval_checkpoint_steps": [0, 2, 4],
                "adamw_foreach": False,
                "local_files_only": True,
                "trust_remote_code": False,
                "save_best_loss": False,
                "save_best_behavior": False,
            },
            "seeds": {
                "python": 17,
                "torch": 17,
                "cuda": 17,
                "data_order": 17,
            },
            "checkpoint_steps": [0, 2, 4],
            "retained_checkpoint_steps": [2, 4],
            "zero_step_before_optimizer": True,
            "reset_rng_before_each_arm": True,
            "publication_authorized": False,
            "sealed_data_accessed": False,
        },
        "plan_payload_sha256",
    )


def _blank_binding() -> dict[str, Any]:
    return {
        "cache_path": "/authenticated/cache/identity",
        "compiler_identity_sha256": _digest("compiler-identity"),
        "receipt_sha256": _digest("cache-receipt"),
        "compiler_report_sha256": _digest("compiler-report"),
        "artifact_tree": {
            "file_count": 4,
            "total_bytes": 1234,
            "files": [],
            "tree_sha256": _digest("cache-tree"),
        },
        "artifact_tree_sha256": _digest("cache-tree"),
        "state": {
            "state_dict_sha256": _digest("blank-state"),
            "compiler_state_sha256": _digest("blank-compiler-state"),
            "state_key_sha256": _TINY_STATE_KEY_SHA256,
            "state_manifest_sha256": _digest("blank-manifest"),
            "parameter_topology_sha256": _digest("parameter-topology"),
            "buffer_topology_sha256": _digest("buffer-topology"),
            "unique_parameter_count": runner.EXACT_UNIQUE_PARAMETERS,
            "logical_parameter_count": runner.EXACT_UNIQUE_PARAMETERS,
        },
        "state_dict_sha256": _digest("blank-state"),
        "compiler_state_sha256": _digest("blank-compiler-state"),
        "state_key_sha256": _TINY_STATE_KEY_SHA256,
        "state_manifest_sha256": _digest("blank-manifest"),
        "parameter_topology_sha256": _digest("parameter-topology"),
        "buffer_topology_sha256": _digest("buffer-topology"),
        "config_file_sha256": _digest("config-file"),
        "config_payload_sha256": _TINY_CONFIG_PAYLOAD_SHA256,
        "config": dict(runner.blank.CANONICAL_CONFIG),
        "exact_unique_parameters": runner.EXACT_UNIQUE_PARAMETERS,
        "fresh_reload_verified": True,
        "model_loaded_for_authentication": False,
        "capability_proven": False,
        "publication_receipt": False,
    }


@pytest.fixture
def contract_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prereg_path = tmp_path / "v4_prereg.json"
    run_path = tmp_path / "v4_run.json"
    winner_path = tmp_path / "v4_winner.json"
    replay_path = tmp_path / "replay.json"
    plan_path = tmp_path / "plan.json"
    prereg = {
        "schema": "fake-authenticated-v4-prereg",
        "source_v3_sha256": _digest("v3-root"),
        "basis_sha256": _digest("basis-root"),
        "mask_sha256": _digest("mask-root"),
    }
    run = {
        "sealed_forward_count": 0,
        "sealed_evaluation_accessed": False,
        "publication_attempted": False,
        "frontier_sha256": _digest("frontier"),
    }
    winner_selection = {
        "candidate_only": True,
        "sealed_evaluation_accessed": False,
        "publication_attempted": False,
        "winner_sha256": _digest("winner"),
    }
    hashes = {
        "prereg": _write_json(prereg_path, prereg),
        "run": _write_json(run_path, run),
        "winner": _write_json(winner_path, winner_selection),
        "replay": _write_json(replay_path, _replay_payload()),
        "plan": _write_json(plan_path, _plan_payload()),
    }
    fake_contract = SimpleNamespace(
        evaluation_contract_sha256=_digest("v4-evaluation"),
    )
    winner = {
        "arm_id": "gud-winner",
        "arm_index": 2,
        "step": 24,
        "portable_factor_state_sha256": _digest("portable-v4"),
        "runtime_factor_state_sha256": _digest("runtime-v4"),
        "factor_pack": str(tmp_path / "winner.pt"),
        "factor_pack_sha256": _digest("factor-pack-file"),
        "candidate_frontier": {
            "path": str(tmp_path / "frontier.json"),
            "sha256": _digest("frontier"),
        },
    }

    monkeypatch.setattr(
        runner.tournament,
        "authenticate_guided_gud_candidate_preregistration",
        lambda *_args, **_kwargs: fake_contract,
    )
    monkeypatch.setattr(
        runner.publication,
        "_winner_payload",
        lambda *_args, **_kwargs: (
            winner,
            {"schema": "neural-forge-target-native-factor-pack-v4"},
        ),
    )
    monkeypatch.setattr(runner, "_compiled_blank_binding", lambda _v4: _blank_binding())
    monkeypatch.setattr(
        child_artifact,
        "model_state_fingerprint",
        _contract_model_state_fingerprint,
    )

    return {
        "paths": {
            "tournament_preregistration": prereg_path,
            "tournament_run_report": run_path,
            "winner_selection": winner_path,
            "replay_bank": replay_path,
            "recovery_plan": plan_path,
        },
        "hashes": hashes,
        "winner": winner,
    }


def _build(inputs) -> tuple[dict[str, Any], Any]:
    paths = inputs["paths"]
    hashes = inputs["hashes"]
    return runner.build_recovery_ab_preregistration_payload(
        tournament_preregistration=paths["tournament_preregistration"],
        tournament_preregistration_sha256=hashes["prereg"],
        tournament_run_report=paths["tournament_run_report"],
        tournament_run_report_sha256=hashes["run"],
        winner_selection=paths["winner_selection"],
        winner_selection_sha256=hashes["winner"],
        replay_bank=paths["replay_bank"],
        replay_bank_sha256=hashes["replay"],
        recovery_plan=paths["recovery_plan"],
        recovery_plan_sha256=hashes["plan"],
        builder_path=Path(builder.__file__).resolve(),
    )


def _write_prereg(tmp_path: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    path = tmp_path / "ab_preregistration.json"
    return path, _write_json(path, payload)


def test_prereg_binds_candidate_blank_code_data_and_symmetric_schedule(
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)

    assert payload["status"] == "experimental-unaccepted"
    assert payload["v4_candidate"]["candidate_only"] is True
    assert payload["v4_candidate"]["sealed_evaluation_accessed"] is False
    assert payload["publication_authorized"] is False
    assert payload["sealed_data_accessed"] is False
    assert payload["accepted_artifact"] is False
    assert payload["exact_unique_parameters"] == 1_410_567_168
    assert payload["compiled_blank"]["fresh_reload_verified"] is True
    assert payload["compiled_blank"]["state_dict_sha256"] == _digest("blank-state")
    assert payload["compiled_blank"]["parameter_topology_sha256"] == _digest(
        "parameter-topology"
    )
    assert [arm["arm_id"] for arm in payload["arms"]] == ["folded", "blank"]
    assert (
        len({arm["shared_execution_contract_sha256"] for arm in payload["arms"]}) == 1
    )
    assert payload["recovery_plan"]["on_policy_event_steps"] == [1, 2, 4]
    assert payload["replay_bank"]["on_policy_event_count"] == 3
    assert payload["replay_bank"]["stepwise_event_count"] == 0
    assert payload["recovery_plan"]["stepwise_donor_kd_weight"] == 0.0
    assert payload["recovery_plan"]["stepwise_replay_bank_required"] is False
    assert payload["recovery_plan"]["adamw_foreach"] is False
    assert payload["maximum_retained_checkpoints_per_arm"] == 3
    assert payload["maximum_total_retained_checkpoints"] == 6
    roots = payload["transitive_sha256_roots"]
    assert _digest("v3-root") in roots["v4_preregistration"].values()
    assert _digest("basis-root") in roots["v4_preregistration"].values()
    assert _digest("mask-root") in roots["v4_preregistration"].values()
    assert payload["production_model_hook_implemented"] is False


def test_authentication_rebuilds_every_bound_source_and_rejects_drift(
    tmp_path: Path,
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)
    path, digest = _write_prereg(tmp_path, payload)
    contract = runner.authenticate_recovery_ab_preregistration(path, digest)
    assert contract.preflight_report()["status"] == "preflight-passed"
    assert contract.preflight_report()["model_weights_loaded"] is False

    replay_path = contract_inputs["paths"]["replay_bank"]
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["calibration_texts"][0] = "tampered"
    _write_json(replay_path, replay)
    with pytest.raises(Exception, match="absent or changed"):
        runner.authenticate_recovery_ab_preregistration(path, digest)


def test_candidate_boundary_and_replay_schedule_fail_closed(
    contract_inputs,
) -> None:
    run_path = contract_inputs["paths"]["tournament_run_report"]
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["sealed_evaluation_accessed"] = True
    contract_inputs["hashes"]["run"] = _write_json(run_path, run)
    with pytest.raises(runner.FoldedBlankRecoveryABError, match="sealed-free"):
        _build(contract_inputs)

    run["sealed_evaluation_accessed"] = False
    contract_inputs["hashes"]["run"] = _write_json(run_path, run)
    replay_path = contract_inputs["paths"]["replay_bank"]
    replay = _replay_payload()
    replay["on_policy_replay_bank"].pop()
    body = dict(replay)
    body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)
    with pytest.raises(runner.FoldedBlankRecoveryABError, match="event count"):
        _build(contract_inputs)


@pytest.mark.parametrize(
    ("split", "sealed_flag_present", "sealed_flag"),
    [
        ("SEALED", True, False),
        ("sealed-holdout", True, False),
        (None, True, False),
        ("candidate", False, False),
        ("candidate", True, True),
    ],
)
def test_replay_suite_requires_explicit_candidate_sealed_free_metadata(
    contract_inputs,
    split,
    sealed_flag_present: bool,
    sealed_flag: bool,
) -> None:
    replay_path = contract_inputs["paths"]["replay_bank"]
    replay = _replay_payload()
    suite = replay["behavior_eval_suite"]
    if split is None:
        suite.pop("split")
    else:
        suite["split"] = split
    if sealed_flag_present:
        suite["sealed_data_accessed"] = sealed_flag
    else:
        suite.pop("sealed_data_accessed")
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)

    with pytest.raises(runner.FoldedBlankRecoveryABError, match="sealed-free"):
        _build(contract_inputs)


def test_replay_and_plan_reject_non_text_or_unknown_recovery_inputs(
    contract_inputs,
) -> None:
    replay_path = contract_inputs["paths"]["replay_bank"]
    replay = _replay_payload()
    replay["calibration_texts"][0] = 7
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)
    with pytest.raises(runner.FoldedBlankRecoveryABError, match="only strings"):
        _build(contract_inputs)

    contract_inputs["hashes"]["replay"] = _write_json(
        replay_path,
        _replay_payload(),
    )
    plan_path = contract_inputs["paths"]["recovery_plan"]
    plan = _plan_payload()
    plan["recovery_kwargs"]["stepz"] = 4
    plan_body = dict(plan)
    plan_body.pop("plan_payload_sha256")
    plan["plan_payload_sha256"] = runner.canonical_sha256(plan_body)
    contract_inputs["hashes"]["plan"] = _write_json(plan_path, plan)
    with pytest.raises(runner.FoldedBlankRecoveryABError, match="unsupported names"):
        _build(contract_inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("local_files_only", False),
        ("local_files_only", "true"),
        ("trust_remote_code", True),
    ],
)
def test_recovery_plan_requires_local_only_non_remote_loading(
    contract_inputs,
    field: str,
    value: Any,
) -> None:
    plan_path = contract_inputs["paths"]["recovery_plan"]
    plan = _plan_payload()
    plan["recovery_kwargs"][field] = value
    plan_body = dict(plan)
    plan_body.pop("plan_payload_sha256")
    plan["plan_payload_sha256"] = runner.canonical_sha256(plan_body)
    contract_inputs["hashes"]["plan"] = _write_json(plan_path, plan)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="local-only trusted model loading",
    ):
        _build(contract_inputs)


def test_stepwise_donor_requires_one_frozen_ordered_event_per_optimizer_step(
    contract_inputs,
) -> None:
    replay_path = contract_inputs["paths"]["replay_bank"]
    plan_path = contract_inputs["paths"]["recovery_plan"]
    replay = _replay_payload()
    replay["stepwise_replay_bank"] = [
        _stepwise_token_event(step) for step in range(1, 5)
    ]
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    plan = _plan_payload()
    plan["recovery_kwargs"]["stepwise_donor_kd_weight"] = 0.75
    plan_body = dict(plan)
    plan_body.pop("plan_payload_sha256")
    plan["plan_payload_sha256"] = runner.canonical_sha256(plan_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)
    contract_inputs["hashes"]["plan"] = _write_json(plan_path, plan)

    payload, _v4 = _build(contract_inputs)
    assert payload["replay_bank"]["stepwise_event_count"] == 4
    assert payload["recovery_plan"]["stepwise_event_steps"] == [1, 2, 3, 4]
    assert payload["recovery_plan"]["stepwise_replay_bank_required"] is True
    shared = payload["shared_execution_contract"]
    assert shared["stepwise_event_count"] == 4
    assert shared["same_stepwise_replay_bank_required_for_both_arms"] is True
    assert shared["dynamic_stepwise_donor_generation_forbidden"] is True
    assert (
        shared["stepwise_replay_bank_schema"]
        == runner.hf_recovery._STEPWISE_REPLAY_SCHEMA
    )
    frozen = runner.hf_recovery._freeze_stepwise_replay_bank(
        replay["stepwise_replay_bank"]
    )
    assert shared["stepwise_replay_bank_sha256"] == (
        runner.hf_recovery._stepwise_replay_bank_sha256(frozen)
    )
    assert (
        len({arm["shared_execution_contract_sha256"] for arm in payload["arms"]}) == 1
    )

    replay["stepwise_replay_bank"].pop()
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)
    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="exactly one ordered event",
    ):
        _build(contract_inputs)

    replay.pop("stepwise_replay_bank")
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)
    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="exactly one ordered event",
    ):
        _build(contract_inputs)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda bank: bank.__setitem__(0, ["unsafe text replay"]),
        lambda bank: bank[0].__setitem__("step", 2),
        lambda bank: bank[0]["rows"][0].__setitem__("rollout_input_ids", [999, 301]),
        lambda bank: bank[0]["rows"][0].__setitem__("prompt_input_ids", [True, 201]),
    ],
)
def test_stepwise_donor_token_bank_rejects_unsafe_or_wrong_events(
    contract_inputs,
    mutate,
) -> None:
    replay_path = contract_inputs["paths"]["replay_bank"]
    plan_path = contract_inputs["paths"]["recovery_plan"]
    replay = _replay_payload()
    replay["stepwise_replay_bank"] = [
        _stepwise_token_event(step) for step in range(1, 5)
    ]
    mutate(replay["stepwise_replay_bank"])
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    plan = _plan_payload()
    plan["recovery_kwargs"]["stepwise_donor_kd_weight"] = 0.75
    plan_body = dict(plan)
    plan_body.pop("plan_payload_sha256")
    plan["plan_payload_sha256"] = runner.canonical_sha256(plan_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)
    contract_inputs["hashes"]["plan"] = _write_json(plan_path, plan)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="token replay bank|exactly one ordered event",
    ):
        _build(contract_inputs)


def _receipt(body: dict[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["receipt_sha256"] = runner.canonical_sha256(result)
    return result


def _materialization_receipts(contract, tmp_path: Path) -> dict[str, dict[str, Any]]:
    blank = contract.raw["compiled_blank"]
    winner = contract.raw["v4_candidate"]["winner"]
    folded_model = _TinyFoldedModel()
    folded_initial_state_sha256 = _contract_model_state_fingerprint(folded_model)[
        "state_dict_sha256"
    ]
    bridge_native_child_state_sha256 = _digest("bridge-native-child-state")
    bridge_live_factor_state_sha256 = _digest("bridge-live-factor-state")
    materialized_bridge_state_sha256 = _dense_model_state_sha256(folded_model)
    experiment_contract = {
        "schema": runner.EXPERIMENTAL_RECOVERY_CONTRACT_SCHEMA,
        "recovery_ab_preregistration_sha256": contract.file_sha256,
        "recovery_ab_semantic_contract_sha256": contract.raw[
            "semantic_contract_sha256"
        ],
        "shared_execution_contract_sha256": contract.raw[
            "shared_execution_contract_sha256"
        ],
        "factor_state_sha256": winner["portable_factor_state_sha256"],
        "factor_payload_schema": contract.raw["v4_candidate"]["factor_pack_schema"],
        "native_child_state_sha256": bridge_native_child_state_sha256,
        "blank_control_state_sha256": bridge_native_child_state_sha256,
        "compiled_blank_cache_state_dict_sha256": blank["state_dict_sha256"],
        "restored_runtime_factor_state_sha256": winner["runtime_factor_state_sha256"],
        "restored_portable_factor_state_sha256": winner["portable_factor_state_sha256"],
        "bridge_native_child_state_sha256": bridge_native_child_state_sha256,
        "bridge_live_factor_state_sha256": bridge_live_factor_state_sha256,
        "exact_unique_parameters": runner.EXACT_UNIQUE_PARAMETERS,
        "publication_authorized": False,
        "sealed_data_accessed": False,
    }
    bridge_receipt = _receipt(
        {
            "schema": runner.EXPERIMENTAL_RECOVERY_MATERIALIZATION_RECEIPT_SCHEMA,
            "experiment_contract_sha256": runner.canonical_sha256(experiment_contract),
            "factor_state_sha256": winner["portable_factor_state_sha256"],
            "live_factor_state_sha256": bridge_live_factor_state_sha256,
            "native_child_state_sha256": bridge_native_child_state_sha256,
            "exact_unique_parameters": runner.EXACT_UNIQUE_PARAMETERS,
            "factor_payload_schema": contract.raw["v4_candidate"]["factor_pack_schema"],
            "status": "experimental-unaccepted",
            "publication_authorized": False,
            "sealed_data_accessed": False,
            "behavior_capability_proven": False,
            "recovery_training_required": True,
            "external_experiment_contract_revalidated_by_bridge": False,
            "external_factor_pack_revalidated_by_bridge": False,
            "live_factor_tensor_state_revalidated_by_bridge": True,
            "native_child_state_revalidated_by_bridge": True,
        }
    )
    bridge_report = {
        "schema": runner.EXACT_TARGET_EXPERIMENTAL_RECOVERY_MATERIALIZATION_SCHEMA,
        "status": "experimental-unaccepted",
        "exact_unique_parameters": runner.EXACT_UNIQUE_PARAMETERS,
        "exact_budget_match": True,
        "native_before": _folded_native_certificate(folded_model),
        "native_materialized": _folded_native_certificate(folded_model),
        "materialized_native_state_sha256": materialized_bridge_state_sha256,
        "native_child_state_sha256": bridge_native_child_state_sha256,
        "live_factor_state_sha256": bridge_live_factor_state_sha256,
        "materialized_registered_adapter_parameter_count": 0,
        "materialized_optimizer_parameter_count": 0,
        "materialized_registered_source_sized_sidecar_parameter_count": 0,
        "experimental_recovery_contract": experiment_contract,
        "experimental_recovery_receipt": bridge_receipt,
        "standalone_structure_materialized": True,
        "serialization_only_context": True,
        "serialized_state_must_match_materialized_state_sha256": True,
        "fresh_reload_and_recovery_required": True,
        "publication_authorized": False,
        "accepted_artifact": False,
        "sealed_data_accessed": False,
        "behavior_capability_proven": False,
        "external_factor_pack_revalidated_by_bridge": False,
        "external_factor_pack_receipt_bound": False,
        "contract_factor_state_sha256_is_external_label_only": True,
        "live_factor_state_sha256_is_bridge_authenticated": True,
    }
    bridge_report["materialization_transaction_sha256"] = runner.canonical_sha256(
        bridge_report
    )
    bridge_report_bytes = json.dumps(
        bridge_report,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    publication_index = len(list(tmp_path.glob("folded-artifact-publication-*")))
    artifact_publication_root = (
        tmp_path / f"folded-artifact-publication-{publication_index}"
    )
    artifact_receipt = child_artifact.publish_experimental_child_artifact(
        artifact_publication_root,
        model=folded_model,
        tokenizer=_TinyFoldedTokenizer(),
        materialization_context=_folded_materialization_context(bridge_report_bytes),
        expectations=child_artifact.ExperimentalChildArtifactExpectations(
            preregistration_sha256=contract.file_sha256,
            shared_execution_contract_sha256=contract.raw[
                "shared_execution_contract_sha256"
            ],
            config_payload_sha256=blank["config_payload_sha256"],
            state_key_sha256=blank["state_key_sha256"],
            parameter_topology_sha256=blank["parameter_topology_sha256"],
            buffer_topology_sha256=blank["buffer_topology_sha256"],
            exact_unique_parameters=runner.EXACT_UNIQUE_PARAMETERS,
            exact_logical_parameters=runner.EXACT_UNIQUE_PARAMETERS,
            arm_id="folded",
        ),
        model_loader=_TinyFoldedModel.from_pretrained,
        native_certifier=_folded_native_certificate,
        runtime_auditor=_folded_runtime_audit,
    )
    common = {
        "schema": runner.MATERIALIZATION_RECEIPT_SCHEMA,
        "status": "experimental-unaccepted",
        "exact_unique_parameters": runner.EXACT_UNIQUE_PARAMETERS,
        "config_payload_sha256": blank["config_payload_sha256"],
        "parameter_topology_sha256": blank["parameter_topology_sha256"],
        "buffer_topology_sha256": blank["buffer_topology_sha256"],
        "shared_execution_contract_sha256": contract.raw[
            "shared_execution_contract_sha256"
        ],
        "optimizer_steps": 0,
        "sealed_data_accessed": False,
        "publication_authorized": False,
        "accepted_artifact": False,
    }
    return {
        "folded": _receipt(
            {
                **common,
                "arm_id": "folded",
                "initial_state_sha256": folded_initial_state_sha256,
                "source_portable_factor_state_sha256": winner[
                    "portable_factor_state_sha256"
                ],
                "source_factor_pack_sha256": winner["factor_pack_sha256"],
                "factor_source_binding_sha256": contract.raw["v4_candidate"][
                    "factor_source_binding_sha256"
                ],
                "factor_pack_authenticated_separately": True,
                "materialization_evidence_sha256": bridge_receipt["receipt_sha256"],
                "materialization_transaction_sha256": bridge_report[
                    "materialization_transaction_sha256"
                ],
                "artifact_publication_root": str(artifact_publication_root),
                "artifact_receipt_sha256": artifact_receipt.receipt_sha256,
                "bridge_native_child_state_sha256": (bridge_native_child_state_sha256),
                "bridge_live_factor_state_sha256": bridge_live_factor_state_sha256,
                "materialized_bridge_state_sha256": materialized_bridge_state_sha256,
            }
        ),
        "blank": _receipt(
            {
                **common,
                "arm_id": "blank",
                "initial_state_sha256": blank["state_dict_sha256"],
                "source_portable_factor_state_sha256": None,
                "source_factor_pack_sha256": None,
                "factor_source_binding_sha256": None,
                "factor_pack_authenticated_separately": False,
                "materialization_evidence_sha256": blank["receipt_sha256"],
                "materialization_transaction_sha256": None,
                "artifact_publication_root": None,
                "artifact_receipt_sha256": None,
                "bridge_native_child_state_sha256": None,
                "bridge_live_factor_state_sha256": None,
                "materialized_bridge_state_sha256": None,
            }
        ),
    }


def _zero_step_receipts(contract, materialization) -> dict[str, dict[str, Any]]:
    common = {
        "schema": runner.ZERO_STEP_RECEIPT_SCHEMA,
        "status": "experimental-unaccepted",
        "shared_execution_contract_sha256": contract.raw[
            "shared_execution_contract_sha256"
        ],
        "replay_bank_payload_sha256": contract.raw["replay_bank"]["payload_sha256"],
        "checkpoint_step": 0,
        "optimizer_steps": 0,
        "sealed_data_accessed": False,
        "publication_authorized": False,
        "accepted_artifact": False,
    }
    values = {}
    for arm_id, score in (("folded", 0.25), ("blank", 0.0)):
        metrics = {"coherent_action_rate": score, "sample_count": 4}
        values[arm_id] = _receipt(
            {
                **common,
                "arm_id": arm_id,
                "initial_state_sha256": materialization["receipts"][arm_id][
                    "initial_state_sha256"
                ],
                "metrics": metrics,
                "metrics_sha256": runner.canonical_sha256(metrics),
                "evaluator_implementation_sha256": contract.raw["implementation"][
                    "recovery_implementation_sha256"
                ],
            }
        )
    return values


def test_materialization_and_zero_step_receipts_remain_unaccepted(
    tmp_path: Path,
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)
    path, digest = _write_prereg(tmp_path, payload)
    contract = runner.authenticate_recovery_ab_preregistration(path, digest)
    blank = contract.raw["compiled_blank"]
    artifact_hooks = _folded_artifact_hooks()
    materialization = runner.authenticate_materialization_receipts(
        contract,
        _materialization_receipts(contract, tmp_path),
        folded_artifact_hooks=artifact_hooks,
    )
    assert materialization["status"] == "materialization-authenticated"
    assert materialization["folded_and_blank_initial_states_distinct"] is True
    assert materialization["folded_artifact_independent_fresh_reload_verified"] is True
    assert materialization["model_loading_performed_by_this_authenticator"] is True
    assert materialization["optimizer_steps"] == 0
    folded_binding = materialization["authenticated_materialization_reports"]["folded"]
    assert (
        folded_binding["fresh_reload_state_dict_sha256"]
        == materialization["receipts"]["folded"]["initial_state_sha256"]
    )
    assert "materialized_cache_state_dict_sha256" not in folded_binding
    zero_step = runner.authenticate_zero_step_receipts(
        contract,
        materialization,
        _zero_step_receipts(contract, materialization),
        folded_artifact_hooks=artifact_hooks,
    )
    assert zero_step["status"] == "zero-step-authenticated"
    assert zero_step["training_launched"] is False
    assert zero_step["publication_authorized"] is False
    assert zero_step["sealed_data_accessed"] is False
    assert zero_step["next_action"] == "explicit-production-recovery-hook-required"

    for field, value in (
        ("preregistration_sha256", _digest("wrong-preregistration")),
        ("publication_authorized", True),
    ):
        tampered_materialization = json.loads(json.dumps(materialization))
        tampered_materialization[field] = value
        with pytest.raises(runner.FoldedBlankRecoveryABError):
            runner.authenticate_zero_step_receipts(
                contract,
                tampered_materialization,
                _zero_step_receipts(contract, materialization),
                folded_artifact_hooks=artifact_hooks,
            )

    extra_materialization = json.loads(json.dumps(materialization))
    extra_materialization["unbound"] = True
    with pytest.raises(runner.FoldedBlankRecoveryABError):
        runner.authenticate_zero_step_receipts(
            contract,
            extra_materialization,
            _zero_step_receipts(contract, materialization),
            folded_artifact_hooks=artifact_hooks,
        )

    bad = _materialization_receipts(contract, tmp_path)
    bad["blank"]["exact_unique_parameters"] -= 1
    with pytest.raises(runner.FoldedBlankRecoveryABError):
        runner.authenticate_materialization_receipts(
            contract,
            bad,
            folded_artifact_hooks=artifact_hooks,
        )

    forged_materialization = {
        "status": "materialization-authenticated",
        "receipts": {
            "folded": {"initial_state_sha256": _digest("forged-folded")},
            "blank": {"initial_state_sha256": _digest("forged-blank")},
        },
    }
    with pytest.raises(runner.FoldedBlankRecoveryABError):
        runner.authenticate_zero_step_receipts(
            contract,
            forged_materialization,
            _zero_step_receipts(contract, materialization),
            folded_artifact_hooks=artifact_hooks,
        )

    unrelated = _materialization_receipts(contract, tmp_path)
    unrelated["folded"]["materialization_evidence_sha256"] = blank["receipt_sha256"]
    unrelated["folded"] = _receipt(
        {
            key: value
            for key, value in unrelated["folded"].items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(runner.FoldedBlankRecoveryABError):
        runner.authenticate_materialization_receipts(
            contract,
            unrelated,
            folded_artifact_hooks=artifact_hooks,
        )

    sidecar = _materialization_receipts(contract, tmp_path)
    report_path = (
        Path(sidecar["folded"]["artifact_publication_root"])
        / child_artifact.BRIDGE_REPORT_FILENAME
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["materialized_registered_source_sized_sidecar_parameter_count"] = 1
    report.pop("materialization_transaction_sha256")
    report["materialization_transaction_sha256"] = runner.canonical_sha256(report)
    sidecar["folded"]["materialization_transaction_sha256"] = report[
        "materialization_transaction_sha256"
    ]
    _write_canonical_json(report_path, report)
    sidecar["folded"] = _receipt(
        {
            key: value
            for key, value in sidecar["folded"].items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(runner.FoldedBlankRecoveryABError):
        runner.authenticate_materialization_receipts(
            contract,
            sidecar,
            folded_artifact_hooks=artifact_hooks,
        )


def test_materialization_rejects_noncanonical_bridge_report_bytes(
    tmp_path: Path,
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)
    path, digest = _write_prereg(tmp_path, payload)
    contract = runner.authenticate_recovery_ab_preregistration(path, digest)
    receipts = _materialization_receipts(contract, tmp_path)
    report_path = (
        Path(receipts["folded"]["artifact_publication_root"])
        / child_artifact.BRIDGE_REPORT_FILENAME
    )
    report = json.loads(report_path.read_text(encoding="ascii"))
    _write_json(report_path, report)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="failed independent fresh reload",
    ):
        runner.authenticate_materialization_receipts(
            contract,
            receipts,
            folded_artifact_hooks=_folded_artifact_hooks(),
        )


def test_materialization_rejects_changed_folded_artifact_bytes(
    tmp_path: Path,
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)
    path, digest = _write_prereg(tmp_path, payload)
    contract = runner.authenticate_recovery_ab_preregistration(path, digest)
    receipts = _materialization_receipts(contract, tmp_path)
    artifact_path = (
        Path(receipts["folded"]["artifact_publication_root"])
        / child_artifact.ARTIFACT_DIRECTORY_NAME
        / "tokenizer.json"
    )
    artifact_path.write_bytes(artifact_path.read_bytes() + b"\n")

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="failed independent fresh reload",
    ):
        runner.authenticate_materialization_receipts(
            contract,
            receipts,
            folded_artifact_hooks=_folded_artifact_hooks(),
        )


def test_materialization_rejects_changed_external_artifact_receipt(
    tmp_path: Path,
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)
    path, digest = _write_prereg(tmp_path, payload)
    contract = runner.authenticate_recovery_ab_preregistration(path, digest)
    receipts = _materialization_receipts(contract, tmp_path)
    receipt_path = (
        Path(receipts["folded"]["artifact_publication_root"])
        / child_artifact.ARTIFACT_RECEIPT_FILENAME
    )
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="failed independent fresh reload",
    ):
        runner.authenticate_materialization_receipts(
            contract,
            receipts,
            folded_artifact_hooks=_folded_artifact_hooks(),
        )


def test_materialization_rejects_changed_independent_reload_evidence(
    tmp_path: Path,
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)
    path, digest = _write_prereg(tmp_path, payload)
    contract = runner.authenticate_recovery_ab_preregistration(path, digest)
    receipts = _materialization_receipts(contract, tmp_path)

    def changed_loader(source: Path) -> _TinyFoldedModel:
        loaded = _TinyFoldedModel.from_pretrained(source)
        with torch.no_grad():
            loaded.weight.add_(1.0)
        return loaded

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="failed independent fresh reload",
    ):
        runner.authenticate_materialization_receipts(
            contract,
            receipts,
            folded_artifact_hooks=_folded_artifact_hooks(model_loader=changed_loader),
        )


def test_materialization_rejects_legacy_bridge_only_folded_receipt(
    tmp_path: Path,
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)
    path, digest = _write_prereg(tmp_path, payload)
    contract = runner.authenticate_recovery_ab_preregistration(path, digest)
    receipts = _materialization_receipts(contract, tmp_path)
    folded_body = {
        key: value
        for key, value in receipts["folded"].items()
        if key
        not in {
            "receipt_sha256",
            "artifact_publication_root",
            "artifact_receipt_sha256",
        }
    }
    publication_root = Path(receipts["folded"]["artifact_publication_root"])
    report_path = publication_root / child_artifact.BRIDGE_REPORT_FILENAME
    folded_body.update(
        {
            "materialization_report": str(report_path),
            "materialization_report_file_sha256": runner.stage._file_sha256(
                report_path
            ),
        }
    )
    receipts["folded"] = _receipt(folded_body)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="fields changed",
    ):
        runner.authenticate_materialization_receipts(
            contract,
            receipts,
            folded_artifact_hooks=_folded_artifact_hooks(),
        )


def test_preparer_writes_only_atomic_model_free_contract(
    tmp_path: Path,
    contract_inputs,
) -> None:
    output = tmp_path / "prepared"
    report = builder.prepare_recovery_ab(
        output_dir=output,
        **contract_inputs["paths"],
    )
    assert report["status"] == "ready-experimental-unaccepted"
    assert report["model_weights_loaded"] is False
    assert report["gpu_work_launched"] is False
    assert report["publication_authorized"] is False
    assert report["sealed_data_accessed"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "build_report.json",
        "preregistration.json",
    ]
    prereg = output / "preregistration.json"
    runner.authenticate_recovery_ab_preregistration(
        prereg,
        runner.stage._file_sha256(prereg),
    )


def test_scaffold_source_has_no_model_or_gpu_execution_path() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "from_pretrained(" not in source
    assert "torch.load(" not in source
    assert ".cuda(" not in source
    assert '"production_model_hook_implemented": False' in source
    assert "verify_compiled_child_cache" in source
    assert "_winner_payload" in source
    assert "pydantic" not in source.lower()
    assert "sqlalchemy" not in source.lower()
