from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qwen_exact_target_native_guided_winner_publication as publication  # noqa: E402
import run_qwen_exact_target_native_stage as stage  # noqa: E402
import prepare_qwen_exact_target_native_guided_winner_publication as winner_prepare  # noqa: E402
from neural_forge.hf_exact_candidate_frontier import (  # noqa: E402
    ExactCandidateFrontier,
    ExactCandidateFrontierError,
    artifact_tree,
    authenticate_terminal_candidate,
    file_sha256,
)
from neural_forge.hf_target_native_behavior_distillation import (  # noqa: E402
    BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _metrics(
    *,
    split: str = "calibration-selection",
    action_accuracy: float = 1.0,
) -> dict[str, Any]:
    return {
        "split": split,
        "action_count": 192,
        "action_agreement_count": round(192 * action_accuracy),
        "action_accuracy": action_accuracy,
        "parent_action_mean_logprob": -1.0,
        "parent_action_topk_fraction": 1.0,
        "parent_action_mean_rank": 1.0,
        "parent_action_worst_rank": 1,
        "rollout_count": 24,
        "exact_rollout_count": 6,
        "exact_rollout_fraction": 0.25,
        "rollout_token_count": 192,
        "token_match_count": 120,
        "token_match_fraction": 0.625,
        "lcp_token_count": 80,
        "lcp_fraction": 80 / 192,
        "lcp_at_least_4_count": 1,
        "repeat_fraction_excess": 0.1,
        "adjacent_repeat_fraction_excess": 0.1,
        "longest_run_excess": 0,
        "report_sha256": _digest(f"metrics:{split}:{action_accuracy}"),
    }


def _thresholds() -> dict[str, Any]:
    return stage.BehaviorThresholds(
        minimum_action_accuracy=0.5,
        minimum_exact_rollout_fraction=0.25,
        minimum_token_match_fraction=0.5,
        minimum_lcp_fraction=0.35,
        minimum_action_count=192,
        minimum_rollout_count=24,
    ).as_dict()


class _FakeTransaction:
    def __init__(self, events: list[str], runtime: "_FakeRuntime") -> None:
        self.events = events
        self.runtime = runtime
        self.is_live = True
        self.qkvo_handle: Any = object()

    def installation_report(self) -> dict[str, Any]:
        return {"zero_step_identity": True}

    def rollback(self) -> None:
        self.events.append("rollback-base")
        self.is_live = False

    def bake_for_publication(self, receipt: Any) -> dict[str, Any]:
        assert receipt.passed is True
        self.events.append("bake")
        if self.runtime.fail_bake:
            raise RuntimeError("synthetic bake failure")
        return {"composite_bake": True}

    def restore_prepublication_native(self) -> None:
        self.events.append("restore-prepublication-native")
        self.is_live = False


class _FakeRuntime:
    def __init__(
        self,
        events: list[str],
        *,
        candidate_matches: bool = True,
        sealed_passes: bool = True,
        fail_bake: bool = False,
    ) -> None:
        self.events = events
        self.candidate_matches = candidate_matches
        self.sealed_passes = sealed_passes
        self.fail_bake = fail_bake
        self.sealed_forward_count = 0
        self.model = SimpleNamespace()
        self.transaction = _FakeTransaction(events, self)
        self._guided_tranche = SimpleNamespace(
            set_training_stage=lambda value: events.append(f"stage-{value}")
        )
        self.load_count = 0

    def prepare_compiler_cache(self, _contract: Any) -> dict[str, Any]:
        return {"cache_authenticated": True}

    def load_models(self, _contract: Any) -> tuple[Any, Any, Any]:
        self.load_count += 1
        self.events.append("load-models")
        return object(), object(), object()

    def compile(self, *_args: Any) -> Any:
        self.events.append("compile")
        return SimpleNamespace(child_model=self.model, report={"compiled": True})

    def state_sha256(self, _model: Any) -> str:
        return _digest("compiled")

    def install_transaction(self, _compiled: Any) -> _FakeTransaction:
        self.events.append("install-base")
        return self.transaction

    def restore_source_v2(self, *_args: Any) -> dict[str, Any]:
        self.events.append("restore-v2")
        return {"restored": True}

    def install_guided_transaction(self, *_args: Any) -> dict[str, Any]:
        self.events.append("install-guided")
        return {"installed": True}

    def rollback_guided_installation(self, _transaction: Any) -> None:
        self.events.append("rollback-guided")

    def handoff_guided_for_publication(
        self, transaction: Any, _contract: Any
    ) -> dict[str, Any]:
        self.events.append("handoff")
        transaction.qkvo_handle = "composite"
        return {"handoff_complete": True}

    def evaluate_publication_split(
        self,
        _model: Any,
        _parent: Any,
        contract: Any,
        split_name: str,
    ) -> dict[str, Any]:
        self.events.append(f"evaluate-{split_name}")
        if split_name == "sealed":
            assert contract.sealed_marker.exists()
            self.sealed_forward_count += 1
            return _metrics(
                split="heldout-evaluation",
                action_accuracy=1.0 if self.sealed_passes else 0.0,
            )
        if split_name == "fit":
            return _metrics(split="calibration-exploratory")
        result = dict(contract.winner["metrics"])
        if not self.candidate_matches and self.events.count("evaluate-candidate") == 1:
            result["action_accuracy"] = 0.0
        return result

    def save_reload_certify(
        self, _model: Any, _tokenizer: Any, output: Path
    ) -> dict[str, Any]:
        self.events.append("save-reload")
        child = output / "accepted_child"
        child.mkdir()
        (child / "model.bin").write_bytes(b"standalone-child")
        tree = artifact_tree(child)
        return {
            "path": str(child),
            "fresh_reload_performed": True,
            "exact_budget_match": True,
            "exact_unique_parameters": stage.QWEN3_1P4B_EXACT_PARAMETERS,
            "state_keys_match_pre_save": True,
            "state_bytes_match_pre_save": True,
            "state_dict_sha256": _digest("published-state"),
            "published_adapter_parameter_count": 0,
            "published_parent_runtime_reference_count": 0,
            "published_source_sized_sidecar_parameter_count": 0,
            "artifact_tree": tree,
        }

    def independent_reload_certify(
        self, child: Path, expected_parameters: int
    ) -> dict[str, Any]:
        self.events.append("independent-reload")
        return {
            "fresh_reload_performed": True,
            "loaded_from_candidate_path": True,
            "exact_unique_parameters": expected_parameters,
            "state_dict_sha256": _digest("published-state"),
            "published_adapter_parameter_count": 0,
            "parent_runtime_reference_count": 0,
            "source_sized_sidecar_parameter_count": 0,
            "artifact_tree": artifact_tree(child),
        }


def _contract(tmp_path: Path) -> publication.AuthenticatedWinnerPublication:
    prereg = tmp_path / "preregistration.json"
    semantic = _digest("semantic")
    source_evaluation = _digest("source-evaluation")
    source_guided_prereg = tmp_path / "source-guided-preregistration.json"
    source_guided_prereg.write_text(
        json.dumps(
            {
                "evaluation_contract_sha256": source_evaluation,
                "thresholds": {
                    name: _thresholds()
                    for name in ("fit", "candidate", "sealed")
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    portable = _digest("portable-winner")
    winner = {
        "step": 16,
        "metrics": _metrics(),
        "portable_factor_state_sha256": portable,
    }
    prereg_payload = {
        "schema": publication.PREREGISTRATION_SCHEMA,
        "semantic_contract_sha256": semantic,
        "model_free_preregistration": True,
        "publication_authorized_before_sealed_pass": False,
        "source_guided": {
            "preregistration": str(source_guided_prereg),
            "preregistration_sha256": file_sha256(source_guided_prereg),
            "evaluation_contract_sha256": source_evaluation,
        },
        "winner": winner,
    }
    prereg.write_text(
        json.dumps(prereg_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source = SimpleNamespace(
        execution_contract=SimpleNamespace(),
        evaluation_contract_sha256=source_evaluation,
    )
    candidate_split = SimpleNamespace(sha256=_digest("candidate-split"))
    sealed_split = SimpleNamespace(sha256=_digest("sealed-split"))
    tournament = SimpleNamespace(
        source=source,
        splits={"candidate": candidate_split, "sealed": sealed_split},
        schedule=SimpleNamespace(as_dict=lambda: {"steps": [16]}),
    )
    return publication.AuthenticatedWinnerPublication(
        path=prereg,
        file_sha256=file_sha256(prereg),
        tournament=tournament,
        winner=winner,
        factor_payload={"schema": BEHAVIOR_FACTOR_PACK_V3_SCHEMA},
        sealed_marker=tmp_path / "sealed_consumption.json",
        runtime={
            "dtype": "float16",
            "parent_device_map": "cuda:0",
            "target_device_map": "cuda:1",
        },
        raw={
            "semantic_contract_sha256": semantic,
            "splits": {"sealed": {"sha256": sealed_split.sha256}},
        },
    )


@pytest.fixture
def execution_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        publication.tournament,
        "validate_candidate_tournament_cuda_preflight",
        lambda *_a, **_kw: {
            "status": "passed",
            "cuda_device_order": "PCI_BUS_ID",
            "cuda_visible_devices": "4,6",
            "devices": [
                {
                    "role": "parent",
                    "logical_index": 0,
                    "device_map": "cuda:0",
                    "name": "NVIDIA GeForce RTX 3090",
                    "compute_capability": [8, 6],
                },
                {
                    "role": "target",
                    "logical_index": 1,
                    "device_map": "cuda:1",
                    "name": "Quadro RTX 8000",
                    "compute_capability": [7, 5],
                },
            ],
            "models_loaded": False,
            "model_forward_calls": 0,
            "gpu_kernel_work_launched": False,
        },
    )
    monkeypatch.setattr(
        publication.dataclasses,
        "replace",
        lambda _source, **_kw: SimpleNamespace(
            thresholds={
                name: stage.BehaviorThresholds.from_mapping(
                    _thresholds(), f"thresholds.{name}"
                )
                for name in ("fit", "candidate", "sealed")
            }
        ),
    )
    monkeypatch.setattr(stage, "_validate_continuation_compilation", lambda *_a: None)
    monkeypatch.setattr(
        publication, "capture_target_native_factor_state", lambda *_a: {}
    )
    monkeypatch.setattr(
        publication, "authenticate_guided_factor_sources", lambda *_a: {}
    )
    monkeypatch.setattr(
        publication,
        "rebind_recovered_v3_factor_payload",
        lambda _payload, _binding: {
            "schema": BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            "factor_state_sha256": _digest("rebound"),
        },
    )
    monkeypatch.setattr(
        publication,
        "portable_v3_factor_state_sha256",
        lambda *_a: _digest("portable-winner"),
    )
    restored = SimpleNamespace(
        schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        factor_state_sha256=_digest("rebound"),
    )
    monkeypatch.setattr(
        publication.importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            restore_target_native_factor_payload=lambda *_a, **_kw: restored
        ),
    )


def _execute(
    contract: publication.AuthenticatedWinnerPublication,
    output: Path,
    runtime: _FakeRuntime,
) -> dict[str, Any]:
    return publication.execute_winner_publication(
        contract,
        output,
        dtype="float16",
        parent_device_map="cuda:0",
        target_device_map="cuda:1",
        runtime=runtime,
    )


def test_model_free_preregistration_binds_exact_sealed32_and_tamper_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_prereg = tmp_path / "source-guided.json"
    tournament_prereg = tmp_path / "tournament-prereg.json"
    tournament_report = tmp_path / "tournament-report.json"
    winner_selection = tmp_path / "winner.json"
    for path in (
        source_prereg,
        tournament_prereg,
        tournament_report,
        winner_selection,
    ):
        path.write_text("{}\n", encoding="utf-8")

    def split(name: str, count: int) -> Any:
        path = tmp_path / f"{name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            path=path,
            sha256=file_sha256(path),
            expected_count=count,
            row_ids=tuple(f"{name}-{index}" for index in range(count)),
            prompt_sha256s=tuple(
                _digest(f"{name}-prompt-{index}") for index in range(count)
            ),
            payload={"split": name},
        )

    source = SimpleNamespace(
        path=tournament_prereg,
        evaluation_contract_sha256=_digest("tournament-evaluation"),
        runtime={
            "dtype": "float16",
            "parent_device_map": "cuda:0",
            "target_device_map": "cuda:1",
        },
        source=SimpleNamespace(
            path=source_prereg,
            file_sha256=file_sha256(source_prereg),
            evaluation_contract_sha256=_digest("source-guided-evaluation"),
        ),
        splits={
            "fit": split("fit", 40),
            "candidate": split("candidate", 32),
            "sealed": split("sealed", 32),
        },
    )
    winner = {
        "step": 16,
        "portable_factor_state_sha256": _digest("portable-winner"),
        "metrics": _metrics(),
        "metrics_sha256": publication.canonical_sha256(_metrics()),
    }
    monkeypatch.setattr(
        publication.tournament,
        "authenticate_candidate_tournament_preregistration",
        lambda *_a: source,
    )
    monkeypatch.setattr(
        publication,
        "_winner_payload",
        lambda *_a: (winner, {"schema": BEHAVIOR_FACTOR_PACK_V3_SCHEMA}),
    )
    implementation = {
        name: _digest(name)
        for name in (
            "runner_sha256",
            "builder_sha256",
            "tournament_runner_sha256",
            "guided_runner_sha256",
            "exact_stage_runner_sha256",
            "trainer_sha256",
            "checkpoint_recovery_sha256",
            "progressive_bridge_sha256",
            "guided_adapter_sha256",
            "independent_authenticator_sha256",
        )
    }
    monkeypatch.setattr(
        publication, "_implementation_hashes", lambda **_kw: implementation
    )
    payload, authenticated_source, _, _ = (
        publication.build_winner_publication_preregistration_payload(
            tournament_preregistration=tournament_prereg,
            tournament_preregistration_sha256=file_sha256(tournament_prereg),
            tournament_run_report=tournament_report,
            tournament_run_report_sha256=file_sha256(tournament_report),
            winner_selection=winner_selection,
            winner_selection_sha256=file_sha256(winner_selection),
            sealed_consumption_marker=tmp_path / "sealed_consumption.json",
            dtype="float16",
            parent_device_map="cuda:0",
            target_device_map="cuda:1",
            builder_path=Path(publication.__file__),
        )
    )
    assert authenticated_source is source
    assert payload["splits"]["sealed"]["expected_count"] == 32
    assert payload["sealed_consumption"]["maximum_forward_count"] == 1
    assert payload["model_free_preregistration"] is True
    assert payload["model_weights_loaded"] is False
    assert payload["gpu_work_launched"] is False
    assert payload["publication_authorized_before_sealed_pass"] is False
    assert not (tmp_path / "sealed_consumption.json").exists()

    with pytest.raises(
        publication.GuidedWinnerPublicationError,
        match="exactly match the authenticated tournament runtime",
    ):
        publication.build_winner_publication_preregistration_payload(
            tournament_preregistration=tournament_prereg,
            tournament_preregistration_sha256=file_sha256(tournament_prereg),
            tournament_run_report=tournament_report,
            tournament_run_report_sha256=file_sha256(tournament_report),
            winner_selection=winner_selection,
            winner_selection_sha256=file_sha256(winner_selection),
            sealed_consumption_marker=tmp_path / "sealed_consumption.json",
            dtype="float16",
            parent_device_map="cuda:0",
            target_device_map="cuda:0",
            builder_path=Path(publication.__file__),
        )

    prereg = tmp_path / "publication-preregistration.json"
    prereg.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        publication,
        "build_winner_publication_preregistration_payload",
        lambda **_kw: (
            payload,
            source,
            winner,
            {"schema": BEHAVIOR_FACTOR_PACK_V3_SCHEMA},
        ),
    )
    authenticated = publication.authenticate_winner_publication_preregistration(
        prereg, file_sha256(prereg)
    )
    assert authenticated.preflight_report()["models_loaded"] is False

    tampered = dict(payload)
    tampered["publication_authorized_before_sealed_pass"] = True
    prereg.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(
        publication.GuidedWinnerPublicationError,
        match="differs from authenticated sources",
    ):
        publication.authenticate_winner_publication_preregistration(
            prereg, file_sha256(prereg)
        )


def test_bound_inputs_and_marker_reject_symlink_ancestors(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    source = real / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    linked = tmp_path / "linked"
    try:
        os.symlink(real, linked, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(
        publication.GuidedWinnerPublicationError,
        match="cannot traverse a symlink",
    ):
        publication._bound_file(
            linked / "source.json",
            file_sha256(source),
            "symlinked source",
        )
    with pytest.raises(
        publication.GuidedWinnerPublicationError,
        match="cannot traverse a symlink",
    ):
        publication._resolve(
            linked / "sealed_consumption.json",
            base=tmp_path,
            name="sealed consumption marker",
        )


def test_partial_save_pretrained_failure_removes_only_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    unrelated = output / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")

    class FailingModel(torch.nn.Linear):
        def save_pretrained(self, path: Path, **_kwargs: Any) -> None:
            destination = Path(path)
            destination.mkdir()
            (destination / "partial.bin").write_bytes(b"partial")
            raise RuntimeError("synthetic partial save failure")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=object),
    )
    runtime = stage.ProductionStageRuntime(
        dtype="float16",
        parent_device_map="cuda:0",
        target_device_map="cuda:1",
    )
    with pytest.raises(RuntimeError, match="synthetic partial save failure"):
        runtime.save_reload_certify(
            FailingModel(2, 2),
            SimpleNamespace(save_pretrained=lambda *_a, **_kw: None),
            output,
        )
    assert not (output / ".accepted_child.staging").exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_builder_uses_owned_sibling_staging_and_atomic_final_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "publication"
    inputs = [tmp_path / f"input-{index}.json" for index in range(3)]
    for path in inputs:
        path.write_text("{}\n", encoding="utf-8")
    portable = _digest("portable")
    payload = {
        "schema": publication.PREREGISTRATION_SCHEMA,
        "semantic_contract_sha256": _digest("semantic"),
        "winner": {"portable_factor_state_sha256": portable},
    }
    authenticated_paths: list[Path] = []

    def build(**kwargs: Any) -> tuple[Any, ...]:
        assert kwargs["sealed_consumption_marker"] == (
            output / "sealed_consumption.json"
        )
        return payload, object(), payload["winner"], {}

    def authenticate(path: Path, _sha256: str) -> Any:
        authenticated_paths.append(Path(path))
        return SimpleNamespace(sealed_marker=output / "sealed_consumption.json")

    real_replace = winner_prepare.os.replace
    replacements: list[tuple[Path, Path]] = []

    def replace(source: Path, destination: Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(
        winner_prepare.publication,
        "build_winner_publication_preregistration_payload",
        build,
    )
    monkeypatch.setattr(
        winner_prepare.publication,
        "authenticate_winner_publication_preregistration",
        authenticate,
    )
    monkeypatch.setattr(winner_prepare.os, "replace", replace)
    report = winner_prepare.prepare_winner_publication(
        output_dir=output,
        tournament_preregistration=inputs[0],
        tournament_run_report=inputs[1],
        winner_selection=inputs[2],
        dtype="float16",
        parent_device_map="cuda:0",
        target_device_map="cuda:1",
    )

    assert replacements and replacements[-1][1] == output
    assert replacements[-1][0].parent == output.parent
    assert replacements[-1][0].name.startswith(f".{output.name}.")
    assert report["preregistration"] == str(output / "preregistration.json")
    assert json.loads((output / "build_report.json").read_text())["preregistration"] == str(
        output / "preregistration.json"
    )
    assert authenticated_paths[-1] == output / "preregistration.json"
    assert not list(tmp_path.glob(f".{output.name}.*.staging"))


def test_builder_never_deletes_final_tree_after_atomic_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "publication"
    inputs = [tmp_path / f"source-{index}.json" for index in range(3)]
    for path in inputs:
        path.write_text("{}\n", encoding="utf-8")
    portable = _digest("portable-final")
    payload = {
        "schema": publication.PREREGISTRATION_SCHEMA,
        "semantic_contract_sha256": _digest("semantic-final"),
        "winner": {"portable_factor_state_sha256": portable},
    }
    authentications = 0

    def authenticate(_path: Path, _sha256: str) -> Any:
        nonlocal authentications
        authentications += 1
        if authentications == 2:
            raise publication.GuidedWinnerPublicationError("final reauth failed")
        return SimpleNamespace(sealed_marker=output / "sealed_consumption.json")

    monkeypatch.setattr(
        winner_prepare.publication,
        "build_winner_publication_preregistration_payload",
        lambda **_kw: (payload, object(), payload["winner"], {}),
    )
    monkeypatch.setattr(
        winner_prepare.publication,
        "authenticate_winner_publication_preregistration",
        authenticate,
    )
    with pytest.raises(
        publication.GuidedWinnerPublicationError, match="final reauth failed"
    ):
        winner_prepare.prepare_winner_publication(
            output_dir=output,
            tournament_preregistration=inputs[0],
            tournament_run_report=inputs[1],
            winner_selection=inputs[2],
            dtype="float16",
            parent_device_map="cuda:0",
            target_device_map="cuda:1",
        )
    assert output.is_dir()
    assert (output / "preregistration.json").is_file()
    assert (output / "build_report.json").is_file()
    assert not list(tmp_path.glob(f".{output.name}.*.staging"))


def test_winner_publication_is_candidate_first_and_sealed_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    events: list[str] = []
    runtime = _FakeRuntime(events)
    report = _execute(contract, tmp_path / "run", runtime)

    assert report["status"] == "accepted"
    assert report["sealed_forward_count"] == 1
    assert report["post_bake_sealed_forward_count"] == 0
    assert runtime.sealed_forward_count == 1
    assert events.index("evaluate-candidate") < events.index("evaluate-sealed")
    assert events.count("evaluate-sealed") == 1
    assert events[-4:] == [
        "evaluate-fit",
        "evaluate-candidate",
        "save-reload",
        "independent-reload",
    ]
    assert report["post_bake_behavior_decisions"]["sealed"] is None
    assert report["cuda_preflight"]["models_loaded"] is False
    assert report["cuda_preflight"]["model_forward_calls"] == 0
    assert (tmp_path / "sealed_consumption.json").is_file()
    assert (tmp_path / "run" / "accepted_child").is_dir()


def test_consumed_marker_blocks_rerun_before_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    runtime = _FakeRuntime([])
    _execute(contract, tmp_path / "run", runtime)
    with pytest.raises(
        publication.GuidedWinnerPublicationError, match="already consumed"
    ):
        _execute(contract, tmp_path / "rerun", runtime)
    assert runtime.load_count == 1
    assert not (tmp_path / "rerun").exists()


@pytest.mark.parametrize(
    "reason",
    (
        "CUDA visibility differs from preregistration",
        "Logical CUDA 0 has the wrong device name",
        "Logical CUDA 1 has the wrong compute capability",
    ),
)
def test_cuda_preflight_drift_fails_before_output_or_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
    reason: str,
) -> None:
    del execution_patches
    contract = _contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )

    def reject(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise publication.tournament.CandidateTournamentError(reason)

    monkeypatch.setattr(
        publication.tournament,
        "validate_candidate_tournament_cuda_preflight",
        reject,
    )
    runtime = _FakeRuntime([])
    output = tmp_path / "run"
    with pytest.raises(publication.tournament.CandidateTournamentError, match=reason):
        _execute(contract, output, runtime)
    assert runtime.load_count == 0
    assert not output.exists()
    assert not contract.sealed_marker.exists()


@pytest.mark.parametrize("failure_point", ("runtime", "base"))
def test_runtime_or_base_setup_failure_leaves_no_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
    failure_point: str,
) -> None:
    del execution_patches
    contract = _contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    runtime: Any | None
    if failure_point == "runtime":
        monkeypatch.setattr(
            publication,
            "ProductionWinnerPublicationRuntime",
            lambda **_kw: (_ for _ in ()).throw(RuntimeError("runtime setup failed")),
        )
        runtime = None
    else:
        monkeypatch.setattr(
            publication.dataclasses,
            "replace",
            lambda *_a, **_kw: (_ for _ in ()).throw(
                RuntimeError("base setup failed")
            ),
        )
        runtime = _FakeRuntime([])
    output = tmp_path / "setup-failure-run"
    with pytest.raises(RuntimeError, match=f"{failure_point} setup failed"):
        _execute(contract, output, runtime)
    assert not output.exists()
    assert not contract.sealed_marker.exists()


def test_cli_preflight_records_public_tournament_cuda_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    assert (
        publication.main(
            [
                "--preregistration",
                str(contract.path),
                "--preregistration-sha256",
                contract.file_sha256,
                "--preflight",
                "--dtype",
                "float16",
                "--parent-device-map",
                "cuda:0",
                "--target-device-map",
                "cuda:1",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["cuda_preflight"]["status"] == "passed"
    assert report["cuda_preflight"]["models_loaded"] is False
    assert report["models_loaded"] is False
    assert report["sealed_forward_count"] == 0


def test_frontier_authenticates_and_retains_new_schema_without_legacy_relaxation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    output = tmp_path / "run"
    report = _execute(contract, output, _FakeRuntime([]))
    report_path = output / "run_report.json"

    candidate = authenticate_terminal_candidate(
        output,
        expected_run_report_sha256=file_sha256(report_path),
        expected_preregistration_sha256=contract.file_sha256,
        preregistration_authenticator=lambda *_a: {
            "semantic_contract_sha256": contract.raw["semantic_contract_sha256"]
        },
        accepted_child_authenticator=lambda *_a: report[
            "independent_reload_certificate"
        ],
    )
    assert candidate.terminal_run_schema == publication.RUN_SCHEMA
    assert candidate.standalone_compaction_passed is True
    retained = ExactCandidateFrontier(tmp_path / "frontier").register(candidate)
    assert retained["candidate_retained"] is True
    assert (
        ExactCandidateFrontier(tmp_path / "frontier").inspect()["retained_record_count"]
        == 1
    )


def test_frontier_rejects_post_bake_sealed_reuse_and_marker_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    output = tmp_path / "run"
    report = _execute(contract, output, _FakeRuntime([]))
    report_path = output / "run_report.json"

    tampered = dict(report)
    tampered["post_bake_behavior_decisions"] = dict(
        report["post_bake_behavior_decisions"]
    )
    tampered["post_bake_behavior_decisions"]["sealed"] = {
        "passed": True,
        "metrics": _metrics(split="heldout-evaluation"),
    }
    report_path.write_text(json.dumps(tampered, sort_keys=True) + "\n")
    with pytest.raises(ExactCandidateFrontierError, match="one-shot sealed policy"):
        authenticate_terminal_candidate(
            output,
            expected_run_report_sha256=file_sha256(report_path),
            expected_preregistration_sha256=contract.file_sha256,
            preregistration_authenticator=lambda *_a: {
                "semantic_contract_sha256": contract.raw["semantic_contract_sha256"]
            },
            accepted_child_authenticator=lambda *_a: report[
                "independent_reload_certificate"
            ],
        )

    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    contract.sealed_marker.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ExactCandidateFrontierError, match="marker bytes"):
        authenticate_terminal_candidate(
            output,
            expected_run_report_sha256=file_sha256(report_path),
            expected_preregistration_sha256=contract.file_sha256,
            preregistration_authenticator=lambda *_a: {
                "semantic_contract_sha256": contract.raw["semantic_contract_sha256"]
            },
            accepted_child_authenticator=lambda *_a: report[
                "independent_reload_certificate"
            ],
        )


@pytest.mark.parametrize(
    "tamper",
    ("forged-passed", "pre-metrics", "method-step", "method-lcp"),
)
def test_frontier_rejects_forged_behavior_and_method_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
    tamper: str,
) -> None:
    del execution_patches
    contract = _contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    output = tmp_path / "run"
    report = _execute(contract, output, _FakeRuntime([]))
    tampered = json.loads(json.dumps(report))

    def rehash(decision: dict[str, Any]) -> None:
        decision["decision_sha256"] = publication.canonical_sha256(
            {
                name: value
                for name, value in decision.items()
                if name != "decision_sha256"
            }
        )

    if tamper == "forged-passed":
        decision = tampered["pre_bake_behavior_decisions"]["sealed"]
        decision["metrics"]["action_accuracy"] = 0.0
        decision["metrics"]["action_agreement_count"] = 0
        rehash(decision)
    elif tamper == "pre-metrics":
        decision = tampered["pre_bake_behavior_decisions"]["candidate"]
        decision["metrics"]["parent_action_mean_logprob"] = -9.0
        rehash(decision)
        tampered["pre_sealed_candidate_decision"] = decision
    elif tamper == "method-step":
        decision = tampered["guided_method_decision"]
        decision["winner_step"] = 99
        rehash(decision)
    else:
        decision = tampered["post_bake_guided_method_decision"]
        decision["candidate_lcp_at_least_4_count"] = 99
        rehash(decision)

    report_path = output / "run_report.json"
    report_path.write_text(json.dumps(tampered, sort_keys=True) + "\n")
    with pytest.raises(ExactCandidateFrontierError, match="behavior gates"):
        authenticate_terminal_candidate(
            output,
            expected_run_report_sha256=file_sha256(report_path),
            expected_preregistration_sha256=contract.file_sha256,
            preregistration_authenticator=lambda *_a: {
                "semantic_contract_sha256": contract.raw[
                    "semantic_contract_sha256"
                ]
            },
            accepted_child_authenticator=lambda *_a: report[
                "independent_reload_certificate"
            ],
        )


def test_candidate_mismatch_rejects_without_consuming_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    events: list[str] = []
    runtime = _FakeRuntime(events, candidate_matches=False)
    with pytest.raises(
        publication.GuidedWinnerPublicationError,
        match="differs from tournament evidence",
    ):
        _execute(contract, tmp_path / "run", runtime)
    assert runtime.sealed_forward_count == 0
    assert not contract.sealed_marker.exists()
    assert "rollback-guided" in events
    assert "restore-prepublication-native" in events
    assert not (tmp_path / "run" / "accepted_child").exists()


def test_sealed_rejection_burns_marker_and_rolls_back_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    events: list[str] = []
    runtime = _FakeRuntime(events, sealed_passes=False)
    report = _execute(contract, tmp_path / "run", runtime)
    assert report["status"] == "rejected"
    assert report["reason"] == "one-shot-sealed-gate-rejected"
    assert report["rollback_exact"] is True
    assert runtime.sealed_forward_count == 1
    assert contract.sealed_marker.is_file()
    assert events[-2:] == ["rollback-guided", "rollback-base"]
    assert not (tmp_path / "run" / "accepted_child").exists()


def test_bake_failure_burns_marker_restores_native_and_leaves_no_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    events: list[str] = []
    runtime = _FakeRuntime(events, fail_bake=True)
    with pytest.raises(RuntimeError, match="synthetic bake failure"):
        _execute(contract, tmp_path / "run", runtime)
    assert runtime.sealed_forward_count == 1
    assert contract.sealed_marker.is_file()
    assert "restore-prepublication-native" in events
    assert not (tmp_path / "run" / "accepted_child").exists()
    failure = json.loads((tmp_path / "run" / "run_report.json").read_text())
    assert failure["status"] == "failed"
    assert failure["sealed_marker_consumed"] is True
