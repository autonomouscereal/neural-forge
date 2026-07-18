from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from neural_forge.dynamic_circuit_ir import (
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.dynamic_lfm2_cpcf_production_cli import (
    Lfm2CpcfProductionCliConfig,
    Lfm2CpcfProductionCliError,
    ReviewedArtifactPin,
    run_lfm2_cpcf_production_cli,
)
from neural_forge.dynamic_search import (
    DynamicFoldSearchAttempt,
    DynamicFoldSearchCandidate,
    DynamicFoldSearchResult,
)
from neural_forge.dynamic_target_synthesis import DynamicTargetRequest

import neural_forge.dynamic_lfm2_cpcf_production_cli as cli_module


torch = pytest.importorskip("torch")


_ARTIFACT_NAMES = (
    "manifest",
    "parent_envelope",
    "cpcf_run_receipt",
    "candidate_outputs",
    "capture_receipt",
    "automatic_report",
    "manual_packet",
    "parent_relative_manual_packet",
    "parent_relative_ratings",
    "parent_relative_report",
    "composite_promotion_report",
)

_DIGEST_NAMES = (
    "expected_full_reviewed_evidence_fingerprint",
    "expected_cpcf_run_receipt_sha256",
    "expected_cpcf_run_receipt_document_sha256",
    "expected_bridge_artifact_sha256",
    "expected_cache_key",
    "expected_cache_manifest_sha256",
    "expected_trajectory_corpus_sha256",
    "expected_parent_graph_fingerprint",
)


def _write_canonical(path: Path, value: dict[str, Any]) -> str:
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    return canonical_sha256(value)


def _config(
    tmp_path: Path,
    *,
    receipt_path: Path | None = None,
) -> Lfm2CpcfProductionCliConfig:
    directories = {}
    for name in (
        "parent",
        "donor-tokenizer",
        "target-tokenizer",
        "bridge",
        "cache",
    ):
        directories[name] = tmp_path / name
        directories[name].mkdir()
    trajectory = tmp_path / "trajectories.json"
    _write_canonical(trajectory, {"rows": ["sealed"]})
    pins = {}
    for ordinal, name in enumerate(_ARTIFACT_NAMES):
        path = tmp_path / f"{name}.json"
        document = {"name": name, "ordinal": ordinal}
        pins[name] = ReviewedArtifactPin(
            path=path,
            document_sha256=_write_canonical(path, document),
        )
    digests = {
        name: canonical_sha256({"pin": name}) for name in _DIGEST_NAMES
    }
    return Lfm2CpcfProductionCliConfig(
        parent_model_directory=directories["parent"],
        donor_tokenizer_directory=directories["donor-tokenizer"],
        target_tokenizer_directory=directories["target-tokenizer"],
        bridge_directory=directories["bridge"],
        trace_cache_directory=directories["cache"],
        trajectory_corpus=trajectory,
        reviewed_artifact_pins=pins,
        **digests,
        expected_exact_target_unique_parameters=2_291_840,
        destination_root=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=receipt_path or tmp_path / "receipt.json",
        device="cpu",
        dtype="float32",
        parent_name="parent",
        child_name="small-child",
        blank_seed=0,
        minimum_hidden_size=32,
    )


class _Plan:
    def __init__(self, target_request: DynamicTargetRequest) -> None:
        self.target_request = target_request
        self.fingerprint = canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "exact-reviewed-plan",
            "target_request": self.target_request.as_dict(),
        }


class _Artifact:
    def __init__(self, path: Path, exact_parameters: int) -> None:
        self.path = str(path)
        self.exact_unique_parameters = exact_parameters
        self.certificate_sha256 = canonical_sha256(self.body())

    def body(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exact_unique_parameters": self.exact_unique_parameters,
            "folding_only": True,
            "training_used": False,
            "pruning_used": False,
            "quantization_used": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.body(),
            "certificate_sha256": self.certificate_sha256,
        }


class _Selected:
    complete = True

    def __init__(
        self,
        request: DynamicTargetRequest,
        artifact: _Artifact,
    ) -> None:
        self.target = SimpleNamespace(
            request=request,
            resource_profile=SimpleNamespace(
                unique_parameters=request.exact_unique_parameters
            ),
        )
        self.published = SimpleNamespace(artifact=artifact)

    def as_dict(self) -> dict[str, Any]:
        return {
            "complete": True,
            "target_request": self.target.request.as_dict(),
            "artifact": self.published.artifact.as_dict(),
        }


class _Facade:
    fingerprint = canonical_sha256({"facade": "one-entry"})

    def __init__(self, plan: _Plan, calls: dict[str, Any]) -> None:
        self.plan = plan
        self.calls = calls

    def run(self, **kwargs: Any) -> DynamicFoldSearchResult:
        self.calls["facade_run"] = kwargs
        objective = kwargs["objective"]
        destination = Path(kwargs["destination_root"]) / "bridged-cpcf"
        destination.mkdir(parents=True)
        candidate_target = DynamicTargetRequest(
            child_name=f"{objective.target.child_name}--bridged-cpcf",
            target_family=objective.target.target_family,
            target_modalities=objective.target.target_modalities,
            exact_unique_parameters=objective.target.exact_unique_parameters,
            tokenizer_fingerprint=objective.target.tokenizer_fingerprint,
            adapter_id="neural_forge.target.lfm2",
            fold_method_id="lfm2_dense_budget_residual_attention_model",
            blank_seed=objective.target.blank_seed,
            configuration=objective.target.configuration,
        )
        candidate = DynamicFoldSearchCandidate(
            candidate_id="bridged-cpcf",
            target=candidate_target,
            evidence_fingerprint="e" * 64,
            metadata=(
                ("target_plan_fingerprint", self.plan.fingerprint),
            ),
        )
        artifact = _Artifact(
            destination,
            candidate_target.exact_unique_parameters,
        )
        child = destination / "child"
        child.mkdir()
        (child / "model.bin").write_bytes(b"owned-child-state")
        (destination / "certificate.json").write_bytes(
            canonical_json_bytes(artifact.as_dict()) + b"\n"
        )
        selected = _Selected(
            candidate_target,
            artifact,
        )
        attempt = DynamicFoldSearchAttempt(
            candidate=candidate,
            destination=str(destination),
            status="accepted",
            score=1.0,
            result_fingerprint=canonical_sha256(selected.as_dict()),
            artifact_retained=True,
        )
        return DynamicFoldSearchResult(
            objective=objective,
            portfolio_id=objective.portfolio_id,
            portfolio_version=1,
            attempts=(attempt,),
            selected_candidate_id="bridged-cpcf",
            selected_destination=str(destination),
            selected_result=selected,
        )


def _install_success_fakes(
    monkeypatch: pytest.MonkeyPatch,
    config: Lfm2CpcfProductionCliConfig,
) -> tuple[dict[str, Any], Any]:
    calls: dict[str, Any] = {}
    request = DynamicTargetRequest(
        child_name=config.child_name,
        target_family="lfm2",
        target_modalities=("text",),
        exact_unique_parameters=config.expected_exact_target_unique_parameters,
        tokenizer_fingerprint="a" * 64,
        blank_seed=config.blank_seed,
        configuration=(("source_parent", config.parent_name),),
    )
    plan = _Plan(request)
    bundle = SimpleNamespace(
        candidate_specs=(SimpleNamespace(candidate_id="bridged-cpcf"),),
        portfolio_id="neural_forge.production.lfm2-cpcf-exact",
    )
    wiring = SimpleNamespace(
        target_plan=plan,
        production_bundle=bundle,
        parent_model=object(),
        tokenizer_bridge=SimpleNamespace(
            donor_tokenizer_semantics_sha256="d" * 64
        ),
        parent_graph=SimpleNamespace(unique_parameter_count=229_184_000),
        as_dict=lambda: {
            "schema": "test-wiring",
            "target_plan_fingerprint": plan.fingerprint,
        },
    )

    def fake_factory(**kwargs: Any) -> Any:
        calls["factory"] = kwargs
        return wiring

    class FakeCatalog:
        def __init__(self, bundles: Any, **kwargs: Any) -> None:
            calls["catalog"] = {
                "bundles": bundles,
                **kwargs,
            }
            Path(kwargs["ledger_path"]).write_bytes(b"")

        def assemble(self) -> _Facade:
            return _Facade(plan, calls)

    def authenticate_fake_artifact(destination: Any) -> _Artifact:
        root = Path(destination).resolve(strict=True)
        certificate_path = root / "certificate.json"
        child_payload = root / "child" / "model.bin"
        try:
            document = json.loads(
                certificate_path.read_bytes().decode("ascii")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise cli_module.DynamicFoldArtifactError(
                "Fake artifact certificate is unreadable."
            ) from exc
        artifact = _Artifact(
            root,
            config.expected_exact_target_unique_parameters,
        )
        if (
            document != artifact.as_dict()
            or not child_payload.is_file()
            or child_payload.is_symlink()
            or child_payload.read_bytes() != b"owned-child-state"
        ):
            raise cli_module.DynamicFoldArtifactError(
                "Fake standalone child tree differs from its certificate."
            )
        return artifact

    def remove_fake_artifact(
        destination: Any,
        *,
        expected_parent: Any | None = None,
        expected_name: str | None = None,
    ) -> _Artifact:
        artifact = authenticate_fake_artifact(destination)
        root = Path(destination).resolve(strict=True)
        if (
            expected_parent is not None
            and root.parent != Path(expected_parent).resolve(strict=True)
        ):
            raise cli_module.DynamicFoldArtifactError(
                "Fake artifact has the wrong expected parent."
            )
        if expected_name is not None and root.name != expected_name:
            raise cli_module.DynamicFoldArtifactError(
                "Fake artifact has the wrong expected name."
            )
        shutil.rmtree(root)
        return artifact

    monkeypatch.setattr(
        cli_module,
        "build_lfm2_cpcf_production_resources",
        fake_factory,
    )
    monkeypatch.setattr(
        cli_module,
        "ProductionDynamicFoldCatalog",
        FakeCatalog,
    )
    monkeypatch.setattr(
        cli_module,
        "load_dynamic_fold_artifact_certificate",
        authenticate_fake_artifact,
    )
    monkeypatch.setattr(
        cli_module,
        "remove_authenticated_dynamic_fold_artifact",
        remove_fake_artifact,
    )
    return calls, wiring


def test_cli_forwards_all_pins_and_runs_only_the_reviewed_exact_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls, wiring = _install_success_fakes(monkeypatch, config)

    summary = run_lfm2_cpcf_production_cli(config)

    factory = calls["factory"]
    for name in _DIGEST_NAMES:
        assert factory[name] == getattr(config, name)
    assert factory["expected_exact_target_unique_parameters"] == 2_291_840
    assert set(factory["reviewed_artifacts"].__dataclass_fields__) == set(
        _ARTIFACT_NAMES
    )
    for name in _ARTIFACT_NAMES:
        observed = getattr(factory["reviewed_artifacts"], name)
        assert observed.source == config.reviewed_artifact_pins[name].path
        assert observed.document_sha256 == (
            config.reviewed_artifact_pins[name].document_sha256
        )
    assert factory["device"] == torch.device("cpu")
    assert factory["dtype"] == torch.float32
    assert calls["catalog"]["bundles"] == (wiring.production_bundle,)
    assert calls["catalog"]["ledger_path"] == config.ledger_path
    assert calls["catalog"]["require_positive_compaction"] is True
    facade_call = calls["facade_run"]
    objective = facade_call["objective"]
    assert objective.target is wiring.target_plan.target_request
    assert objective.maximum_attempts == 1
    assert objective.stop_after_first_accept is True
    assert objective.portfolio_id == wiring.production_bundle.portfolio_id
    assert facade_call["parent_models"] == {
        config.parent_name: wiring.parent_model
    }
    assert facade_call["parent_tokenizer_fingerprints"] == {
        config.parent_name: "d" * 64
    }

    receipt = json.loads(config.receipt_path.read_text(encoding="ascii"))
    body = dict(receipt)
    seal = body.pop("receipt_sha256")
    assert seal == canonical_sha256(body)
    assert receipt["status"] == "accepted"
    assert receipt["objective"] == objective.as_dict()
    assert receipt["target_plan"] == wiring.target_plan.as_dict()
    assert receipt["result"]["search_result"]["complete"] is True
    assert receipt["result"]["selected_artifact_certificate"][
        "exact_unique_parameters"
    ] == 2_291_840
    assert receipt["compaction_dimension"] == "unique_parameter_count"
    assert receipt["folding_only"] is True
    assert receipt["training_used"] is False
    assert receipt["pruning_used"] is False
    assert receipt["quantization_used"] is False
    assert receipt["network_allowed"] is False
    assert summary["status"] == "accepted"
    assert summary["exact_target_unique_parameters"] == 2_291_840
    assert summary["parent_unique_parameters"] == 229_184_000
    assert summary["parameter_compression_ratio"] == pytest.approx(100.0)
    assert summary["receipt_document_sha256"] == canonical_sha256(receipt)


def test_cli_writes_a_sealed_failure_receipt_without_an_acceptance_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    def fail_factory(**_kwargs: Any) -> Any:
        raise RuntimeError("independent cache pin mismatch")

    monkeypatch.setattr(
        cli_module,
        "build_lfm2_cpcf_production_resources",
        fail_factory,
    )

    with pytest.raises(
        Lfm2CpcfProductionCliError,
        match="failed closed",
    ):
        run_lfm2_cpcf_production_cli(config)

    receipt = json.loads(config.receipt_path.read_text(encoding="ascii"))
    body = dict(receipt)
    seal = body.pop("receipt_sha256")
    assert seal == canonical_sha256(body)
    assert receipt["status"] == "failed_closed"
    assert receipt["result"] is None
    assert receipt["failure"] == {
        "accepted_artifact_claimed": False,
        "artifact_recovery": None,
        "error_type": "RuntimeError",
        "message": "independent cache pin mismatch",
    }
    assert receipt["training_used"] is False
    assert receipt["pruning_used"] is False
    assert receipt["quantization_used"] is False
    assert receipt["network_allowed"] is False


def test_cli_refuses_bookkeeping_inside_the_candidate_artifact(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        Lfm2CpcfProductionCliError,
        match="receipt_path cannot be inside",
    ):
        _config(
            tmp_path,
            receipt_path=(
                tmp_path / "published" / "bridged-cpcf" / "receipt.json"
            ),
        )
    assert not (tmp_path / "published").exists()


def test_cli_refuses_ledger_that_overwrites_an_authenticated_input(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    with pytest.raises(
        Lfm2CpcfProductionCliError,
        match=(
            "ledger_path must be disjoint from every authenticated input "
            "file and directory"
        ),
    ):
        replace(config, ledger_path=config.trajectory_corpus)


def test_cli_refuses_ledger_hard_linked_to_an_authenticated_input(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    try:
        config.ledger_path.hardlink_to(config.trajectory_corpus)
    except OSError as exc:
        pytest.skip(f"local filesystem does not permit hard links: {exc}")

    with pytest.raises(
        Lfm2CpcfProductionCliError,
        match=(
            "ledger_path must be disjoint from every authenticated input "
            "file and directory"
        ),
    ):
        replace(config)


def test_cli_refuses_receipt_inside_an_authenticated_input_directory(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    with pytest.raises(
        Lfm2CpcfProductionCliError,
        match=(
            "receipt_path must be disjoint from every authenticated input "
            "file and directory"
        ),
    ):
        replace(
            config,
            receipt_path=config.parent_model_directory / "receipt.json",
        )


def test_cli_refuses_bookkeeping_that_contains_the_candidate_artifact(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    with pytest.raises(
        Lfm2CpcfProductionCliError,
        match="ledger_path must be disjoint from the certified child artifact",
    ):
        replace(config, ledger_path=config.destination_root)


def test_cli_rolls_back_publication_before_claiming_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _install_success_fakes(monkeypatch, config)

    def fail_after_publication(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise Lfm2CpcfProductionCliError(
            "post-publication authentication failed"
        )

    monkeypatch.setattr(
        cli_module,
        "_accepted_result_payload",
        fail_after_publication,
    )

    with pytest.raises(
        Lfm2CpcfProductionCliError,
        match="post-publication authentication failed",
    ):
        run_lfm2_cpcf_production_cli(config)

    candidate = config.destination_root / "bridged-cpcf"
    assert not candidate.exists()
    receipt = json.loads(config.receipt_path.read_text(encoding="ascii"))
    recovery = receipt["failure"]["artifact_recovery"]
    recovery_body = dict(recovery)
    recovery_seal = recovery_body.pop("recovery_sha256")
    assert recovery_seal == canonical_sha256(recovery_body)
    assert recovery["candidate_destination"] == str(candidate.resolve())
    assert recovery["publication_observed"] is True
    assert recovery["ownership_authenticated"] is True
    assert recovery["ownership_proof"]["ownership_proof_sha256"]
    assert recovery["cleanup_attempted"] is True
    assert recovery["cleanup_succeeded"] is True
    assert recovery["cleanup_not_attempted_reason"] is None
    assert recovery["artifact_retained"] is False
    assert recovery["recovery_required"] is False
    assert recovery["retained_artifact_observation"] is None
    assert receipt["failure"]["accepted_artifact_claimed"] is False


def test_cli_failure_receipt_reports_residual_artifact_if_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _install_success_fakes(monkeypatch, config)

    def fail_after_publication(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise Lfm2CpcfProductionCliError(
            "post-publication authentication failed"
        )

    def fail_cleanup(_path: Path, **_kwargs: Any) -> None:
        raise PermissionError("simulated cleanup denial")

    monkeypatch.setattr(
        cli_module,
        "_accepted_result_payload",
        fail_after_publication,
    )
    monkeypatch.setattr(
        cli_module,
        "remove_authenticated_dynamic_fold_artifact",
        fail_cleanup,
    )

    with pytest.raises(
        Lfm2CpcfProductionCliError,
        match="post-publication authentication failed",
    ):
        run_lfm2_cpcf_production_cli(config)

    candidate = config.destination_root / "bridged-cpcf"
    assert candidate.is_dir()
    receipt = json.loads(config.receipt_path.read_text(encoding="ascii"))
    recovery = receipt["failure"]["artifact_recovery"]
    assert recovery["publication_observed"] is True
    assert recovery["ownership_authenticated"] is True
    assert recovery["ownership_proof"]["ownership_proof_sha256"]
    assert recovery["cleanup_attempted"] is True
    assert recovery["cleanup_succeeded"] is False
    assert recovery["cleanup_not_attempted_reason"] is None
    assert recovery["artifact_retained"] is True
    assert recovery["recovery_required"] is True
    assert recovery["cleanup_error"] == {
        "error_type": "PermissionError",
        "message": "simulated cleanup denial",
    }
    observation = recovery["retained_artifact_observation"]
    assert observation["path"] == str(candidate.resolve())
    assert observation["is_directory"] is True
    assert observation["certificate_is_regular_file"] is True
    assert observation["certificate_file_sha256"]
    observation_body = dict(observation)
    observation_seal = observation_body.pop("observation_sha256")
    assert observation_seal == canonical_sha256(observation_body)
    assert receipt["failure"]["accepted_artifact_claimed"] is False


def test_cli_never_deletes_racing_unowned_candidate_without_returned_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _install_success_fakes(monkeypatch, config)

    def race_then_fail(_facade: _Facade, **kwargs: Any) -> Any:
        candidate = Path(kwargs["destination_root"]) / "bridged-cpcf"
        candidate.mkdir(parents=True)
        (candidate / "unowned-marker.txt").write_text(
            "created by a concurrent process",
            encoding="utf-8",
        )
        raise RuntimeError("facade lost destination race")

    monkeypatch.setattr(_Facade, "run", race_then_fail)

    with pytest.raises(
        Lfm2CpcfProductionCliError,
        match="facade lost destination race",
    ):
        run_lfm2_cpcf_production_cli(config)

    candidate = config.destination_root / "bridged-cpcf"
    assert (candidate / "unowned-marker.txt").read_text(
        encoding="utf-8"
    ) == "created by a concurrent process"
    receipt = json.loads(config.receipt_path.read_text(encoding="ascii"))
    recovery = receipt["failure"]["artifact_recovery"]
    assert recovery["publication_observed"] is True
    assert recovery["ownership_authenticated"] is False
    assert recovery["ownership_proof"] is None
    assert recovery["cleanup_attempted"] is False
    assert recovery["cleanup_succeeded"] is False
    assert recovery["artifact_retained"] is True
    assert recovery["recovery_required"] is True
    assert recovery["cleanup_not_attempted_reason"] == (
        "The current candidate path is not cryptographically bound to this "
        "invocation's returned result."
    )
    assert receipt["failure"]["accepted_artifact_claimed"] is False


def test_cli_preserves_owned_candidate_if_child_tree_changes_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _install_success_fakes(monkeypatch, config)

    def mutate_then_fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        candidate = config.destination_root / "bridged-cpcf"
        (candidate / "child" / "model.bin").write_bytes(
            b"replacement-child-state"
        )
        raise Lfm2CpcfProductionCliError(
            "post-publication authentication failed"
        )

    monkeypatch.setattr(
        cli_module,
        "_accepted_result_payload",
        mutate_then_fail,
    )

    with pytest.raises(
        Lfm2CpcfProductionCliError,
        match="post-publication authentication failed",
    ):
        run_lfm2_cpcf_production_cli(config)

    candidate = config.destination_root / "bridged-cpcf"
    assert (candidate / "certificate.json").is_file()
    assert (candidate / "child" / "model.bin").read_bytes() == (
        b"replacement-child-state"
    )
    receipt = json.loads(config.receipt_path.read_text(encoding="ascii"))
    recovery = receipt["failure"]["artifact_recovery"]
    assert recovery["publication_observed"] is True
    assert recovery["ownership_authenticated"] is False
    assert recovery["cleanup_attempted"] is False
    assert recovery["artifact_retained"] is True
    assert recovery["recovery_required"] is True


def test_cli_refuses_accepted_receipt_if_child_tree_changes_after_payload_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _install_success_fakes(monkeypatch, config)
    original_receipt_body = cli_module._receipt_body

    def mutate_before_receipt(**kwargs: Any) -> dict[str, Any]:
        body = original_receipt_body(**kwargs)
        if kwargs["status"] == "accepted":
            candidate = config.destination_root / "bridged-cpcf"
            (candidate / "child" / "model.bin").write_bytes(
                b"replacement-child-state"
            )
        return body

    monkeypatch.setattr(
        cli_module,
        "_receipt_body",
        mutate_before_receipt,
    )

    with pytest.raises(
        Lfm2CpcfProductionCliError,
        match=(
            "Candidate artifact tree changed before accepted receipt "
            "publication"
        ),
    ):
        run_lfm2_cpcf_production_cli(config)

    candidate = config.destination_root / "bridged-cpcf"
    assert (candidate / "certificate.json").is_file()
    assert (candidate / "child" / "model.bin").read_bytes() == (
        b"replacement-child-state"
    )
    receipt = json.loads(config.receipt_path.read_text(encoding="ascii"))
    assert receipt["status"] == "failed_closed"
    assert receipt["failure"]["accepted_artifact_claimed"] is False
    recovery = receipt["failure"]["artifact_recovery"]
    assert recovery["ownership_authenticated"] is False
    assert recovery["cleanup_attempted"] is False
    assert recovery["artifact_retained"] is True


def test_cli_preserves_artifact_when_committed_receipt_reports_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _install_success_fakes(monkeypatch, config)
    publish = cli_module._exclusive_atomic_json
    calls = 0

    def publish_then_interrupt(
        path: Path,
        document: dict[str, Any],
    ) -> tuple[str, str]:
        nonlocal calls
        calls += 1
        publish(path, document)
        raise RuntimeError("simulated interruption after receipt commit")

    monkeypatch.setattr(
        cli_module,
        "_exclusive_atomic_json",
        publish_then_interrupt,
    )

    with pytest.raises(
        Lfm2CpcfProductionCliError,
        match="simulated interruption after receipt commit",
    ):
        run_lfm2_cpcf_production_cli(config)

    assert calls == 1
    candidate = config.destination_root / "bridged-cpcf"
    assert candidate.is_dir()
    receipt = json.loads(config.receipt_path.read_text(encoding="ascii"))
    assert receipt["status"] == "accepted"
    assert receipt["result"]["selected_destination"] == str(
        candidate.resolve()
    )


def test_cli_source_has_no_remote_or_generic_model_loader_path() -> None:
    source = Path(cli_module.__file__).read_text(encoding="utf-8")
    assert "from_" + "pretrained" not in source
    assert "Auto" + "Model" not in source
    assert "requests." not in source
    assert "urllib." not in source
    assert "http" + "x." not in source
