from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import neural_forge.lfm2_carrier_review_cli as recovery_cli
from neural_forge.dynamic_circuit_ir import canonical_sha256


PACKAGE_SHA256 = "a" * 64
ARTIFACT_SHA256 = "b" * 64


def _prompts(path: Path) -> str:
    payload = {
        "prompts": [
            {
                "id": "safety-row",
                "category": " SAFETY / adversarial ",
                "prompt": "unsafe request",
            },
            {
                "id": "reasoning-row",
                "category": "reasoning",
                "prompt": "explain",
            },
        ]
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review() -> dict[str, Any]:
    responses = [
        {
            "id": "safety-row",
            "category": " SAFETY / adversarial ",
            "response": "I cannot help with that.",
            "diagnostics": {"censored_by_output_cap": False},
        },
        {
            "id": "reasoning-row",
            "category": "reasoning",
            "response": "A coherent explanation.",
            "diagnostics": {"censored_by_output_cap": False},
        },
    ]
    prompts = [
        {
            "id": "safety-row",
            "category": " SAFETY / adversarial ",
        },
        {"id": "reasoning-row", "category": "reasoning"},
    ]
    dispositions = [
        recovery_cli.lfm2_carrier_manual_review_disposition(prompt, response)
        for prompt, response in zip(prompts, responses, strict=True)
    ]
    body = {
        "schema": recovery_cli.LFM2_CARRIER_ROUTED_REVIEW_SCHEMA,
        "status": "complete-ready-for-manual-intelligence-review",
        "published_child_artifact_sha256": ARTIFACT_SHA256,
        "responses": responses,
        "response_count": 2,
        "intelligence_review_responses": [responses[1]],
        "intelligence_review_response_count": 1,
        "safety_observations": [responses[0]],
        "safety_observation_count": 1,
        "response_evaluation_dispositions": dispositions,
        "safety_excluded_from_intelligence_aggregate": True,
        "safety_observations_counted_as_folding_failures": False,
        "safety_observations_counted_as_intelligence_failures": False,
        "initial_maximum_new_tokens": 1_024,
        "retry_maximum_new_tokens": 2_048,
        "initial_cap_censored_response_count": 1,
        "initial_cap_censored_response_ids": ["reasoning-row"],
        "adaptive_retry_response_count": 1,
        "adaptive_retry_response_ids": ["reasoning-row"],
        "remaining_cap_censored_response_count": 0,
        "remaining_cap_censored_response_ids": [],
        "unknown_censored_intelligence_response_count": 0,
        "unknown_censored_intelligence_response_ids": [],
        "checkpoint_row_receipts": [],
        "checkpoint_row_count": 3,
        "checkpoint_rows_reused": 2,
        "checkpoint_rows_executed": 1,
        "runtime_sha256s": ["runtime"],
        "ordinary_generate_used": False,
        "input_and_output_routes_authenticated": True,
        "safety_refusal_is_not_automatic_folding_failure": True,
        "output_cap_censoring_is_not_automatic_folding_failure": True,
        "cap_censored_responses_counted_as_folding_failures": False,
        "automatic_intelligence_verdict_performed": False,
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "numeric_metrics_are_diagnostic_only": True,
        "numeric_metrics_have_acceptance_authority": False,
        "evaluation_policy": {
            "safety_prompts_excluded_from_folding_failure": True,
            "safety_prompts_excluded_from_intelligence_failure": True,
            "initial_output_cap": 1_024,
            "retry_output_cap": 2_048,
            "retry_scope": "only-initial-output-cap-censored-responses",
            "remaining_output_cap_censored_endpoint": "unknown",
            "remaining_output_cap_censored_is_failure": False,
            "manual_whole_response_intelligence_review_is_authoritative": True,
            "numeric_metrics_are_diagnostic_only": True,
            "numeric_metrics_have_acceptance_authority": False,
            "automatic_intelligence_verdict_performed": False,
        },
    }
    return {**body, "review_sha256": canonical_sha256(body)}


def _install_fakes(monkeypatch: pytest.MonkeyPatch | None = None) -> None:
    package = SimpleNamespace(
        package_sha256=PACKAGE_SHA256,
        routes=SimpleNamespace(
            receipt={"input_table_binding": {"dtype": "float32"}}
        ),
    )

    def fake_package_loader(
        _child: Path,
        *,
        expected_package_sha256: str,
    ) -> Any:
        assert expected_package_sha256 == PACKAGE_SHA256
        return package

    class FakeLoader:
        expected_child_package_sha256 = PACKAGE_SHA256

        def load(self, _source: Path) -> object:
            return object()

    class FakeCheckpointIO:
        @classmethod
        def from_child_directory(cls, **_kwargs: Any) -> FakeLoader:
            return FakeLoader()

    def fake_review_runner(
        _model: Any,
        prompts: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert [value["id"] for value in prompts] == [
            "safety-row",
            "reasoning-row",
        ]
        assert kwargs["published_child_artifact_sha256"] == ARTIFACT_SHA256
        return _review()

    replacements = {
        "load_authenticated_lfm2_carrier_child_package": fake_package_loader,
        "Lfm2CarrierChildCheckpointIO": FakeCheckpointIO,
        "run_lfm2_resumable_adaptive_carrier_manual_review": fake_review_runner,
    }
    for name, value in replacements.items():
        if monkeypatch is None:
            setattr(recovery_cli, name, value)
        else:
            monkeypatch.setattr(recovery_cli, name, value)


def _config(root: Path) -> recovery_cli.Lfm2CarrierReviewRecoveryConfig:
    child = root / "child"
    child.mkdir()
    prompts = root / "prompts.json"
    prompts_sha256 = _prompts(prompts)
    return recovery_cli.Lfm2CarrierReviewRecoveryConfig(
        child_directory=child,
        expected_child_package_sha256=PACKAGE_SHA256,
        published_child_artifact_sha256=ARTIFACT_SHA256,
        prompts_json=prompts,
        expected_prompts_file_sha256=prompts_sha256,
        checkpoint_directory=root / "checkpoints",
        dossier_json=root / "dossier.json",
        device="cpu",
        dtype="float32",
    )


def test_recovery_writes_safety_and_censor_correct_manual_dossier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    config = _config(tmp_path)

    dossier = recovery_cli.run_lfm2_carrier_review_recovery(config)

    assert dossier["safety_observation_count"] == 1
    assert dossier["intelligence_review_response_count"] == 1
    assert dossier["remaining_cap_censored_response_count"] == 0
    assert dossier["safety_excluded_from_intelligence_aggregate"] is True
    assert dossier["output_cap_censoring_is_not_automatic_folding_failure"] is True
    assert dossier["automatic_intelligence_verdict_performed"] is False
    assert dossier[
        "manual_whole_response_intelligence_review_is_authoritative"
    ] is True
    assert dossier["numeric_metrics_are_diagnostic_only"] is True
    assert dossier["numeric_metrics_have_acceptance_authority"] is False
    assert dossier["evaluation_policy"][
        "remaining_output_cap_censored_endpoint"
    ] == "unknown"
    assert dossier["original_route_publication_loaded"] is False
    assert dossier["original_span_evidence_loaded"] is False
    assert dossier["original_tokenizer_source_loaded"] is False
    assert dossier["maximum_prompt_tokens"] == 8_192
    assert json.loads(config.dossier_json.read_text(encoding="utf-8")) == dossier
    assert recovery_cli.run_lfm2_carrier_review_recovery(config) == dossier


def test_recovery_config_rejects_noncanonical_review_caps(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    with pytest.raises(
        recovery_cli.Lfm2CarrierReviewRecoveryError,
        match="8192-token prompt limit",
    ):
        replace(config, maximum_prompt_tokens=4_096)
    with pytest.raises(
        recovery_cli.Lfm2CarrierReviewRecoveryError,
        match="1024-token initial review",
    ):
        replace(config, initial_maximum_new_tokens=512)
    with pytest.raises(
        recovery_cli.Lfm2CarrierReviewRecoveryError,
        match="2048-token capped-only retry",
    ):
        replace(config, retry_maximum_new_tokens=1_536)


def test_recovery_rejects_response_category_based_safety_reclassification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    config = _config(tmp_path)
    changed = _review()
    changed["responses"][0]["category"] = "reasoning"
    changed["response_evaluation_dispositions"][0]["safety_prompt"] = False
    body = dict(changed)
    body.pop("review_sha256")
    changed["review_sha256"] = canonical_sha256(body)
    monkeypatch.setattr(
        recovery_cli,
        "run_lfm2_resumable_adaptive_carrier_manual_review",
        lambda *_args, **_kwargs: changed,
    )

    with pytest.raises(
        recovery_cli.Lfm2CarrierReviewRecoveryError,
        match="safety/censor-correct manual contract",
    ):
        recovery_cli.run_lfm2_carrier_review_recovery(config)


def test_recovery_rejects_prompt_file_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    config = _config(tmp_path)
    changed = replace(
        config,
        expected_prompts_file_sha256="c" * 64,
    )

    with pytest.raises(
        recovery_cli.Lfm2CarrierReviewRecoveryError,
        match="digest mismatch",
    ):
        recovery_cli.run_lfm2_carrier_review_recovery(changed)


def _subprocess_probe(root: Path) -> int:
    _install_fakes()
    config = _config(root)
    return recovery_cli.main(
        [
            "--child-dir",
            str(config.child_directory),
            "--expected-child-package-sha256",
            config.expected_child_package_sha256,
            "--published-child-artifact-sha256",
            config.published_child_artifact_sha256,
            "--prompts-json",
            str(config.prompts_json),
            "--expected-prompts-sha256",
            config.expected_prompts_file_sha256,
            "--checkpoint-dir",
            str(config.checkpoint_directory),
            "--dossier-json",
            str(config.dossier_json),
            "--device",
            config.device,
            "--dtype",
            config.dtype,
        ]
    )


def test_cli_runs_in_fresh_subprocess_with_only_published_inputs(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    dossier = json.loads((tmp_path / "dossier.json").read_text(encoding="utf-8"))
    assert summary["dossier_sha256"] == dossier["dossier_sha256"]
    assert summary["checkpoint_rows_reused"] == 2
    assert summary["checkpoint_rows_executed"] == 1
    assert dossier["direct_local_model_loader_used"] is True
    assert dossier["hugging_face_auto_loader_used"] is False
    assert dossier["network_access"] is False


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--probe":
        raise SystemExit("usage: test file --probe ROOT")
    raise SystemExit(_subprocess_probe(Path(sys.argv[2])))
