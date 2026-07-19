from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from experiments import evaluate_lfm2_cpcf_v5_standalone as evaluator
from neural_forge.dynamic_circuit_ir import (
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.hf_lfm2_cpcf_model_fold import (
    independent_parameter_inventory,
)
from neural_forge.hf_lfm2_intelligence_gate import (
    bind_manual_ratings,
    load_lfm2_intelligence_holdout_manifest,
)
from neural_forge.hf_lfm2_intelligence_holdout import (
    seal_lfm2_intelligence_holdout_parent_envelope,
)


EOS_TOKEN_ID = 7


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, document: Any) -> None:
    path.write_bytes(canonical_json_bytes(document) + b"\n")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _TinyChild(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight


class _TinyTokenizer:
    eos_token_id = EOS_TOKEN_ID
    pad_token_id = EOS_TOKEN_ID


def _parent_envelope(tmp_path: Path) -> tuple[Path, str]:
    manifest = load_lfm2_intelligence_holdout_manifest()
    rows = [
        {
            "case_id": spec.case_id,
            "input_token_count": 8 + index,
            "output": (
                f"The cached parent gives a coherent response for {spec.case_id}."
            ),
            "output_token_ids": [1000 + index, EOS_TOKEN_ID],
        }
        for index, spec in enumerate(manifest.cases)
    ]
    document = seal_lfm2_intelligence_holdout_parent_envelope(
        manifest,
        rows,
        parent_state_fingerprint=_sha("parent-state"),
        parent_model_fingerprint=_sha("parent-model"),
        tokenizer_fingerprint=_sha("parent-tokenizer"),
        eos_token_ids=(EOS_TOKEN_ID,),
    )
    path = tmp_path / "parent-envelope.json"
    _write_json(path, document)
    return path, _file_sha(path)


def _capture_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    candidate_directory = tmp_path / "quarantine" / "candidate"
    candidate_directory.mkdir(parents=True)
    output_directory = tmp_path / "evaluation"
    parent_path, parent_file_sha = _parent_envelope(tmp_path)

    reference_model = _TinyChild()
    reference_model.requires_grad_(False)
    reference_model.eval()
    candidate_state = fingerprint_model_state(reference_model)
    inventory = dict(independent_parameter_inventory(reference_model))
    semantics_sha = _sha("target-tokenizer-semantics")
    package_sha = _sha("target-tokenizer-package")
    metadata = {
        "schema": evaluator.LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA,
        "artifact_sha256": _sha("candidate-artifact"),
        "context_binding_sha256": _sha("context-binding"),
        "model_state_sha256": candidate_state,
        "parameter_inventory_sha256": inventory["inventory_sha256"],
        "materialized_ffn_width": 2,
        "tokenizer_semantics_sha256": semantics_sha,
        "tokenizer_package_sha256": package_sha,
    }
    monkeypatch.setattr(
        evaluator,
        "_read_artifact_metadata",
        lambda _directory: dict(metadata),
    )

    loader_calls: list[dict[str, Any]] = []

    class _StrictLoader:
        def __init__(self, **kwargs: Any) -> None:
            loader_calls.append({"constructor": dict(kwargs)})

        def __call__(self, source: Path) -> _TinyChild:
            loader_calls.append({"source": Path(source)})
            model = _TinyChild()
            model.requires_grad_(False)
            model.eval()
            setattr(
                model,
                "_neural_forge_quarantine_tokenizer",
                _TinyTokenizer(),
            )
            setattr(
                model,
                "_neural_forge_quarantine_load_audit",
                {
                    "direct_local_json_and_safetensors": True,
                    "from_pretrained_used": False,
                    "hub_loader_used": False,
                    "network_access": False,
                    "model_state_sha256": candidate_state,
                },
            )
            return model

    generated_limits: list[int] = []
    generated_text: list[str] = []

    def _generate_once(
        model: _TinyChild,
        tokenizer: _TinyTokenizer,
        manifest: Any,
    ) -> tuple[dict[str, Any], ...]:
        rows = []
        for index, spec in enumerate(manifest.cases):
            model(torch.ones((1, 2)))
            text = f"Persisted decoded child response for {spec.case_id}."
            generated_limits.append(spec.generation_limit)
            generated_text.append(text)
            rows.append(
                {
                    "case_id": spec.case_id,
                    "input_token_count": 12 + index,
                    "output": text,
                    "output_token_ids": [
                        2000 + index,
                        tokenizer.eos_token_id,
                    ],
                    "generation_limit": spec.generation_limit,
                    "topical_relevance": None,
                }
            )
        return tuple(rows)

    monkeypatch.setattr(
        evaluator,
        "Lfm2CpcfV5StrictLocalLoader",
        _StrictLoader,
    )
    monkeypatch.setattr(
        evaluator,
        "generate_candidate_rows_once",
        _generate_once,
    )
    config = evaluator.CaptureConfig(
        candidate_directory=candidate_directory,
        parent_envelope=parent_path,
        expected_parent_envelope_file_sha256=parent_file_sha,
        expected_candidate_state_sha256=candidate_state,
        expected_target_tokenizer_semantics_sha256=semantics_sha,
        output_directory=output_directory,
        device="cpu",
        dtype="float32",
    )
    report = evaluator.capture_standalone_intelligence(config)
    return {
        "candidate_directory": candidate_directory,
        "output_directory": output_directory,
        "parent_path": parent_path,
        "parent_file_sha": parent_file_sha,
        "candidate_state": candidate_state,
        "semantics_sha": semantics_sha,
        "package_sha": package_sha,
        "config": config,
        "report": report,
        "loader_calls": loader_calls,
        "generated_limits": generated_limits,
        "generated_text": generated_text,
    }


def test_capture_persists_text_limits_and_strict_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _capture_fixture(tmp_path, monkeypatch)
    output_directory = fixture["output_directory"]
    child = json.loads((output_directory / evaluator.CHILD_OUTPUT_FILENAME).read_text())
    automatic = json.loads(
        (output_directory / evaluator.AUTOMATIC_REPORT_FILENAME).read_text()
    )
    packet = json.loads(
        (output_directory / evaluator.MANUAL_PACKET_FILENAME).read_text()
    )

    assert len(child["rows"]) == 64
    assert [row["output"] for row in child["rows"]] == fixture["generated_text"]
    assert any(row["generation_limit"] > 64 for row in child["rows"])
    assert max(fixture["generated_limits"]) == 96
    assert automatic["generation_protocol"]["global_64_token_cap_used"] is False
    assert automatic["generation_protocol"]["limits_above_64_preserved"] is True
    assert automatic["generation_protocol"]["child_generate_invocations"] == 64
    assert automatic["direct_local_reload"]["strict_v5_loader_used"] is True
    assert automatic["direct_local_reload"]["from_pretrained_used"] is False
    assert automatic["operation_audit"]["parent_forward_passes"] == 0
    assert automatic["operation_audit"]["training_steps"] == 0
    assert automatic["operation_audit"]["parameters_pruned"] == 0
    assert automatic["operation_audit"]["parameters_quantized"] == 0
    assert len(fixture["loader_calls"]) == 2
    assert fixture["loader_calls"][1]["source"] == fixture["candidate_directory"]
    assert packet["packet_sha256"] == automatic["manual_review_packet_sha256"]
    assert packet["rubric"]["exact_wording_comparison_permitted"] is False
    assert len(packet["cases"]) == 6

    with pytest.raises(
        evaluator.Lfm2CpcfV5StandaloneEvaluationError,
        match="already exist",
    ):
        evaluator.capture_standalone_intelligence(fixture["config"])
    assert len(fixture["loader_calls"]) == 2

    forbidden_output = fixture["candidate_directory"] / "evaluation"
    with pytest.raises(
        evaluator.Lfm2CpcfV5StandaloneEvaluationError,
        match="disjoint from the child",
    ):
        evaluator.capture_standalone_intelligence(
            replace(
                fixture["config"],
                output_directory=forbidden_output,
            )
        )
    assert not forbidden_output.exists()

    containing_output = fixture["candidate_directory"].parent
    with pytest.raises(
        evaluator.Lfm2CpcfV5StandaloneEvaluationError,
        match="disjoint from the child",
    ):
        evaluator.capture_standalone_intelligence(
            replace(
                fixture["config"],
                output_directory=containing_output,
            )
        )


def test_finalize_binds_manual_ratings_and_refuses_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _capture_fixture(tmp_path, monkeypatch)
    capture_directory = fixture["output_directory"]
    packet = json.loads(
        (capture_directory / evaluator.MANUAL_PACKET_FILENAME).read_text()
    )
    ratings = bind_manual_ratings(
        packet,
        [
            {
                "case_id": row["case_id"],
                "correctness": 4,
                "coherence": 4,
                "instruction_following": 4,
                "capability_retention": 4,
            }
            for row in packet["cases"]
        ],
        reviewer_id="manual-reviewer",
    ).as_dict()
    ratings_path = tmp_path / "manual-ratings-64.json"
    _write_json(ratings_path, ratings)
    ratings_file_sha = _file_sha(ratings_path)
    config = evaluator.FinalizeConfig(
        capture_directory=capture_directory,
        parent_envelope=fixture["parent_path"],
        expected_parent_envelope_file_sha256=fixture["parent_file_sha"],
        manual_ratings=ratings_path,
        expected_manual_ratings_file_sha256=ratings_file_sha,
        output_directory=capture_directory,
    )
    report = evaluator.finalize_standalone_intelligence(config)
    stored = json.loads(
        (capture_directory / evaluator.FINAL_REPORT_FILENAME).read_text()
    )
    assert stored == report
    assert report["manual_ratings_sha256"] == ratings["ratings_sha256"]
    assert report["reviewer_id"] == "manual-reviewer"
    assert report["final_gate_report"]["manual_review"]["status"] == "accepted"
    assert report["acceptance_semantics"]["exact_wording_matching_used"] is False
    assert report["operation_audit"]["parent_forward_passes"] == 0
    assert report["operation_audit"]["child_model_loaded"] is False

    with pytest.raises(
        evaluator.Lfm2CpcfV5StandaloneEvaluationError,
        match="already exists",
    ):
        evaluator.finalize_standalone_intelligence(config)

    forbidden_output = fixture["candidate_directory"] / "final-evaluation"
    with pytest.raises(
        evaluator.Lfm2CpcfV5StandaloneEvaluationError,
        match="disjoint from the child",
    ):
        evaluator.finalize_standalone_intelligence(
            replace(config, output_directory=forbidden_output)
        )
    assert not forbidden_output.exists()

    containing_output = fixture["candidate_directory"].parent
    with pytest.raises(
        evaluator.Lfm2CpcfV5StandaloneEvaluationError,
        match="disjoint from the child",
    ):
        evaluator.finalize_standalone_intelligence(
            replace(config, output_directory=containing_output)
        )

    tampered = dict(ratings)
    tampered["packet_sha256"] = _sha("different-packet")
    tampered_core = dict(tampered)
    tampered_core.pop("ratings_sha256")
    tampered["ratings_sha256"] = canonical_sha256(tampered_core)
    tampered_path = tmp_path / "manual-ratings-tampered.json"
    _write_json(tampered_path, tampered)
    bad_config = evaluator.FinalizeConfig(
        capture_directory=capture_directory,
        parent_envelope=fixture["parent_path"],
        expected_parent_envelope_file_sha256=fixture["parent_file_sha"],
        manual_ratings=tampered_path,
        expected_manual_ratings_file_sha256=_file_sha(tampered_path),
        output_directory=tmp_path / "bad-final",
    )
    with pytest.raises(
        evaluator.Lfm2CpcfV5StandaloneEvaluationError,
        match="exact stored review packet",
    ):
        evaluator.finalize_standalone_intelligence(bad_config)
