from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from neural_forge.dynamic_behavior_gate import resolve_behavior_gate_binding
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_qwen_direct_local import file_sha256


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_dynamic_qwen3_compensated_v10.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_dynamic_qwen3_compensated_v10",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _generation_rows(*, empty_candidate: bool = False):
    rows = []
    diagnostics = []
    for index in range(25):
        case_id = f"case-{index:02d}"
        prompt = f"Prompt {index}: "
        continuation = "" if empty_candidate else "A coherent fact."
        rows.append(
            {
                "id": case_id,
                "prompt": prompt,
                "output": prompt + continuation,
                "category": "facts",
                "relevance_rules": [[r".*"]],
                "diagnostics": {
                    "generated_token_count": 0 if empty_candidate else 4,
                    "repeated_bigram_share": 0.0,
                },
            }
        )
        diagnostics.append(
            {
                "id": case_id,
                "category": "facts",
                "passed": not empty_candidate,
                "diagnostics": {
                    "generated_token_count": 0 if empty_candidate else 4,
                    "repeated_bigram_share": 0.0,
                },
            }
        )
    return rows, diagnostics


def _fixture_paths(tmp_path: Path, *, empty_candidate: bool = False):
    parent_rows, parent_diagnostics = _generation_rows()
    candidate_rows, candidate_diagnostics = _generation_rows(
        empty_candidate=empty_candidate
    )
    eval_suite = {
        "behavior_probe_count": 1,
        "choice_probe_count": 1,
        "generation_prompt_count": 25,
        "source": "tiny-cached-suite",
        "text_eval_count": 1,
    }
    identity = {
        "schema": "test-parent-identity",
        "state_sha256": "1" * 64,
        "state_tensor_count": 7,
        "config_sha256": "2" * 64,
        "tokenizer_sha256": "3" * 64,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    parent = {
        "model": "tiny-parent",
        "param_count": 100,
        "identity_binding": identity,
        "eval_suite": eval_suite,
        "generations": parent_rows,
        "generation_diagnostics": parent_diagnostics,
    }
    candidate = {
        "model": "qwen3-compensated-ragged-parent-free",
        "param_count": 91,
        "eval_suite": eval_suite,
        "generations": candidate_rows,
        "generation_diagnostics": candidate_diagnostics,
    }
    transaction = {
        "accepted_count": 1,
        "accepted_transactions": [
            {
                "decision": "accepted-committed",
                "zero_optimizer_steps": 0,
                "backward_called": False,
                "proposal": {
                    "layer_index": 2,
                    "source_width": 4,
                    "target_width": 3,
                    "parameter_savings": 9,
                    "fold_sha256": "4" * 64,
                    "merges": [
                        {
                            "dropped": 3,
                            "anchors": [0, 2],
                            "coefficients": [0.25, -0.5],
                        }
                    ],
                },
                "compute_shape": {
                    "compute_indices_sha256": "5" * 64,
                },
                "before": {"unique_parameter_count": 100},
                "after": {"unique_parameter_count": 91},
            }
        ],
        "stop_reason": "target-reached",
        "zero_optimizer_steps": 0,
        "backward_called": False,
        "recovery_examples": 0,
        "initial": {"unique_parameter_count": 100},
        "final": {"unique_parameter_count": 91},
        "target": {
            "reached": True,
            "unique_parameter_count": 91,
        },
    }
    certification = {
        "passed": True,
        "parent_forward_passes": 0,
        "perplexity_used_for_acceptance": False,
        "domains": {
            name: {
                "passed": True,
                "loss_count": 0,
                "parent_correct": 1,
                "retained_parent_passes": 1,
            }
            for name in ("behavior", "choice", "generation")
        },
    }
    packet = {
        "schema": "neural-forge-qualitative-parent-child-review-v1",
        "required_sample_count": 128,
        "selected_sample_count": 128,
        "status": "awaiting-explicit-review",
        "samples": [{"id": f"sample-{index}"} for index in range(128)],
    }
    packet["packet_sha256"] = canonical_sha256(packet)
    verdict = {
        "schema": "neural-forge-qualitative-review-verdict-v1",
        "packet_sha256": packet["packet_sha256"],
        "reviewed_sample_count": 128,
        "reviewer": "unit-test-reviewer",
        "notes": "Inspected all fixture outputs; each is coherent.",
        "verdict": "pass",
    }
    paths = MODULE.EvidencePaths(
        transaction_report=tmp_path / "transactions.json",
        parent_cache=tmp_path / "cached-source.json",
        candidate_eval=tmp_path / "candidate.json",
        manual_packet=tmp_path / "review-packet.json",
        certification_verdict=tmp_path / "certification.json",
        manual_verdict=tmp_path / "review-verdict.json",
    )
    values = {
        paths.transaction_report: transaction,
        paths.parent_cache: parent,
        paths.candidate_eval: candidate,
        paths.manual_packet: packet,
        paths.certification_verdict: certification,
        paths.manual_verdict: verdict,
    }
    for path, value in values.items():
        _write(path, value)
    hashes = MODULE.EvidenceHashes(
        transaction_report=file_sha256(paths.transaction_report),
        parent_cache=file_sha256(paths.parent_cache),
        candidate_eval=file_sha256(paths.candidate_eval),
        manual_packet=file_sha256(paths.manual_packet),
        certification_verdict=file_sha256(paths.certification_verdict),
    )
    expected_steps = (
        {
            "source_width": 4,
            "target_width": 3,
            "dropped": 3,
            "anchors": [0, 2],
            "coefficients": [0.25, -0.5],
            "fold_sha256": "4" * 64,
            "compute_indices_sha256": "5" * 64,
        },
    )
    return paths, hashes, expected_steps


def test_cached_candidate_preflight_accepts_bound_whole_outputs(
    tmp_path: Path,
) -> None:
    paths, hashes, expected_steps = _fixture_paths(tmp_path)

    result = MODULE.preflight_v10(
        paths,
        expected_hashes=hashes,
        expected_steps=expected_steps,
        expected_parent_parameters=100,
        expected_target_parameters=91,
        expected_layer_index=2,
    )

    assert result.accepted
    assert len(result.baseline.cases) == 25
    assert result.decision.metrics
    assert result.as_dict()["model_loading_performed"] is False
    assert result.evidence.removed_neurons == 1

    gate = MODULE._build_gate(
        result.baseline,
        lambda _model, _cases: {},
    )
    final_binding = resolve_behavior_gate_binding(gate.evaluate)
    fresh_binding = resolve_behavior_gate_binding(gate.evaluate_model)
    assert final_binding.manifest_fingerprint == fresh_binding.manifest_fingerprint
    assert final_binding.gate_version == 2
    assert final_binding.manifest["coverage_policy"]["required_splits"] == [
        "heldout"
    ]


def test_cached_candidate_behavior_failure_rejects_before_model_load(
    tmp_path: Path,
) -> None:
    paths, hashes, expected_steps = _fixture_paths(
        tmp_path,
        empty_candidate=True,
    )

    with pytest.raises(
        MODULE.Qwen3V10RunnerError,
        match="whole-output preflight",
    ):
        MODULE.preflight_v10(
            paths,
            expected_hashes=hashes,
            expected_steps=expected_steps,
            expected_parent_parameters=100,
            expected_target_parameters=91,
            expected_layer_index=2,
        )


def test_run_skips_loader_when_evidence_is_missing(tmp_path: Path) -> None:
    args = MODULE.build_parser().parse_args(
        [
            "--transaction-report",
            str(tmp_path / "missing-transactions.json"),
            "--parent-cache",
            str(tmp_path / "missing-source.json"),
            "--candidate-eval",
            str(tmp_path / "missing-candidate.json"),
            "--manual-packet",
            str(tmp_path / "missing-packet.json"),
            "--certification-verdict",
            str(tmp_path / "missing-certification.json"),
            "--manual-verdict",
            str(tmp_path / "missing-verdict.json"),
            "--output",
            str(tmp_path / "output"),
            "--report",
            str(tmp_path / "report.json"),
            "--snapshot-root",
            str(tmp_path),
        ]
    )
    calls = 0

    def forbidden_loader(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("The 4B loader must not run.")

    with pytest.raises(MODULE.Qwen3V10RunnerError, match="does not exist"):
        MODULE.run_production(args, model_loader=forbidden_loader)

    assert calls == 0


class _TinyTokenizer:
    pad_token_id = 0

    def __call__(self, *_args, **_kwargs):
        return {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def decode(self, token_ids, **_kwargs):
        return " ".join(str(value) for value in token_ids)


class _MalformedGenerationModel(nn.Module):
    def __init__(self, generated: torch.Tensor) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.generated = generated

    def generate(self, **_kwargs):
        return self.generated.to(device=self.anchor.device)


@pytest.mark.parametrize(
    ("generated", "message"),
    (
        (torch.tensor([11, 12, 13]), "invalid whole-output tensor"),
        (torch.tensor([[11], [11]]), "invalid whole-output tensor"),
        (torch.tensor([[11]]), "invalid whole-output tensor"),
        (torch.tensor([[11, 99, 13]]), "exact prompt prefix"),
    ),
)
def test_greedy_generator_rejects_malformed_or_changed_prompt_prefix(
    generated: torch.Tensor,
    message: str,
) -> None:
    generator = MODULE.GreedyWholeOutputGenerator(
        _TinyTokenizer(),
        maximum_prompt_tokens=16,
    )
    model = _MalformedGenerationModel(generated)
    case = MODULE.WholeOutputCase(
        case_id="prefix-contract",
        prompt="hello",
        parent_output="world",
    )

    with pytest.raises(MODULE.Qwen3V10RunnerError, match=message):
        generator(model, (case,))


def test_greedy_generator_receipt_preserves_exact_output_and_token_hash() -> None:
    generator = MODULE.GreedyWholeOutputGenerator(
        _TinyTokenizer(),
        maximum_prompt_tokens=16,
    )
    model = _MalformedGenerationModel(torch.tensor([[11, 12, 41, 42]]))
    case = MODULE.WholeOutputCase(
        case_id="receipt-contract",
        prompt="hello",
        parent_output="41 42",
    )

    outputs = generator(model, (case,))

    assert outputs["receipt-contract"].output == "41 42"
    summary = generator.invocations[0]["summaries"][0]
    assert summary["output"] == "41 42"
    assert summary["token_ids_sha256"] == MODULE.canonical_sha256([41, 42])
