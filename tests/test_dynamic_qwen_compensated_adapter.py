from __future__ import annotations

from pathlib import Path

import pytest

from neural_forge.dynamic_behavior_gate import (
    CachedWholeOutputBehaviorGate,
    CandidateWholeOutput,
    WholeOutputCase,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from neural_forge.dynamic_qwen_compensated_adapter import (
    Qwen3CompensatedFoldStackAdapter,
    Qwen3CompensatedReplayEvidence,
    Qwen3CompensatedTargetSynthesisAdapter,
)
from neural_forge.dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
)
from neural_forge.hf_qwen_compensated_runtime import (
    load_compensated_qwen3,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)


def _model(seed: int):
    torch.manual_seed(seed)
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=12,
            hidden_size=8,
            intermediate_size=6,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    ).eval()


def _transaction(
    *,
    source_width: int,
    dropped: int,
    anchors: tuple[int, ...],
    coefficients: tuple[float, ...],
    before: int,
    digest_character: str,
) -> dict:
    target_width = source_width - 1
    after = before - 24
    return {
        "decision": "accepted-committed",
        "zero_optimizer_steps": 0,
        "backward_called": False,
        "proposal": {
            "layer_index": 1,
            "source_width": source_width,
            "target_width": target_width,
            "parameter_savings": 24,
            "fold_sha256": digest_character * 64,
            "merges": [
                {
                    "dropped": dropped,
                    "anchors": list(anchors),
                    "coefficients": list(coefficients),
                }
            ],
        },
        "compute_shape": {
            "compute_indices_sha256": canonical_sha256({"step": digest_character})
        },
        "before": {"unique_parameter_count": before},
        "after": {"unique_parameter_count": after},
    }


def _evidence(parent) -> Qwen3CompensatedReplayEvidence:
    parent_count = sum(value.numel() for value in parent.parameters())
    first = _transaction(
        source_width=6,
        dropped=5,
        anchors=(1, 3),
        coefficients=(0.5, -0.25),
        before=parent_count,
        digest_character="a",
    )
    second = _transaction(
        source_width=5,
        dropped=3,
        anchors=(0, 2),
        coefficients=(0.2, 0.1),
        before=parent_count - 24,
        digest_character="b",
    )
    report = {
        "accepted_count": 2,
        "accepted_transactions": [first, second],
        "stop_reason": "target-reached",
        "zero_optimizer_steps": 0,
        "backward_called": False,
        "recovery_examples": 0,
        "initial": {"unique_parameter_count": parent_count},
        "final": {"unique_parameter_count": parent_count - 48},
        "target": {
            "reached": True,
            "unique_parameter_count": parent_count - 48,
        },
    }
    return Qwen3CompensatedReplayEvidence.from_transaction_report(
        report,
        transaction_report_sha256="c" * 64,
        cached_parent_sha256="d" * 64,
        cached_candidate_sha256="e" * 64,
        manual_packet_sha256="f" * 64,
        expected_parent_state_sha256="1" * 64,
        expected_parent_state_tensor_count=len(parent.state_dict()),
        evidence_fingerprint=canonical_sha256({"tiny": "evidence"}),
        max_transient_compute_bytes=16 * 1024**2,
    )


class _ForwardGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, model, cases):
        self.calls += 1
        with torch.inference_mode():
            output = model(
                input_ids=torch.tensor([[1, 3, 4]], dtype=torch.long),
                use_cache=False,
            )
        assert torch.isfinite(output.logits).all()
        return {
            case.case_id: CandidateWholeOutput("A coherent answer.") for case in cases
        }


def test_tiny_real_qwen_replay_runs_through_automatic_system(
    tmp_path: Path,
) -> None:
    parent = _model(17)
    evidence = _evidence(parent)
    tokenizer_fingerprint = canonical_sha256({"tokenizer": "tiny-qwen"})
    generator = _ForwardGenerator()
    before_reload_calls = []
    gate = CachedWholeOutputBehaviorGate(
        (
            WholeOutputCase(
                case_id="coherence",
                prompt="Explain one fact.",
                parent_output="A coherent answer.",
                critical_nonempty=True,
            ),
        ),
        generator=generator,
    )
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    target_adapter = Qwen3CompensatedTargetSynthesisAdapter(evidence)
    stack_adapter = Qwen3CompensatedFoldStackAdapter(
        evidence=evidence,
        behavior_gate=gate,
        tokenizer_source=None,
        reload_device="cpu",
        snapshot_config=DynamicFoldSnapshotConfig.disk(
            snapshots,
            chunk_bytes=97,
        ),
        evidence_binding={"tiny_evidence": evidence.evidence_fingerprint},
        before_reload=lambda: before_reload_calls.append("released"),
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((target_adapter,)),
        stack_adapters=DynamicFoldStackAdapterRegistry((stack_adapter,)),
    )
    destination = tmp_path / "published"

    result = system.run(
        parent_models={"qwen-parent": parent},
        target_request=DynamicTargetRequest(
            child_name="qwen-compensated-child",
            target_family="qwen3",
            target_modalities=("text",),
            exact_unique_parameters=evidence.expected_target_parameters,
            tokenizer_fingerprint=tokenizer_fingerprint,
            adapter_id=target_adapter.adapter_id,
        ),
        evidence_fingerprint=evidence.evidence_fingerprint,
        destination=destination,
        parent_tokenizer_fingerprints={
            "qwen-parent": tokenizer_fingerprint,
        },
    )

    assert result.complete
    assert result.published.compilation.run.whole_model_snapshot_backend == "disk"
    assert result.target.resource_profile.unique_parameters == (
        evidence.expected_target_parameters
    )
    assert result.published.compilation.plan.actions[0].method_id == (
        "qwen3_compensated_transaction_replay"
    )
    assert result.published.artifact.exact_unique_parameters == (
        evidence.expected_target_parameters
    )
    assert generator.calls == 3
    assert before_reload_calls == ["released"]
    assert not tuple(snapshots.iterdir())

    reloaded = load_compensated_qwen3(
        destination / "child",
        device="cpu",
    )
    assert sum(value.numel() for value in reloaded.parameters()) == (
        evidence.expected_target_parameters
    )
    assert reloaded.model.layers[1].mlp.gate_proj.out_features == 4
    with torch.inference_mode():
        logits = reloaded(
            input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            use_cache=False,
        ).logits
    assert torch.isfinite(logits).all()


def test_transaction_report_rejects_training_or_broken_width_chain() -> None:
    parent = _model(19)
    parent_count = sum(value.numel() for value in parent.parameters())
    transaction = _transaction(
        source_width=6,
        dropped=5,
        anchors=(1,),
        coefficients=(0.5,),
        before=parent_count,
        digest_character="a",
    )
    report = {
        "accepted_count": 1,
        "accepted_transactions": [transaction],
        "stop_reason": "target-reached",
        "zero_optimizer_steps": 1,
        "backward_called": False,
        "recovery_examples": 0,
        "initial": {"unique_parameter_count": parent_count},
        "final": {"unique_parameter_count": parent_count - 24},
        "target": {
            "reached": True,
            "unique_parameter_count": parent_count - 24,
        },
    }

    with pytest.raises(RuntimeError, match="zero-training"):
        Qwen3CompensatedReplayEvidence.from_transaction_report(
            report,
            transaction_report_sha256="a" * 64,
            cached_parent_sha256="b" * 64,
            cached_candidate_sha256="c" * 64,
            manual_packet_sha256="d" * 64,
            expected_parent_state_sha256="e" * 64,
            expected_parent_state_tensor_count=len(parent.state_dict()),
            evidence_fingerprint="f" * 64,
        )
