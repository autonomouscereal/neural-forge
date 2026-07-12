from __future__ import annotations

import hashlib

import pytest

from neural_forge.hf_on_policy_probe import GreedyProbeSettings
from neural_forge.hf_on_policy_refold import (
    OnPolicyRefoldConfig,
    OnPolicyRefoldIntegrityError,
    ParentRelativeMetricGate,
    run_on_policy_refold,
)


BENCHMARK_SHA256 = "b" * 64


class ByteTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self) -> int:
        return 258

    def get_vocab(self):
        return {
            "<pad>": 0,
            "<eos>": 1,
            **{f"byte-{value:03d}": value + 2 for value in range(256)},
        }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [value + 2 for value in text.encode("utf-8")]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return bytes(token_id - 2 for token_id in token_ids).decode("utf-8")


def _model(torch, value: float):
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([value]))
            self.generate_calls = 0

        def generate(
            self,
            input_ids,
            attention_mask,
            max_new_tokens,
            do_sample,
            num_beams,
            **_kwargs,
        ):
            assert not torch.is_grad_enabled()
            assert do_sample is False
            assert num_beams == 1
            assert attention_mask.shape == input_ids.shape
            self.generate_calls += 1
            suffix = torch.tensor(
                [[ord("!") + 2]], dtype=torch.long, device=input_ids.device
            )[:, :max_new_tokens]
            return torch.cat((input_ids, suffix), dim=1)

    return TinyModel()


def _probe_settings() -> GreedyProbeSettings:
    return GreedyProbeSettings(
        max_new_tokens=1,
        max_context_tokens=32,
        trace_batch_size=2,
        seed=11,
    )


def _config(*, min_retention: float) -> OnPolicyRefoldConfig:
    return OnPolicyRefoldConfig(
        metric_gates=(
            ParentRelativeMetricGate(
                "quality",
                min_parent_retention=min_retention,
                min_incumbent_gain=0.0,
            ),
        )
    )


def _evaluation(torch):
    def evaluate(context):
        assert not torch.is_grad_enabled()
        return context.seal(
            {"quality": float(context.model.weight.detach().item())},
            metadata={"suite": "tiny-local-free-run"},
        )

    return evaluate


def _cache_sha(role: str, corpus_identity_sha256: str) -> str:
    return hashlib.sha256(
        f"{role}:{corpus_identity_sha256}".encode("ascii")
    ).hexdigest()


def _run(
    torch,
    *,
    parent,
    child,
    fold,
    config,
    cache_root=None,
):
    return run_on_policy_refold(
        parent_model=parent,
        child_model=child,
        parent_tokenizer=ByteTokenizer(),
        child_tokenizer=ByteTokenizer(),
        seed_prompts=("A",),
        benchmark_sha256=BENCHMARK_SHA256,
        trace_refold_fn=fold,
        free_run_eval_fn=_evaluation(torch),
        config=config,
        probe_settings=_probe_settings(),
        generation_cache_root=cache_root,
        torch_module=torch,
    )


def test_accepted_refold_commits_and_emits_fingerprint_audit() -> None:
    torch = pytest.importorskip("torch")
    parent = _model(torch, 1.0)
    child = _model(torch, 0.2)
    child.weight.grad = torch.tensor([0.125])
    gradient = child.weight.grad

    def fold(context):
        assert not torch.is_grad_enabled()
        assert context.parent_trace_batches[0].tokenizer_name == "parent"
        assert context.child_trace_batches[0].tokenizer_name == "child"
        context.child_model.weight.fill_(0.8)
        return context.seal_result(
            parent_trace_cache_sha256=_cache_sha(
                "parent", context.corpus_identity.identity_sha256
            ),
            child_trace_cache_sha256=_cache_sha(
                "child", context.corpus_identity.identity_sha256
            ),
            details={"method": "tiny-closed-form"},
        )

    result = _run(
        torch,
        parent=parent,
        child=child,
        fold=fold,
        config=_config(min_retention=0.75),
    )

    assert result.accepted is True
    assert result.changed is True
    assert result.accepted_rounds == 1
    assert result.rejected_rounds == 0
    assert child.weight.item() == pytest.approx(0.8)
    assert child.weight.grad is gradient
    assert child.weight.grad.item() == pytest.approx(0.125)
    round_report = result.report["rounds"][0]
    assert round_report["accepted"] is True
    assert round_report["phase_order"] == (
        "child-prefix-generation",
        "parent-child-trace-refold",
        "parent-relative-free-run-gate",
        "commit",
    )
    assert round_report["gate"]["metric_gates"][0]["parent_retention"] == (
        pytest.approx(0.8)
    )
    assert round_report["mutation_audit"] == {
        "parent_before_sha256": result.report["initial_parent_sha256"],
        "parent_after_sha256": result.report["final_parent_sha256"],
        "parent_unchanged": True,
        "child_before_sha256": result.initial_child_sha256,
        "child_candidate_sha256": result.final_child_sha256,
        "child_final_sha256": result.final_child_sha256,
        "child_changed_by_refold": True,
        "parameter_identities_unchanged": True,
        "gradient_references_and_values_unchanged": True,
        "corpus_unchanged": True,
    }
    assert result.report["execution"]["optimizer_steps"] == 0
    assert result.report["execution"]["backward_calls"] == 0
    assert (
        result.report["execution"]["network_resources_called_by_orchestrator"] is False
    )
    with pytest.raises(TypeError):
        result.report["schema"] = "changed"


def test_rejected_refold_restores_exact_incumbent_state() -> None:
    torch = pytest.importorskip("torch")
    parent = _model(torch, 1.0)
    child = _model(torch, 0.2)
    parameter = child.weight
    child.weight.grad = torch.tensor([0.25])
    gradient = child.weight.grad

    def fold(context):
        context.child_model.weight.fill_(0.4)
        return context.seal_result(
            parent_trace_cache_sha256=_cache_sha(
                "parent", context.corpus_identity.identity_sha256
            ),
            child_trace_cache_sha256=_cache_sha(
                "child", context.corpus_identity.identity_sha256
            ),
        )

    result = _run(
        torch,
        parent=parent,
        child=child,
        fold=fold,
        config=_config(min_retention=0.75),
    )

    assert result.accepted is False
    assert result.changed is False
    assert result.accepted_rounds == 0
    assert result.rejected_rounds == 1
    assert child.weight is parameter
    assert child.weight.item() == pytest.approx(0.2)
    assert child.weight.grad is gradient
    assert child.weight.grad.item() == pytest.approx(0.25)
    round_report = result.report["rounds"][0]
    assert round_report["accepted"] is False
    assert round_report["rollback"] == {"performed": True, "exact": True}
    assert (
        round_report["mutation_audit"]["child_candidate_sha256"]
        != (round_report["mutation_audit"]["child_final_sha256"])
    )
    assert round_report["gate"]["metric_gates"][0]["retention_passed"] is False


def test_generation_and_trace_cache_identities_are_reused(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    seen_trace_identities: set[str] = set()

    def fold(context):
        identity = context.corpus_identity.identity_sha256
        hit = identity in seen_trace_identities
        seen_trace_identities.add(identity)
        context.child_model.weight.fill_(0.8)
        return context.seal_result(
            parent_trace_cache_sha256=_cache_sha("parent", identity),
            child_trace_cache_sha256=_cache_sha("child", identity),
            parent_trace_cache_hit=hit,
            child_trace_cache_hit=hit,
        )

    first_child = _model(torch, 0.2)
    first = _run(
        torch,
        parent=_model(torch, 1.0),
        child=first_child,
        fold=fold,
        config=_config(min_retention=0.75),
        cache_root=tmp_path,
    )
    second_child = _model(torch, 0.2)
    second = _run(
        torch,
        parent=_model(torch, 1.0),
        child=second_child,
        fold=fold,
        config=_config(min_retention=0.75),
        cache_root=tmp_path,
    )

    first_round = first.report["rounds"][0]
    second_round = second.report["rounds"][0]
    assert first_child.generate_calls == 1
    assert second_child.generate_calls == 0
    assert first_round["corpus_identity"]["generation_cache_hit"] is False
    assert second_round["corpus_identity"]["generation_cache_hit"] is True
    assert (
        first_round["corpus_identity"]["identity_sha256"]
        == (second_round["corpus_identity"]["identity_sha256"])
    )
    assert first_round["trace_refold"]["parent_trace_cache_hit"] is False
    assert second_round["trace_refold"]["parent_trace_cache_hit"] is True
    assert (
        first_round["trace_refold"]["parent_trace_cache_sha256"]
        == (second_round["trace_refold"]["parent_trace_cache_sha256"])
    )


def test_verified_parent_evaluation_cache_skips_parent_callback() -> None:
    torch = pytest.importorskip("torch")
    parent = _model(torch, 1.0)
    child = _model(torch, 0.2)
    roles = []

    def evaluate(context):
        roles.append(context.role)
        return context.seal({"quality": float(context.model.weight.item())})

    def fold(context):
        context.child_model.weight.fill_(0.8)
        return context.seal_result(
            parent_trace_cache_sha256=_cache_sha(
                "parent", context.corpus_identity.identity_sha256
            ),
            child_trace_cache_sha256=_cache_sha(
                "child", context.corpus_identity.identity_sha256
            ),
        )

    result = run_on_policy_refold(
        parent_model=parent,
        child_model=child,
        parent_tokenizer=ByteTokenizer(),
        child_tokenizer=ByteTokenizer(),
        seed_prompts=("A",),
        benchmark_sha256=BENCHMARK_SHA256,
        trace_refold_fn=fold,
        free_run_eval_fn=evaluate,
        cached_parent_evaluation={"quality": 1.0},
        config=_config(min_retention=0.75),
        probe_settings=_probe_settings(),
        torch_module=torch,
    )

    assert roles == ["incumbent", "candidate"]
    assert result.report["baseline"]["parent_source"] == "verified-cache"
    assert result.report["execution"]["parent_evaluation_callback_calls"] == 0


def test_mutation_audit_rejects_and_restores_parent_changes() -> None:
    torch = pytest.importorskip("torch")
    parent = _model(torch, 1.0)
    child = _model(torch, 0.2)
    parent_parameter = parent.weight
    child_parameter = child.weight

    def fold(context):
        context.parent_model.weight.fill_(9.0)
        context.child_model.weight.fill_(0.8)
        return context.seal_result(
            parent_trace_cache_sha256=_cache_sha(
                "parent", context.corpus_identity.identity_sha256
            ),
            child_trace_cache_sha256=_cache_sha(
                "child", context.corpus_identity.identity_sha256
            ),
        )

    with pytest.raises(
        OnPolicyRefoldIntegrityError,
        match="mutated the protected parent model",
    ):
        _run(
            torch,
            parent=parent,
            child=child,
            fold=fold,
            config=_config(min_retention=0.75),
        )

    assert parent.weight is parent_parameter
    assert child.weight is child_parameter
    assert parent.weight.item() == pytest.approx(1.0)
    assert child.weight.item() == pytest.approx(0.2)
