from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    fingerprint_model_state,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_macro_cell_attractor_gate import (  # noqa: E402
    Lfm2MacroCellAttractorGate,
    Lfm2MacroCellAttractorGateConfig,
    Lfm2TargetTokenFreeRolloutConfig,
    Lfm2TargetTokenFreeRolloutSource,
)
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (  # noqa: E402
    Lfm2TargetTokenizedRecord,
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    seal_strict_tokenizer_bridge_artifact,
)


def _parent_model():
    config = transformers.Lfm2Config(
        vocab_size=41,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2701)
        return transformers.Lfm2ForCausalLM(config)


def _batch(prompt, forced):
    input_ids = torch.tensor([list(prompt)], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "forced_decode_ids": torch.tensor([list(forced)], dtype=torch.long),
    }


def _greedy_forced(model, prompt, steps):
    prefix = torch.tensor([list(prompt)], dtype=torch.long)
    generated = []
    model.eval()
    with torch.inference_mode():
        for _ in range(steps):
            output = model(
                input_ids=prefix,
                attention_mask=torch.ones_like(prefix),
                use_cache=False,
                return_dict=True,
            )
            token = int(output.logits[0, -1].argmax().item())
            generated.append(token)
            prefix = torch.cat(
                (prefix, torch.tensor([[token]], dtype=torch.long)),
                dim=1,
            )
    return tuple(generated)


@pytest.fixture(scope="module")
def sealed_trace_cache(tmp_path_factory):
    parent = _parent_model()
    prompts = ((1, 2, 3, 4), (5, 6, 7, 8))
    cache = build_lfm2_circuit_trace_cache(
        tmp_path_factory.mktemp("lfm2-macro-cell-attractor-parent"),
        parent,
        [
            _batch(prompt, _greedy_forced(parent, prompt, 4))
            for prompt in prompts
        ],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=4,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    return parent, cache


class _BridgeTokenizer:
    def __init__(self, surfaces: list[str]) -> None:
        self.surfaces = list(surfaces)
        self.all_special_ids: list[int] = []
        self.bos_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None
        self.unk_token_id = None

    def __len__(self) -> int:
        return len(self.surfaces)

    def get_vocab(self) -> dict[str, int]:
        return {
            f"raw-{index}-{surface}": index
            for index, surface in enumerate(self.surfaces)
        }

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"raw-{token_id}-{self.surfaces[token_id]}"

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.surfaces[token_id] for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [self.surfaces.index(text)]


@pytest.fixture(scope="module")
def bridged_target_rollout(tmp_path_factory):
    parent = _parent_model()
    donor_tokenizer = _BridgeTokenizer([f"<{index}>" for index in range(41)])
    target_tokenizer = _BridgeTokenizer(
        donor_tokenizer.surfaces[1:] + donor_tokenizer.surfaces[:1]
    )
    bridge_root = tmp_path_factory.mktemp("lfm2-target-rollout-bridge")
    (bridge_root / "tokenizer.json").write_text("{}", encoding="utf-8")
    bridge = seal_strict_tokenizer_bridge_artifact(
        donor_tokenizer,
        target_tokenizer,
        bridge_root,
        donor_row_capacity=41,
        target_row_capacity=41,
    )
    mapping = bridge.transport_profile.target_to_donor_row_indices
    inverse = {donor_id: target_id for target_id, donor_id in enumerate(mapping)}
    target_prompts = ((1, 2, 3, 4), (5, 6, 7, 8))
    target_records = []
    donor_batches = []
    target_forced_sequences = []
    for index, target_prompt in enumerate(target_prompts):
        donor_prompt = tuple(mapping[token_id] for token_id in target_prompt)
        donor_forced = _greedy_forced(parent, donor_prompt, 4)
        if index == 0:
            donor_forced = (
                (donor_forced[0] + 1) % 41,
                *donor_forced[1:],
            )
        target_forced = tuple(inverse[token_id] for token_id in donor_forced)
        target_forced_sequences.append(target_forced)
        donor_batches.append(_batch(donor_prompt, donor_forced))
        target_input_ids = torch.tensor([target_prompt], dtype=torch.long)
        target_records.append(
            Lfm2TargetTokenizedRecord(
                record_id=f"target-record-{index}",
                split="evaluation",
                input_ids=target_input_ids,
                attention_mask=torch.ones_like(target_input_ids),
                forced_decode_ids=torch.tensor(
                    [target_forced],
                    dtype=torch.long,
                ),
            )
        )
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        bridge,
        target_records,
    )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path_factory.mktemp("lfm2-target-rollout-parent-cache"),
        parent,
        donor_batches,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=4,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    used_target_ids = {
        token_id for forced in target_forced_sequences for token_id in forced
    }
    eos_token_id = next(
        token_id for token_id in range(41) if token_id not in used_target_ids
    )
    return parent, cache, adapter, eos_token_id


class _ScriptedTargetChild(torch.nn.Module):
    def __init__(self, adapter, *, eos_token_id: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=adapter.target_row_capacity,
            eos_token_id=eos_token_id,
        )
        self.anchor = torch.nn.Parameter(torch.tensor([0.0]))
        self.probe = torch.nn.Dropout(p=0.5)
        self.script: dict[tuple[int, ...], int] = {}
        for record in adapter.records:
            input_ids = record.target_input_ids
            attention_mask = record.attention_mask.to(dtype=torch.bool)
            forced_decode_ids = record.target_forced_decode_ids
            for row_index in range(int(input_ids.shape[0])):
                prefix = [
                    int(value)
                    for value in input_ids[row_index][attention_mask[row_index]]
                    .detach()
                    .cpu()
                    .tolist()
                ]
                for token_id in forced_decode_ids[row_index].tolist():
                    self.script[tuple(prefix)] = int(token_id)
                    prefix.append(int(token_id))
        self.fallback_token_id = next(
            token_id
            for token_id in range(self.config.vocab_size)
            if token_id != eos_token_id
        )

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        use_cache,
        return_dict,
    ):
        assert use_cache is False
        assert return_dict is True
        assert attention_mask.shape == input_ids.shape
        logits = torch.full(
            (
                int(input_ids.shape[0]),
                int(input_ids.shape[1]),
                self.config.vocab_size,
            ),
            -15.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for row_index in range(int(input_ids.shape[0])):
            prefix = tuple(
                int(value)
                for value in input_ids[row_index][
                    attention_mask[row_index].to(dtype=torch.bool)
                ]
                .detach()
                .cpu()
                .tolist()
            )
            target = self.script.get(prefix, self.fallback_token_id)
            logits[row_index, -1, target] = 15.0
        return SimpleNamespace(logits=logits + self.anchor * 0.0)


class _TransitionChild(torch.nn.Module):
    def __init__(
        self,
        *,
        delta: int | None = None,
        constant_token: int | None = None,
        logit_scale: float = 20.0,
    ) -> None:
        super().__init__()
        if (delta is None) == (constant_token is None):
            raise ValueError("Select exactly one transition behavior.")
        self.config = SimpleNamespace(vocab_size=41, eos_token_id=40)
        self.anchor = torch.nn.Parameter(torch.tensor([0.0]))
        self.logit_scale = torch.nn.Parameter(
            torch.tensor(float(logit_scale)),
            requires_grad=False,
        )
        if constant_token is not None:
            transition = torch.full((41,), int(constant_token), dtype=torch.long)
        else:
            # Keep generated continuations below EOS while changing their wording.
            transition = (torch.arange(41, dtype=torch.long) + int(delta)) % 40
        self.register_buffer("transition", transition)
        self.probe = torch.nn.Dropout(p=0.5)

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        use_cache,
        return_dict,
    ):
        assert use_cache is False
        assert return_dict is True
        assert attention_mask.shape == input_ids.shape
        targets = self.transition[input_ids]
        logits = torch.nn.functional.one_hot(
            targets,
            num_classes=self.config.vocab_size,
        ).to(dtype=torch.float32)
        logits = logits * self.logit_scale + self.anchor * 0.0
        return SimpleNamespace(logits=logits)


def _gate(cache):
    return Lfm2MacroCellAttractorGate(
        cache,
        config=Lfm2MacroCellAttractorGateConfig(
            maximum_records=2,
            maximum_steps=4,
            eos_token_id=40,
            maximum_entropy_parent_distance_regression=1e-9,
            maximum_repetition_excess_regression=1e-9,
            maximum_dominant_token_excess_regression=1e-9,
            maximum_eos_position_error_regression=0.0,
            collapse_entropy_threshold=0.10,
            collapse_repetition_share_threshold=0.80,
            collapse_dominant_token_share_threshold=0.80,
            maximum_new_collapsed_records=0,
        ),
    )


def _without_sha(payload, field):
    return {key: value for key, value in payload.items() if key != field}


def test_coherent_nonexact_rollout_passes_without_parent_or_state_mutation(
    sealed_trace_cache,
    monkeypatch,
):
    parent, cache = sealed_trace_cache

    def forbidden_parent_forward(*args, **kwargs):
        raise AssertionError("The sealed parent must never execute in this gate.")

    monkeypatch.setattr(parent, "forward", forbidden_parent_forward)
    gate = _gate(cache)
    incumbent_model = _TransitionChild(delta=1)
    candidate_model = _TransitionChild(delta=2)
    candidate_model.train()
    candidate_model.probe.eval()
    candidate_model.anchor.grad = torch.tensor([3.0])
    state_before = fingerprint_model_state(candidate_model)
    gradient_before = candidate_model.anchor.grad.detach().clone()
    flags_before = (
        candidate_model.training,
        candidate_model.probe.training,
    )

    incumbent = gate.capture(incumbent_model, phase="pre-cell")
    candidate = gate.capture(candidate_model, phase="post-cell")
    repeated_candidate = gate.capture(candidate_model, phase="post-cell")
    decision = gate.compare(incumbent, candidate)

    assert decision.accepted
    assert decision.failed_checks == ()
    assert candidate.snapshot_sha256 == repeated_candidate.snapshot_sha256
    assert fingerprint_model_state(candidate_model) == state_before
    assert torch.equal(candidate_model.anchor.grad, gradient_before)
    assert (
        candidate_model.training,
        candidate_model.probe.training,
    ) == flags_before

    candidate_payload = candidate.as_dict()
    decision_payload = decision.as_dict()
    assert (
        candidate_payload["metrics"][
            "diagnostic_mean_forced_exact_token_match_fraction"
        ]
        == 0.0
    )
    assert candidate_payload["metrics"]["collapsed_record_count"] == 0
    assert all(
        record["diagnostics"]["forced_exact_token_match_fraction"] == 0.0
        for record in candidate_payload["records"]
    )
    assert all("exact_token" not in name for name in decision_payload["checks"])
    assert decision_payload["audit"]["parent_forward_calls"] == 0
    assert decision_payload["audit"]["tokenizer_accessed"] is False
    assert (
        decision_payload["audit"]["exact_token_metrics_used_for_acceptance"]
        is False
    )
    assert candidate_payload["audit"]["state_preserved"] is True
    assert candidate_payload["audit"]["child_use_cache"] is False
    assert (
        candidate_payload["audit"][
            "sealed_parent_greedy_rollouts_authenticated"
        ]
        is True
    )
    assert candidate_payload["audit"]["training_steps"] == 0
    assert candidate_payload["audit"]["backward_calls"] == 0
    assert candidate_payload["audit"]["quantization_used"] is False
    assert candidate_payload["audit"]["pruning_used"] is False
    assert candidate_payload["audit"]["audit_sha256"] == canonical_sha256(
        _without_sha(candidate_payload["audit"], "audit_sha256")
    )
    assert decision_payload["audit"]["audit_sha256"] == canonical_sha256(
        _without_sha(decision_payload["audit"], "audit_sha256")
    )


def test_selection_excludes_non_greedy_rows_before_deterministic_ranking(
    tmp_path,
):
    parent = _parent_model()
    prompt = (1, 2, 3, 4)
    greedy = _greedy_forced(parent, prompt, 4)
    non_greedy = ((greedy[0] + 1) % 40, *greedy[1:])
    cache = build_lfm2_circuit_trace_cache(
        tmp_path / "mixed-greedy-cache",
        parent,
        [
            _batch(prompt, non_greedy),
            _batch((5, 6, 7, 8), _greedy_forced(parent, (5, 6, 7, 8), 4)),
        ],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=4,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    gate = Lfm2MacroCellAttractorGate(
        cache,
        config=Lfm2MacroCellAttractorGateConfig(
            maximum_records=1,
            maximum_steps=4,
            eos_token_id=40,
        ),
    )

    payload = gate.capture(
        _TransitionChild(delta=1),
        phase="authenticated-only",
    ).as_dict()

    assert payload["selection"]["eligible_authenticated_record_count"] == 1
    assert payload["selection"]["rejected_non_greedy_record_count"] == 1
    assert payload["selection"]["selected_records"] == [
        {
            "batch_index": 1,
            "row_index": 0,
            "input_sha256": cache.load_batch(1).input_sha256,
            "horizon": 4,
        }
    ]
    assert payload["metrics"]["record_count"] == 1
    assert payload["records"][0]["diagnostics"][
        "sealed_parent_greedy_rollout_authenticated"
    ]


def test_repetitive_dominant_low_entropy_attractor_is_rejected(
    sealed_trace_cache,
):
    _, cache = sealed_trace_cache
    gate = _gate(cache)
    incumbent = gate.capture(_TransitionChild(delta=1), phase="pre-cell")
    collapsed_model = _TransitionChild(constant_token=9)
    state_before = fingerprint_model_state(collapsed_model)

    candidate = gate.capture(collapsed_model, phase="post-cell")
    decision = gate.compare(incumbent, candidate)

    assert not decision.accepted
    assert "new_collapsed_records" in decision.failed_checks
    assert fingerprint_model_state(collapsed_model) == state_before
    candidate_payload = candidate.as_dict()
    decision_payload = decision.as_dict()
    assert candidate_payload["metrics"]["collapsed_record_count"] == 2
    assert len(decision_payload["new_collapsed_record_ids"]) == 2
    assert all(
        record["child"]["normalized_entropy"] < 0.10
        and record["child"]["maximum_repeated_bigram_share"] == 1.0
        and record["child"]["dominant_token_share"] == 1.0
        and record["collapsed"] is True
        for record in candidate_payload["records"]
    )
    assert (
        candidate_payload["metrics"][
            "diagnostic_mean_forced_exact_token_match_fraction"
        ]
        == 0.0
    )
    assert (
        decision_payload["diagnostics"]["exact_token_match_is_diagnostic_only"]
        is True
    )


def test_target_token_rollout_uses_bridge_domain_without_parent_or_mutation(
    bridged_target_rollout,
    monkeypatch,
):
    parent, cache, adapter, eos_token_id = bridged_target_rollout

    def forbidden_parent_forward(*args, **kwargs):
        raise AssertionError("Target-token rollout must not execute the parent.")

    monkeypatch.setattr(parent, "forward", forbidden_parent_forward)
    source = Lfm2TargetTokenFreeRolloutSource(
        cache,
        adapter,
        config=Lfm2TargetTokenFreeRolloutConfig(
            maximum_records=2,
            maximum_steps=4,
            eos_token_id=eos_token_id,
        ),
    )
    child = _ScriptedTargetChild(adapter, eos_token_id=eos_token_id)
    child.train()
    child.probe.eval()
    child.anchor.grad = torch.tensor([7.0])
    state_before = fingerprint_model_state(child)
    gradient_before = child.anchor.grad.detach().clone()
    flags_before = (child.training, child.probe.training)

    snapshot = source.capture(child, phase="target-aware")
    repeated = source.capture(child, phase="target-aware")
    payload = snapshot.as_dict()

    assert snapshot.snapshot_sha256 == repeated.snapshot_sha256
    assert fingerprint_model_state(child) == state_before
    assert torch.equal(child.anchor.grad, gradient_before)
    assert (child.training, child.probe.training) == flags_before
    assert payload["metrics"]["record_count"] == 2
    assert (
        payload["metrics"][
            "diagnostic_mean_teacher_route_exact_token_match_fraction"
        ]
        == 1.0
    )
    assert payload["metrics"]["evaluation_only_step_count"] == 0
    assert payload["quality_contract"]["numeric_quality_gate"] is False
    assert payload["quality_contract"]["horizon_censored_is_failure"] is False
    assert '"accepted"' not in json.dumps(payload, sort_keys=True)
    assert any(
        int(cache.load_batch(index).tensors["prefill.logits"][0, -1].argmax())
        != int(record.donor_forced_decode_ids[0, 0])
        for index, record in enumerate(adapter.records)
    )
    for record in payload["records"]:
        adapter_record = adapter.records[record["batch_index"]]
        target_prompt = adapter_record.target_input_ids[0].tolist()
        donor_prompt = adapter_record.donor_input_ids[0].tolist()
        assert record["target_prompt_token_ids"] == target_prompt
        assert target_prompt != donor_prompt
        assert record["child"]["token_ids"] == record["teacher_forced_route"][
            "token_ids"
        ]
        assert record["child"]["termination"]["termination_kind"] == "horizon_censored"
        assert record["child"]["termination"]["censored"] is True
        assert all(step["paired_fit_eligible"] for step in record["steps"])
        assert all(
            step["reference_target_token_id"]
            == step["cached_parent_projected_target_top1_token_id"]
            for step in record["steps"]
        )
        assert all(
            step["child_matches_teacher_forced_route"]
            for step in record["steps"]
        )
    assert payload["audit"]["parent_forward_calls"] == 0
    assert payload["audit"]["parent_trace_regeneration_calls"] == 0
    assert payload["audit"]["tokenizer_accessed"] is False
    assert payload["audit"]["training_steps"] == 0
    assert payload["audit"]["backward_calls"] == 0
    assert payload["audit"]["optimizer_steps"] == 0
    assert payload["audit"]["state_preserved"] is True
    assert payload["audit"]["audit_sha256"] == canonical_sha256(
        _without_sha(payload["audit"], "audit_sha256")
    )


def test_target_token_rollout_marks_post_divergence_unpaired_and_eos_terminal(
    bridged_target_rollout,
):
    _, cache, adapter, eos_token_id = bridged_target_rollout
    source = Lfm2TargetTokenFreeRolloutSource(
        cache,
        adapter,
        config=Lfm2TargetTokenFreeRolloutConfig(
            maximum_records=1,
            maximum_steps=4,
            eos_token_id=eos_token_id,
        ),
    )
    baseline = source.capture(
        _ScriptedTargetChild(adapter, eos_token_id=eos_token_id),
        phase="locate-selection",
    ).as_dict()
    selected = baseline["records"][0]
    prompt = tuple(selected["target_prompt_token_ids"])
    reference_first = int(selected["teacher_forced_route"]["token_ids"][0])
    wrong_token = next(
        token_id
        for token_id in range(adapter.target_row_capacity)
        if token_id not in (eos_token_id, reference_first)
    )

    divergent_child = _ScriptedTargetChild(adapter, eos_token_id=eos_token_id)
    divergent_child.script[prompt] = wrong_token
    divergent = source.capture(
        divergent_child,
        phase="divergent-evaluation",
    ).as_dict()
    record = divergent["records"][0]
    assert record["diagnostics"]["teacher_route_first_divergence_step"] == 0
    assert record["steps"][0]["paired_fit_eligible"] is True
    assert all(not step["paired_fit_eligible"] for step in record["steps"][1:])
    assert record["causal_fit"]["paired_fit_step_count"] == 1
    assert record["causal_fit"]["first_unpaired_step"] == 1
    assert record["causal_fit"]["post_divergence_steps_are_evaluation_only"]
    assert record["child"]["termination"]["termination_kind"] == "horizon_censored"
    assert divergent["quality_contract"]["horizon_censored_is_failure"] is False
    assert '"accepted"' not in json.dumps(divergent, sort_keys=True)

    eos_child = _ScriptedTargetChild(adapter, eos_token_id=eos_token_id)
    eos_child.script[prompt] = eos_token_id
    eos_payload = source.capture(eos_child, phase="explicit-eos").as_dict()
    eos_record = eos_payload["records"][0]
    assert len(eos_record["steps"]) == 1
    assert eos_record["steps"][0]["paired_fit_eligible"] is True
    assert eos_record["child"]["termination"] == {
        "termination_kind": "eos",
        "censored": False,
        "eos_position": 0,
        "observed_token_count": 1,
        "configured_horizon": 4,
    }
    assert eos_payload["metrics"]["child_eos_termination_count"] == 1
