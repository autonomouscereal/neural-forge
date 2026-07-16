from __future__ import annotations

from types import SimpleNamespace

import pytest

from neural_forge import hf_recovery


def test_heldout_behavior_summary_preserves_literal_prefix_generation_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def generated_batch(
        _torch,
        _model,
        _tokenizer,
        prompts,
        _device,
        _max_new_tokens,
    ):
        return [
            {
                "prompt": prompt,
                "output": f"{prompt}Expected reasoning stays coherent",
                "diagnostics": {
                    "distinct_1": 1.0,
                    "distinct_2": 1.0,
                    "repeated_bigram_share": 0.0,
                    "top_token_share": 0.25,
                    "max_token_run": 1,
                    "generated_token_ids": [10, 11, 12, 13],
                    "generated_token_count": 4,
                    "generated_distinct_token_ratio": 1.0,
                    "generated_max_token_id_run": 1,
                },
            }
            for prompt in prompts
        ]

    monkeypatch.setattr(hf_recovery, "_generate_text_batch", generated_batch)
    probe = {
        "id": "literal-prefix-probe",
        "category": "parent-lcp4-decoded-prefix",
        "prompt": "Prompt: ",
        "kind": "literal_prefix",
        "expected_prefix": "Expected",
        "expected_prefix_generated_token_count": 1,
    }

    summary = hf_recovery._heldout_behavior_summary(
        None,
        object(),
        object(),
        "cpu",
        16,
        1,
        behavior_probes=[probe],
        generation_prompts=[
            {**probe, "relevance_rules": [], "require_generation_health": True}
        ],
        max_generation_prompts=1,
    )

    assert summary["correct"] == summary["total"] == 1
    assert summary["generation_health_correct"] == 1
    assert summary["generation_health_total"] == 1
    assert summary["generation_health_rate"] == pytest.approx(1.0)
    assert summary["generation_items"][0]["passed"] is True


def test_heldout_behavior_summary_rejects_junk_after_literal_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def generated_batch(*_args, **_kwargs):
        return [
            {
                "prompt": "Prompt: ",
                "output": "Prompt: Expected !?.;",
                "diagnostics": {
                    "distinct_1": 1.0,
                    "distinct_2": 1.0,
                    "repeated_bigram_share": 0.0,
                    "top_token_share": 0.25,
                    "max_token_run": 1,
                    "generated_token_ids": [10, 11, 12, 13, 14],
                    "generated_token_count": 5,
                    "generated_distinct_token_ratio": 1.0,
                    "generated_max_token_id_run": 1,
                },
            }
        ]

    monkeypatch.setattr(hf_recovery, "_generate_text_batch", generated_batch)
    generation = {
        "id": "literal-prefix-generation",
        "category": "parent-lcp4-decoded-prefix",
        "prompt": "Prompt: ",
        "kind": "literal_prefix",
        "expected_prefix": "Expected",
        "expected_prefix_generated_token_count": 1,
        "relevance_rules": [],
        "require_generation_health": True,
    }
    summary = hf_recovery._heldout_behavior_summary(
        None,
        object(),
        object(),
        "cpu",
        16,
        1,
        behavior_probes=[generation],
        generation_prompts=[generation],
        max_generation_prompts=1,
    )

    assert summary["generation_health_correct"] == 0
    assert summary["generation_health_total"] == 1
    assert summary["generation_items"][0]["passed"] is False


def test_frozen_on_policy_replay_bank_preserves_ordered_utf8_bytes() -> None:
    mutable_bank = [["alpha\r\nline", "βeta"], ["third", "fourth"]]

    folded_arm = hf_recovery._freeze_on_policy_replay_bank(mutable_bank)
    blank_arm = hf_recovery._freeze_on_policy_replay_bank(
        [["alpha\r\nline", "βeta"], ["third", "fourth"]]
    )
    expected_sha = hf_recovery._on_policy_replay_bank_sha256(folded_arm)

    mutable_bank[0][0] = "changed after freeze"
    mutable_bank.append(["extra"])

    assert (
        folded_arm
        == blank_arm
        == (
            ("alpha\r\nline", "βeta"),
            ("third", "fourth"),
        )
    )
    assert hf_recovery._on_policy_replay_bank_sha256(folded_arm) == expected_sha
    assert hf_recovery._on_policy_replay_bank_sha256(blank_arm) == expected_sha
    assert (
        hf_recovery._on_policy_replay_bank_sha256(tuple(reversed(blank_arm)))
        != expected_sha
    )
    assert hf_recovery._ordered_recovery_texts_sha256(
        tuple(reversed(blank_arm[0]))
    ) != (hf_recovery._ordered_recovery_texts_sha256(blank_arm[0]))


def test_recovery_checkpoint_observer_payload_is_deeply_read_only() -> None:
    observations = []

    def observer(event) -> None:
        with pytest.raises(TypeError):
            event["step"] = 99
        with pytest.raises(TypeError):
            event["behavior"]["nested"]["value"] = "changed"
        observations.append(event)

    hf_recovery._notify_recovery_checkpoint_observer(
        observer,
        operation="recover_cross_tokenizer_causal_lm",
        checkpoint_index=2,
        behavior_entry={"step": 8, "nested": {"value": "stable"}},
    )

    assert len(observations) == 1
    assert (
        observations[0]["schema"]
        == "neural-forge.recovery-behavior-checkpoint-observer.v1"
    )
    assert observations[0]["checkpoint_index"] == 2
    assert observations[0]["behavior"]["nested"]["value"] == "stable"


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"
    all_special_ids = (0, 1)

    def __len__(self) -> int:
        return 4

    def __call__(
        self,
        texts,
        *,
        return_tensors: str,
        padding=False,
        truncation: bool,
        max_length: int,
        **_kwargs,
    ):
        assert return_tensors == "pt"
        torch = pytest.importorskip("torch")
        rows = [texts] if isinstance(texts, str) else list(texts)
        encoded = [
            [2 + (ord(char) % 2) for char in text][:max_length] or [2] for text in rows
        ]
        width = max(len(row) for row in encoded) if padding else len(encoded[0])
        padded = []
        masks = []
        for row in encoded:
            pad_count = width - len(row)
            if self.padding_side == "left":
                padded.append([self.pad_token_id] * pad_count + row)
                masks.append([0] * pad_count + [1] * len(row))
            else:
                padded.append(row + [self.pad_token_id] * pad_count)
                masks.append([1] * len(row) + [0] * pad_count)
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def decode(self, token_ids, **_kwargs) -> str:
        return "x" * sum(
            int(token_id) not in self.all_special_ids for token_id in token_ids
        )

    def save_pretrained(self, _target) -> None:
        return None


class _FakeCausalLM:
    def __init__(
        self,
        torch,
        initial_value: float,
        *,
        generation_calls: list[str],
        forward_calls: list[dict[str, tuple[tuple[int, ...], ...]]],
        name: str,
        allow_generation: bool = False,
    ) -> None:
        self._torch = torch
        self._module = torch.nn.Module()
        self._module.register_parameter(
            "weight",
            torch.nn.Parameter(torch.tensor(float(initial_value), dtype=torch.float32)),
        )
        self.config = SimpleNamespace(use_cache=True)
        self.generation_calls = generation_calls
        self.forward_calls = forward_calls
        self.name = name
        self.allow_generation = allow_generation

    @property
    def training(self) -> bool:
        return self._module.training

    def parameters(self):
        return self._module.parameters()

    def named_parameters(self):
        return self._module.named_parameters()

    def state_dict(self):
        return self._module.state_dict()

    def load_state_dict(self, state) -> None:
        self._module.load_state_dict(state)

    def to(self, _device):
        return self

    def train(self, mode: bool = True):
        self._module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, **kwargs):
        weight = next(self.parameters())
        input_ids = kwargs.get("input_ids")
        if input_ids is not None:
            attention_mask = kwargs.get("attention_mask")
            self.forward_calls.append(
                {
                    "input_ids": tuple(
                        tuple(int(token) for token in row) for row in input_ids.tolist()
                    ),
                    "attention_mask": (
                        ()
                        if attention_mask is None
                        else tuple(
                            tuple(int(token) for token in row)
                            for row in attention_mask.tolist()
                        )
                    ),
                }
            )
        batch_size = 1 if input_ids is None else int(input_ids.shape[0])
        sequence_length = 3 if input_ids is None else int(input_ids.shape[1])
        vocab_offsets = (
            self._torch.arange(4, dtype=weight.dtype, device=weight.device) * 0.01
        )
        logits = weight.expand(batch_size, sequence_length, 4) + vocab_offsets
        return SimpleNamespace(logits=logits, hidden_states=(logits,))

    def generate(self, **kwargs):
        self.generation_calls.append(self.name)
        if not self.allow_generation:
            raise AssertionError("Frozen replay must bypass arm-specific generation.")
        input_ids = kwargs["input_ids"]
        continuation = self._torch.full(
            (int(input_ids.shape[0]), 1),
            2,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return self._torch.cat([input_ids, continuation], dim=1)

    def save_pretrained(self, _target, *, safe_serialization: bool) -> None:
        assert safe_serialization is True


def _install_fake_cross_tokenizer_recovery(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_initial_value: float,
    allow_candidate_generation: bool = False,
):
    torch = pytest.importorskip("torch")
    generation_calls: list[str] = []
    source_forward_calls: list[dict[str, tuple[tuple[int, ...], ...]]] = []
    candidate_forward_calls: list[dict[str, tuple[tuple[int, ...], ...]]] = []
    reference_forward_calls: list[dict[str, tuple[tuple[int, ...], ...]]] = []
    tokenizer_load_calls: list[tuple[object, dict[str, object]]] = []
    model_load_calls: list[tuple[object, dict[str, object]]] = []
    paired_text_calls: list[tuple[str, ...]] = []
    teacher_batch_calls: list[tuple[str, ...]] = []
    adamw_kwargs: list[dict[str, object]] = []
    optimizer_step_calls: list[None] = []
    source = _FakeCausalLM(
        torch,
        0.25,
        generation_calls=generation_calls,
        forward_calls=source_forward_calls,
        name="teacher",
    )
    candidate = _FakeCausalLM(
        torch,
        candidate_initial_value,
        generation_calls=generation_calls,
        forward_calls=candidate_forward_calls,
        name="candidate",
        allow_generation=allow_candidate_generation,
    )
    reference = _FakeCausalLM(
        torch,
        -0.1,
        generation_calls=generation_calls,
        forward_calls=reference_forward_calls,
        name="reference",
    )
    tokenizer = _FakeTokenizer()

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model, **kwargs):
            tokenizer_load_calls.append((model, dict(kwargs)))
            return tokenizer

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model, **kwargs):
            model_load_calls.append((model, dict(kwargs)))
            if model == "teacher":
                return source
            if model == "reference":
                return reference
            return candidate

    fake_transformers = SimpleNamespace(
        AutoTokenizer=_AutoTokenizer,
        AutoModelForCausalLM=_AutoModelForCausalLM,
    )
    real_adamw = torch.optim.AdamW

    def audited_adamw(parameters, **kwargs):
        adamw_kwargs.append(dict(kwargs))
        optimizer = real_adamw(parameters, **kwargs)
        original_step = optimizer.step

        def audited_step(*args, **step_kwargs):
            optimizer_step_calls.append(None)
            return original_step(*args, **step_kwargs)

        optimizer.step = audited_step
        return optimizer

    def paired_batches(_source_tokenizer, _candidate_tokenizer, texts, **_kwargs):
        paired_text_calls.append(tuple(texts))
        input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "raw_texts": list(texts),
        }
        return [(dict(batch), dict(batch))]

    def candidate_batches(_tokenizer, texts, **_kwargs):
        teacher_batch_calls.append(tuple(texts))
        input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
        return [{"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}]

    def differentiable_loss(_torch, student, _teacher, *_args, **_kwargs):
        return student.logits.to(torch.float32).square().mean()

    monkeypatch.setattr(
        hf_recovery, "_torch_and_transformers", lambda: (torch, fake_transformers)
    )
    monkeypatch.setattr(
        hf_recovery, "_shared_recovery_anchors", lambda *_args, **_kwargs: [("a", 0, 0)]
    )
    monkeypatch.setattr(
        hf_recovery,
        "_set_train_scope",
        lambda model, _scope: sum(p.numel() for p in model.parameters()),
    )
    monkeypatch.setattr(hf_recovery, "_make_paired_batches", paired_batches)
    monkeypatch.setattr(hf_recovery, "_make_batches", candidate_batches)
    monkeypatch.setattr(
        hf_recovery,
        "_cross_tokenizer_span_pairs",
        lambda *_args, **_kwargs: torch.empty((0, 3), dtype=torch.long),
    )
    monkeypatch.setattr(
        hf_recovery,
        "_candidate_to_source_token_groups",
        lambda *_args, **_kwargs: (
            torch.arange(4, dtype=torch.long).view(4, 1),
            torch.ones((4, 1), dtype=torch.float32),
            torch.ones((4, 1), dtype=torch.float32),
            torch.ones(4, dtype=torch.bool),
        ),
    )
    monkeypatch.setattr(
        hf_recovery,
        "_projected_vocab_group_summary",
        lambda *_args, **_kwargs: {
            "valid_rows": 4,
            "multi_source_rows": 0,
            "digit_valid_rows": 0,
            "code_valid_rows": 0,
        },
    )
    monkeypatch.setattr(
        hf_recovery, "_cross_tokenizer_recovery_loss", differentiable_loss
    )
    monkeypatch.setattr(
        hf_recovery,
        "_candidate_lm_stabilizer_loss",
        lambda _torch, student, *_args: (
            student.logits.to(torch.float32).square().mean()
        ),
    )
    monkeypatch.setattr(
        hf_recovery,
        "_teacher_generated_lm_texts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Precomputed teacher texts must bypass donor generation.")
        ),
    )
    monkeypatch.setattr(
        hf_recovery,
        "_heldout_behavior_summary",
        lambda *_args, **_kwargs: {
            "correct": 1,
            "total": 1,
            "pass_rate": 1.0,
            "generation_health_correct": 1,
            "generation_health_total": 1,
            "generation_health_rate": 1.0,
            "items": [],
            "generation_items": [],
        },
    )
    monkeypatch.setattr(torch.optim, "AdamW", audited_adamw)
    return {
        "generation_calls": generation_calls,
        "source_forward_calls": source_forward_calls,
        "candidate_forward_calls": candidate_forward_calls,
        "reference_forward_calls": reference_forward_calls,
        "tokenizer_load_calls": tokenizer_load_calls,
        "model_load_calls": model_load_calls,
        "paired_text_calls": paired_text_calls,
        "teacher_batch_calls": teacher_batch_calls,
        "adamw_kwargs": adamw_kwargs,
        "optimizer_step_calls": optimizer_step_calls,
        "candidate": candidate,
    }


def _run_fake_fairness_arm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    arm_name: str,
    candidate_initial_value: float,
    adamw_foreach: bool | None,
    behavior_eval_checkpoint_steps=(),
    behavior_checkpoint_observer=None,
    stepwise_donor_kd_weight: float = 0.0,
    stepwise_replay_bank=None,
    allow_candidate_generation: bool = False,
    local_files_only: bool = False,
    reference_kd_weight: float = 0.0,
):
    audit = _install_fake_cross_tokenizer_recovery(
        monkeypatch,
        candidate_initial_value=candidate_initial_value,
        allow_candidate_generation=allow_candidate_generation,
    )
    replay_bank = (
        ("Question: one\r\nAnswer: shared β", "prompt two -> shared"),
        ("Question: three\nAnswer: shared",),
    )
    report = hf_recovery.recover_cross_tokenizer_causal_lm(
        "teacher",
        arm_name,
        tmp_path / arm_name,
        steps=2,
        batch_size=1,
        max_length=16,
        lr=1e-3,
        anchor_kd_weight=0.0,
        lm_weight=1.0,
        reference_model="reference" if reference_kd_weight > 0 else None,
        reference_kd_weight=reference_kd_weight,
        teacher_generate_lm_weight=0.1,
        stepwise_donor_kd_weight=stepwise_donor_kd_weight,
        on_policy_anchor_kd_weight=0.2,
        on_policy_hidden_kd_weight=0.1,
        on_policy_interval=2,
        hidden_kd_metric="cka",
        calibration_texts=("calibration",),
        precomputed_teacher_generated_texts=("teacher shared α", "teacher shared two"),
        on_policy_replay_bank=replay_bank,
        stepwise_replay_bank=stepwise_replay_bank,
        behavior_eval_checkpoint_steps=behavior_eval_checkpoint_steps,
        behavior_checkpoint_observer=behavior_checkpoint_observer,
        adamw_foreach=adamw_foreach,
        local_files_only=local_files_only,
        device="cpu",
    )
    return report, audit, replay_bank


def _fake_stepwise_event(
    step: int, continuation: tuple[int, ...] = (2,)
) -> dict[str, object]:
    prompt_input_ids = (3, 3)
    return {
        "step": step,
        "rows": [
            {
                "prompt_input_ids": list(prompt_input_ids),
                "rollout_input_ids": list(prompt_input_ids + continuation),
            }
        ],
    }


def test_stepwise_token_replay_freezes_exact_ids_and_hashes_order() -> None:
    mutable_bank = [
        _fake_stepwise_event(1, (2, 3)),
        _fake_stepwise_event(2, (3,)),
    ]
    frozen = hf_recovery._freeze_stepwise_replay_bank(mutable_bank)
    frozen_hash = hf_recovery._stepwise_replay_bank_sha256(frozen)

    mutable_bank[0]["rows"][0]["prompt_input_ids"][0] = 2
    mutable_bank[0]["rows"][0]["rollout_input_ids"].append(2)

    assert frozen[0].rows[0].prompt_input_ids == (3, 3)
    assert frozen[0].rows[0].rollout_input_ids == (3, 3, 2, 3)
    assert hf_recovery._stepwise_replay_bank_sha256(frozen) == frozen_hash
    assert (
        hf_recovery._stepwise_replay_bank_sha256(tuple(reversed(frozen))) != frozen_hash
    )
    changed_token = hf_recovery._freeze_stepwise_replay_bank(
        (_fake_stepwise_event(1, (3, 3)), _fake_stepwise_event(2, (3,)))
    )
    assert hf_recovery._stepwise_replay_bank_sha256(changed_token) != frozen_hash


@pytest.mark.parametrize(
    ("bank", "error_type", "message"),
    [
        (("legacy text replay",), TypeError, "mapping with step and rows"),
        (
            (
                {
                    "step": 1,
                    "rows": [{"prompt_input_ids": [2], "rollout_input_ids": [2]}],
                },
            ),
            ValueError,
            "continuation token",
        ),
        (
            (
                {
                    "step": 1,
                    "rows": [{"prompt_input_ids": [2], "rollout_input_ids": [3, 2]}],
                },
            ),
            ValueError,
            "exact rollout prefix",
        ),
        (
            (
                {
                    "step": 1,
                    "rows": [{"prompt_input_ids": [True], "rollout_input_ids": [1, 2]}],
                },
            ),
            TypeError,
            "integer token ID",
        ),
    ],
)
def test_stepwise_token_replay_schema_is_fail_closed(
    bank, error_type, message: str
) -> None:
    with pytest.raises(error_type, match=message):
        hf_recovery._freeze_stepwise_replay_bank(bank)


def test_stepwise_token_replay_rejects_decode_reencode_boundary_change() -> None:
    prompt_ids_from_generation = [2]
    reencoded_full_text_ids = [3, 2]
    with pytest.raises(ValueError, match="exact rollout prefix"):
        hf_recovery._freeze_stepwise_replay_bank(
            (
                {
                    "step": 1,
                    "rows": [
                        {
                            "prompt_input_ids": prompt_ids_from_generation,
                            "rollout_input_ids": reencoded_full_text_ids,
                        }
                    ],
                },
            )
        )


def test_folded_and_blank_arms_consume_identical_replay_without_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    folded_report, folded_audit, replay_bank = _run_fake_fairness_arm(
        monkeypatch,
        tmp_path,
        arm_name="folded",
        candidate_initial_value=0.8,
        adamw_foreach=False,
    )
    monkeypatch.undo()
    blank_report, blank_audit, _ = _run_fake_fairness_arm(
        monkeypatch,
        tmp_path,
        arm_name="blank",
        candidate_initial_value=-0.4,
        adamw_foreach=False,
    )

    assert folded_audit["generation_calls"] == []
    assert blank_audit["generation_calls"] == []
    assert folded_audit["paired_text_calls"] == [
        ("calibration",),
        replay_bank[0],
        replay_bank[0],
        replay_bank[1],
        replay_bank[1],
    ]
    assert blank_audit["paired_text_calls"] == folded_audit["paired_text_calls"]
    assert folded_audit["teacher_batch_calls"] == [
        ("teacher shared α", "teacher shared two"),
    ]
    assert blank_audit["teacher_batch_calls"] == folded_audit["teacher_batch_calls"]
    assert folded_report["teacher_generated_text_source"] == "precomputed"
    assert (
        blank_report["teacher_generated_ordered_text_sha256"]
        == (folded_report["teacher_generated_ordered_text_sha256"])
    )
    assert folded_report["on_policy_dynamic_candidate_generation"] is False
    assert blank_report["on_policy_dynamic_candidate_generation"] is False
    assert (
        blank_report["on_policy_replay_bank_sha256"]
        == folded_report["on_policy_replay_bank_sha256"]
    )
    assert (
        blank_report["on_policy_replay_consumption"]
        == folded_report["on_policy_replay_consumption"]
    )
    assert [item["step"] for item in folded_report["on_policy_replay_consumption"]] == [
        1,
        2,
    ]
    assert folded_audit["adamw_kwargs"] == [{"lr": 1e-3, "foreach": False}]
    assert blank_audit["adamw_kwargs"] == [{"lr": 1e-3, "foreach": False}]


def test_legacy_adamw_path_omits_foreach_argument(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    report, audit, _ = _run_fake_fairness_arm(
        monkeypatch,
        tmp_path,
        arm_name="legacy-optimizer",
        candidate_initial_value=0.1,
        adamw_foreach=None,
    )

    assert audit["adamw_kwargs"] == [{"lr": 1e-3}]
    assert report["adamw_foreach"] is None
    assert report["adamw_foreach_explicit"] is False
    assert report["local_files_only"] is False
    assert all(
        call_kwargs["local_files_only"] is False
        for _model, call_kwargs in (
            audit["tokenizer_load_calls"] + audit["model_load_calls"]
        )
    )


def test_local_files_only_reaches_source_candidate_and_reference_loaders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    report, audit, _ = _run_fake_fairness_arm(
        monkeypatch,
        tmp_path,
        arm_name="offline-loaders",
        candidate_initial_value=0.1,
        adamw_foreach=False,
        local_files_only=True,
        reference_kd_weight=0.2,
    )

    assert [model for model, _kwargs in audit["tokenizer_load_calls"]] == [
        "teacher",
        "offline-loaders",
        "reference",
    ]
    assert [model for model, _kwargs in audit["model_load_calls"]] == [
        "teacher",
        "offline-loaders",
        "reference",
    ]
    assert all(
        call_kwargs["local_files_only"] is True
        for _model, call_kwargs in (
            audit["tokenizer_load_calls"] + audit["model_load_calls"]
        )
    )
    assert report["local_files_only"] is True


@pytest.mark.parametrize("invalid_value", [1, "true", None])
def test_invalid_local_files_only_fails_before_any_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    invalid_value,
) -> None:
    audit = _install_fake_cross_tokenizer_recovery(
        monkeypatch,
        candidate_initial_value=0.0,
    )

    with pytest.raises(TypeError, match="local_files_only must be a bool"):
        hf_recovery.recover_cross_tokenizer_causal_lm(
            "teacher",
            "candidate",
            tmp_path / "invalid-local-files-only",
            steps=1,
            anchor_kd_weight=0.0,
            lm_weight=1.0,
            calibration_texts=("calibration",),
            local_files_only=invalid_value,
            device="cpu",
        )

    assert audit["tokenizer_load_calls"] == []
    assert audit["model_load_calls"] == []


def test_folded_and_blank_arms_consume_identical_stepwise_replay_without_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    stepwise_bank = (
        _fake_stepwise_event(1, (2,)),
        _fake_stepwise_event(2, (3, 2)),
    )
    folded_report, folded_audit, _ = _run_fake_fairness_arm(
        monkeypatch,
        tmp_path,
        arm_name="stepwise-folded",
        candidate_initial_value=0.6,
        adamw_foreach=False,
        stepwise_donor_kd_weight=0.3,
        stepwise_replay_bank=stepwise_bank,
    )
    monkeypatch.undo()
    blank_report, blank_audit, _ = _run_fake_fairness_arm(
        monkeypatch,
        tmp_path,
        arm_name="stepwise-blank",
        candidate_initial_value=-0.2,
        adamw_foreach=False,
        stepwise_donor_kd_weight=0.3,
        stepwise_replay_bank=stepwise_bank,
    )

    assert folded_audit["generation_calls"] == []
    assert blank_audit["generation_calls"] == []
    assert folded_report["stepwise_dynamic_candidate_generation"] is False
    assert blank_report["stepwise_dynamic_candidate_generation"] is False
    assert (
        folded_report["stepwise_replay_bank_sha256"]
        == blank_report["stepwise_replay_bank_sha256"]
    )
    assert (
        folded_report["stepwise_replay_consumption"]
        == blank_report["stepwise_replay_consumption"]
    )
    assert [item["step"] for item in folded_report["stepwise_replay_consumption"]] == [
        1,
        2,
    ]
    assert [
        item["row_count"] for item in folded_report["stepwise_replay_consumption"]
    ] == [1, 1]
    assert [
        item["kd_position_count"]
        for item in folded_report["stepwise_replay_consumption"]
    ] == [1, 2]
    assert folded_report["stepwise_scheduled_steps"] == [1, 2]
    assert (
        folded_report["stepwise_replay_bank_schema"]
        == hf_recovery._STEPWISE_REPLAY_SCHEMA
    )
    assert (
        folded_report["stepwise_replay_preflight"]
        == blank_report["stepwise_replay_preflight"]
    )
    preflight = folded_report["stepwise_replay_preflight"]
    assert preflight["schema"] == hf_recovery._STEPWISE_REPLAY_PREFLIGHT_SCHEMA
    assert preflight["status"] == "validated"
    assert preflight["event_count"] == 2
    assert [event["optimizer_step"] for event in preflight["events"]] == [1, 2]
    assert [event["kd_position_count"] for event in preflight["events"]] == [1, 2]
    preflight_body = dict(preflight)
    preflight_sha256 = preflight_body.pop("sha256")
    assert preflight_sha256 == hf_recovery._stepwise_replay_preflight_sha256(
        preflight_body
    )
    assert folded_audit["optimizer_step_calls"] == [None, None]
    assert blank_audit["optimizer_step_calls"] == [None, None]
    assert any(
        call["input_ids"] == ((3, 3, 2),)
        for call in folded_audit["candidate_forward_calls"]
    )
    assert any(
        call["input_ids"] == ((3, 3, 3, 2),)
        for call in folded_audit["candidate_forward_calls"]
    )


def test_stepwise_replay_bank_rejects_wrong_optimizer_step_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_cross_tokenizer_recovery(monkeypatch, candidate_initial_value=0.0)

    with pytest.raises(ValueError, match="exactly one ordered token replay event"):
        hf_recovery.recover_cross_tokenizer_causal_lm(
            "teacher",
            "candidate",
            tmp_path / "bad-stepwise-bank",
            steps=2,
            anchor_kd_weight=0.0,
            lm_weight=1.0,
            stepwise_donor_kd_weight=0.1,
            calibration_texts=("calibration",),
            stepwise_replay_bank=(_fake_stepwise_event(1),),
            device="cpu",
        )


def test_stepwise_replay_bank_rejects_wrong_step_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_cross_tokenizer_recovery(monkeypatch, candidate_initial_value=0.0)

    with pytest.raises(ValueError, match="expected steps"):
        hf_recovery.recover_cross_tokenizer_causal_lm(
            "teacher",
            "candidate",
            tmp_path / "bad-stepwise-order",
            steps=2,
            anchor_kd_weight=0.0,
            lm_weight=1.0,
            stepwise_donor_kd_weight=0.1,
            calibration_texts=("calibration",),
            stepwise_replay_bank=(_fake_stepwise_event(2), _fake_stepwise_event(1)),
            device="cpu",
        )


@pytest.mark.parametrize(
    ("first_event", "message"),
    [
        (
            {
                "step": 1,
                "rows": [{"prompt_input_ids": [2, 3], "rollout_input_ids": [2, 3, 2]}],
            },
            "do not match the current runtime prompt",
        ),
        (
            {
                "step": 1,
                "rows": [
                    {"prompt_input_ids": [3, 3], "rollout_input_ids": [3, 3, 2]},
                    {"prompt_input_ids": [3, 3], "rollout_input_ids": [3, 3, 2]},
                ],
            },
            "exactly one row per current prompt",
        ),
        (
            {
                "step": 1,
                "rows": [{"prompt_input_ids": [3, 3], "rollout_input_ids": [3, 3, 4]}],
            },
            "outside the candidate vocabulary",
        ),
        (
            {
                "step": 1,
                "rows": [
                    {
                        "prompt_input_ids": [3, 3],
                        "rollout_input_ids": [3, 3] + [2] * 17,
                    }
                ],
            },
            "exceeds max_new_tokens",
        ),
        (
            {
                "step": 1,
                "rows": [
                    {
                        "prompt_input_ids": [3, 3],
                        "rollout_input_ids": [3, 3] + [2] * 16,
                    }
                ],
            },
            "exceeds max_length",
        ),
    ],
)
def test_stepwise_token_replay_runtime_validation_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    first_event,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _run_fake_fairness_arm(
            monkeypatch,
            tmp_path,
            arm_name="invalid-stepwise-runtime",
            candidate_initial_value=0.0,
            adamw_foreach=False,
            stepwise_donor_kd_weight=0.3,
            stepwise_replay_bank=(first_event, _fake_stepwise_event(2)),
        )


def test_stepwise_preflight_rejects_bad_later_event_before_any_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    audit = _install_fake_cross_tokenizer_recovery(
        monkeypatch,
        candidate_initial_value=0.7,
    )
    candidate = audit["candidate"]
    initial_state = {
        key: value.detach().clone() for key, value in candidate.state_dict().items()
    }
    observer_events = []
    bad_second_event = {
        "step": 2,
        "rows": [
            {
                "prompt_input_ids": [2, 3],
                "rollout_input_ids": [2, 3, 2],
            }
        ],
    }

    with pytest.raises(ValueError, match="do not match the current runtime prompt"):
        hf_recovery.recover_cross_tokenizer_causal_lm(
            "teacher",
            "candidate",
            tmp_path / "preflight-must-fail-before-training",
            steps=2,
            batch_size=1,
            max_length=16,
            lr=1e-3,
            anchor_kd_weight=0.0,
            lm_weight=1.0,
            stepwise_donor_kd_weight=0.3,
            calibration_texts=("calibration",),
            stepwise_replay_bank=(_fake_stepwise_event(1), bad_second_event),
            behavior_eval_checkpoint_steps=(0, 2),
            behavior_checkpoint_observer=observer_events.append,
            adamw_foreach=False,
            device="cpu",
        )

    assert audit["generation_calls"] == []
    assert audit["source_forward_calls"] == []
    assert audit["candidate_forward_calls"] == []
    assert audit["optimizer_step_calls"] == []
    assert observer_events == []
    assert not (tmp_path / "preflight-must-fail-before-training").exists()
    for key, initial_value in initial_state.items():
        assert torch.equal(candidate.state_dict()[key], initial_value)


def test_stepwise_token_replay_padding_mask_keeps_valid_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _install_fake_cross_tokenizer_recovery(
        monkeypatch, candidate_initial_value=0.0
    )
    torch, transformers = hf_recovery._torch_and_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained("candidate")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    candidate = transformers.AutoModelForCausalLM.from_pretrained("candidate")
    replay_event = hf_recovery._freeze_stepwise_replay_bank(
        (
            {
                "step": 1,
                "rows": [
                    {"prompt_input_ids": [3, 3], "rollout_input_ids": [3, 3, 1]},
                    {"prompt_input_ids": [2, 2], "rollout_input_ids": [2, 2, 3, 1]},
                ],
            },
        )
    )[0]

    generated, mask, prompt_lengths, rollout_lengths = (
        hf_recovery._validated_stepwise_replay_tensors(
            torch,
            candidate,
            tokenizer,
            ("ca", "db"),
            replay_event,
            prompt_max_length=8,
            max_new_tokens=4,
            max_length=8,
            device="cpu",
        )
    )

    assert tuple(tuple(int(token) for token in row) for row in generated.tolist()) == (
        (3, 3, 1, 1),
        (2, 2, 3, 1),
    )
    assert tuple(tuple(int(token) for token in row) for row in mask.tolist()) == (
        (1, 1, 1, 0),
        (1, 1, 1, 1),
    )
    assert prompt_lengths == [2, 2]
    assert rollout_lengths == [3, 4]
    assert audit["generation_calls"] == []


def test_omitted_stepwise_replay_bank_preserves_dynamic_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    report, audit, _ = _run_fake_fairness_arm(
        monkeypatch,
        tmp_path,
        arm_name="legacy-stepwise-generation",
        candidate_initial_value=0.2,
        adamw_foreach=None,
        stepwise_donor_kd_weight=0.3,
        stepwise_replay_bank=None,
        allow_candidate_generation=True,
    )

    assert audit["generation_calls"] == ["candidate", "candidate"]
    assert report["stepwise_dynamic_candidate_generation"] is True
    assert report["stepwise_replay_bank_supplied"] is False
    assert report["stepwise_replay_preflight"] is None
    assert report["stepwise_replay_consumption"] == []


def test_explicit_behavior_checkpoints_notify_read_only_observer_without_best_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    events = []

    def observer(event) -> None:
        with pytest.raises(TypeError):
            event["behavior"]["pass_rate"] = 0.0
        events.append(event)

    report, _audit, _ = _run_fake_fairness_arm(
        monkeypatch,
        tmp_path,
        arm_name="observed",
        candidate_initial_value=0.3,
        adamw_foreach=False,
        behavior_eval_checkpoint_steps=(0, 2),
        behavior_checkpoint_observer=observer,
    )

    assert report["save_best_behavior"] is False
    assert report["behavior_eval_checkpoint_steps"] == [0, 2]
    assert [item["step"] for item in report["behavior_history"]] == [0, 2]
    assert report["behavior_checkpoint_observer_calls"] == 2
    assert [event["step"] for event in events] == [0, 2]
    assert all(
        event["operation"] == "recover_cross_tokenizer_causal_lm" for event in events
    )


@pytest.mark.parametrize(
    ("checkpoint_steps", "message"),
    [
        ((1, 1), "strictly increasing"),
        ((2, 1), "strictly increasing"),
        ((-1,), "must be in"),
        ((3,), "must be in"),
    ],
)
def test_explicit_behavior_checkpoint_steps_are_fail_closed(
    checkpoint_steps, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        hf_recovery._normalize_behavior_checkpoint_steps(checkpoint_steps, steps=2)


def test_replay_bank_requires_one_batch_per_scheduled_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_cross_tokenizer_recovery(monkeypatch, candidate_initial_value=0.0)

    with pytest.raises(ValueError, match="exactly one ordered text batch"):
        hf_recovery.recover_cross_tokenizer_causal_lm(
            "teacher",
            "candidate",
            tmp_path / "bad-bank",
            steps=3,
            anchor_kd_weight=0.0,
            lm_weight=1.0,
            on_policy_anchor_kd_weight=0.1,
            on_policy_interval=2,
            calibration_texts=("calibration",),
            on_policy_replay_bank=(("only step one",),),
            device="cpu",
        )
