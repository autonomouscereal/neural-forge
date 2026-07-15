from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any

import pytest
import torch

from neural_forge import hf_recovery
from neural_forge import hf_teacher_token_replay as replay


class _Backend:
    def __init__(self, identity: str) -> None:
        self.identity = identity

    def to_str(self) -> str:
        return json.dumps({"identity": self.identity}, sort_keys=True)


class _Config:
    def __init__(self, **values: Any) -> None:
        self.values = values

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


def _padded_tensor_batch(
    rows: Sequence[Sequence[int]],
    *,
    pad_id: int,
    padding_side: str,
    truncation_side: str,
    truncation: bool,
    max_length: int | None,
    ignore_truncation: bool = False,
) -> dict[str, torch.Tensor]:
    normalized = [list(row) for row in rows]
    if truncation and max_length is not None and not ignore_truncation:
        normalized = [
            (
                row[-int(max_length) :]
                if truncation_side == "left"
                else row[: int(max_length)]
            )
            for row in normalized
        ]
    width = max(len(row) for row in normalized)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for row in normalized:
        padding = [int(pad_id)] * (width - len(row))
        zeros = [0] * len(padding)
        ones = [1] * len(row)
        if padding_side == "left":
            input_rows.append(padding + row)
            mask_rows.append(zeros + ones)
        else:
            input_rows.append(row + padding)
            mask_rows.append(ones + zeros)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
    }


class _SourceTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    unk_token_id = 3
    sep_token_id = None
    cls_token_id = None
    mask_token_id = None
    all_special_ids = [0, 1, 2, 3]

    def __init__(self, *, backend_identity: str = "source-v1") -> None:
        self.padding_side = "right"
        self.truncation_side = "right"
        self.backend_tokenizer = _Backend(backend_identity)
        self._vocabulary = {f"source-{index}": index for index in range(64)}

    def __len__(self) -> int:
        return 64

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocabulary)

    @staticmethod
    def _prompt_ids(text: str) -> list[int]:
        return {
            "c": [2, 30],
            "d": [2, 31, 32],
            "e": [2, 33],
        }.get(text, [2, 34])

    def __call__(
        self,
        texts: Sequence[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert padding is True
        return _padded_tensor_batch(
            [self._prompt_ids(text) for text in texts],
            pad_id=self.pad_token_id,
            padding_side=self.padding_side,
            truncation_side=self.truncation_side,
            truncation=truncation,
            max_length=max_length,
        )

    @staticmethod
    def decode(
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        lookup = {
            (50,): "alibration",
            (51,): " changed",
            (52,): "   ",
        }
        frozen = tuple(int(token_id) for token_id in token_ids)
        continuation = lookup.get(frozen)
        if continuation is not None:
            return continuation
        prompt = ""
        if 30 in frozen:
            prompt = "c"
        elif 31 in frozen and 32 in frozen:
            prompt = "d"
        elif 33 in frozen:
            prompt = "e"
        if prompt and 50 in frozen:
            return f"{prompt}alibration"
        if prompt and 51 in frozen:
            return f"{prompt} changed"
        if prompt and 52 in frozen:
            return "   "
        return prompt or "unknown"


class _CandidateTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    unk_token_id = 3
    sep_token_id = None
    cls_token_id = None
    mask_token_id = None

    def __init__(
        self,
        *,
        continuation_mode: str = "normal",
        backend_identity: str = "candidate-v1",
        reverse_vocabulary: bool = False,
        ignore_prompt_truncation: bool = False,
        long_prompt: bool = False,
        eos_token_id: int = 1,
    ) -> None:
        self.padding_side = "right"
        self.truncation_side = "right"
        self.eos_token_id = eos_token_id
        self.all_special_ids = [0, eos_token_id, 2, 3]
        self.backend_tokenizer = _Backend(backend_identity)
        pairs = [(f"candidate-{index}", index) for index in range(128)]
        if reverse_vocabulary:
            pairs.reverse()
        self._vocabulary = dict(pairs)
        self.continuation_mode = continuation_mode
        self.ignore_prompt_truncation = ignore_prompt_truncation
        self.long_prompt = long_prompt
        self.generate_calls = 0

    def __len__(self) -> int:
        return 128

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocabulary)

    def generate(self, *_args: Any, **_kwargs: Any) -> None:
        self.generate_calls += 1
        raise AssertionError("The candidate side must never generate teacher replay.")

    def _prompt_ids(self, text: str) -> list[int]:
        if self.long_prompt:
            return [2, 10, 14, 15, 16, 17]
        return {
            "c": [2, 10],
            "d": [2, 11, 12],
            "e": [2, 13],
        }.get(text, [2, 18])

    def _continuation_ids(self, text: str) -> list[int]:
        if self.continuation_mode == "empty":
            return []
        if self.continuation_mode == "invalid-vocabulary":
            return [999]
        if self.continuation_mode == "too-many":
            return [20, 22]
        return {"alibration": [20], " changed": [21]}.get(text, [23])

    def __call__(self, value: Any, **kwargs: Any) -> dict[str, Any]:
        if isinstance(value, list):
            assert kwargs["return_tensors"] == "pt"
            return _padded_tensor_batch(
                [self._prompt_ids(text) for text in value],
                pad_id=self.pad_token_id,
                padding_side=self.padding_side,
                truncation_side=self.truncation_side,
                truncation=bool(kwargs["truncation"]),
                max_length=int(kwargs["max_length"]),
                ignore_truncation=self.ignore_prompt_truncation,
            )
        if kwargs.get("add_special_tokens") is False:
            assert kwargs.get("truncation") is False
            return {"input_ids": self._continuation_ids(str(value))}
        # Deliberately merge across the c + alibration boundary. Re-tokenizing the
        # full text is therefore not equivalent to preserving the prompt prefix.
        if value == "calibration":
            return {"input_ids": [2, 99]}
        return {"input_ids": self._prompt_ids(str(value))}

    @staticmethod
    def decode(
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        frozen = tuple(int(token_id) for token_id in token_ids)
        prompt = {
            (2, 10): "c",
            (2, 11, 12): "d",
            (2, 13): "e",
        }
        for prefix, text in prompt.items():
            if frozen[: len(prefix)] != prefix:
                continue
            continuation = frozen[len(prefix) :]
            if continuation == (20,):
                return f"candidate::{text}alibration"
            if continuation == (21,):
                return f"candidate::{text} changed"
            if continuation:
                return f"{text} continuation"
            return text
        return "decoded"


class _SourceModel(torch.nn.Module):
    def __init__(
        self,
        *,
        continuation_id: int = 50,
        emit_continuation: bool = True,
        fail_on_max_new_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor([3.25]))
        self.continuation_id = continuation_id
        self.emit_continuation = emit_continuation
        self.fail_on_max_new_tokens = fail_on_max_new_tokens
        self.config = _Config(
            architecture="fake-source", continuation_id=continuation_id
        )
        self.generation_config = _Config(do_sample=False)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        pad_token_id: int | None,
        eos_token_id: int | Sequence[int] | None,
    ) -> torch.Tensor:
        assert self.training is False
        assert torch.is_grad_enabled() is False
        assert do_sample is False
        self.calls.append(
            {
                "batch_size": int(input_ids.shape[0]),
                "max_new_tokens": int(max_new_tokens),
                "input_ids": input_ids.detach().cpu().tolist(),
                "attention_mask": attention_mask.detach().cpu().tolist(),
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
            }
        )
        if max_new_tokens == self.fail_on_max_new_tokens:
            raise RuntimeError("deliberate teacher generation failure")
        if not self.emit_continuation:
            return input_ids.clone()
        continuation = torch.full(
            (int(input_ids.shape[0]), 1),
            int(self.continuation_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, continuation), dim=1)


def _build(
    *,
    source_model: _SourceModel | None = None,
    source_tokenizer: _SourceTokenizer | None = None,
    candidate_tokenizer: _CandidateTokenizer | None = None,
    calibration_texts: Sequence[str] = ("c", "d", "e"),
    steps: int = 3,
    batch_size: int = 2,
    max_length: int = 8,
    calibration_order: str = "original",
    stepwise_max_new_tokens: int = 1,
    on_policy_max_new_tokens: int = 1,
    teacher_max_new_tokens: int = 2,
    on_policy_interval: int = 2,
) -> tuple[dict[str, Any], _SourceModel, _SourceTokenizer, _CandidateTokenizer]:
    model = source_model or _SourceModel()
    source = source_tokenizer or _SourceTokenizer()
    candidate = candidate_tokenizer or _CandidateTokenizer()
    payload = replay.build_teacher_token_replay_payload(
        torch=torch,
        source_model=model,
        source_tokenizer=source,
        candidate_tokenizer=candidate,
        calibration_texts=calibration_texts,
        steps=steps,
        batch_size=batch_size,
        max_length=max_length,
        response_loss_mode="none",
        calibration_order=calibration_order,
        stepwise_donor_max_new_tokens=stepwise_max_new_tokens,
        on_policy_max_new_tokens=on_policy_max_new_tokens,
        teacher_generate_max_new_tokens=teacher_max_new_tokens,
        on_policy_interval=on_policy_interval,
        device="cpu",
    )
    return payload, model, source, candidate


def _payload_sha256(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("payload_sha256")
    encoded = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_clone(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _reseal_payload(payload: dict[str, Any]) -> None:
    payload["payload_sha256"] = _payload_sha256(payload)


def test_source_only_generation_builds_exact_future_token_schedule(monkeypatch) -> None:
    prompt_calls: list[tuple[str, str]] = []
    original_prompt = hf_recovery._on_policy_prompt_text

    def observed_prompt(text: str, mode: str) -> str:
        prompt_calls.append((text, mode))
        return original_prompt(text, mode)

    monkeypatch.setattr(replay.hf_recovery, "_on_policy_prompt_text", observed_prompt)
    payload, model, source, candidate = _build()

    expected_rows = {
        "c": {"prompt_input_ids": [2, 10], "rollout_input_ids": [2, 10, 20]},
        "d": {
            "prompt_input_ids": [2, 11, 12],
            "rollout_input_ids": [2, 11, 12, 20],
        },
        "e": {"prompt_input_ids": [2, 13], "rollout_input_ids": [2, 13, 20]},
    }
    assert payload["stepwise_scheduled_steps"] == [1, 2, 3]
    assert payload["stepwise_replay_bank"] == [
        {"step": 1, "rows": [expected_rows["c"], expected_rows["d"]]},
        {"step": 2, "rows": [expected_rows["e"]]},
        {"step": 3, "rows": [expected_rows["c"], expected_rows["d"]]},
    ]
    assert payload["on_policy_scheduled_steps"] == [1, 2]
    assert payload["on_policy_replay_bank"] == [
        ["calibration", "dalibration"],
        ["ealibration"],
    ]
    assert payload["precomputed_teacher_generated_texts"] == [
        "calibration",
        "dalibration",
        "ealibration",
    ]
    assert prompt_calls == [
        ("c", "none"),
        ("d", "none"),
        ("e", "none"),
        ("c", "none"),
        ("d", "none"),
        ("c", "none"),
        ("d", "none"),
        ("e", "none"),
        ("c", "none"),
        ("d", "none"),
        ("e", "none"),
    ]

    # The full text would merge to token 99, but the replay retains the exact
    # candidate prompt [2, 10] and appends only the separately encoded suffix.
    assert candidate("calibration")["input_ids"] == [2, 99]
    assert (
        candidate.decode(
            [2, 10, 20],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        == "candidate::calibration"
    )
    assert payload["on_policy_replay_bank"][0][0] == "calibration"
    assert payload["stepwise_replay_bank"][0]["rows"][0]["rollout_input_ids"] == [
        2,
        10,
        20,
    ]
    assert candidate.generate_calls == 0
    assert [call["batch_size"] for call in model.calls] == [2, 1, 2, 2, 1, 2, 1]
    assert [call["max_new_tokens"] for call in model.calls] == [1, 1, 1, 1, 1, 2, 2]
    assert source.padding_side == "right"
    assert candidate.padding_side == "right"
    assert model.training is True
    assert torch.equal(model.marker.detach(), torch.tensor([3.25]))

    frozen_stepwise = hf_recovery._freeze_stepwise_replay_bank(
        payload["stepwise_replay_bank"]
    )
    frozen_on_policy = hf_recovery._freeze_on_policy_replay_bank(
        payload["on_policy_replay_bank"]
    )
    provenance = payload["provenance_sha256"]
    assert provenance["stepwise_replay_bank"] == (
        hf_recovery._stepwise_replay_bank_sha256(frozen_stepwise)
    )
    assert provenance["on_policy_replay_bank"] == (
        hf_recovery._on_policy_replay_bank_sha256(frozen_on_policy)
    )
    assert payload["payload_sha256"] == _payload_sha256(payload)


def test_payload_and_fingerprints_are_deterministic_and_sensitive() -> None:
    first, _model, _source, _candidate = _build()
    second, _model, _source, _candidate = _build(
        candidate_tokenizer=_CandidateTokenizer(reverse_vocabulary=True)
    )
    changed, _model, _source, _candidate = _build(
        source_model=_SourceModel(continuation_id=51)
    )

    assert first == second
    assert first["payload_sha256"] != changed["payload_sha256"]
    assert (
        first["provenance_sha256"]["stepwise_replay_bank"]
        != (changed["provenance_sha256"]["stepwise_replay_bank"])
    )
    assert (
        first["provenance_sha256"]["on_policy_replay_bank"]
        != (changed["provenance_sha256"]["on_policy_replay_bank"])
    )

    baseline = replay.tokenizer_functional_fingerprint(_CandidateTokenizer())
    reverse = replay.tokenizer_functional_fingerprint(
        _CandidateTokenizer(reverse_vocabulary=True)
    )
    backend_changed = replay.tokenizer_functional_fingerprint(
        _CandidateTokenizer(backend_identity="candidate-v2")
    )
    specials_changed = replay.tokenizer_functional_fingerprint(
        _CandidateTokenizer(eos_token_id=4)
    )
    assert baseline == reverse
    assert baseline["class"].endswith("._CandidateTokenizer")
    assert baseline["full_vocabulary_stream_sha256"]
    assert baseline["backend_tokenizer_json_sha256"]
    assert baseline["special_token_ids"]["eos_token_id"] == 1
    assert baseline["sha256"] != backend_changed["sha256"]
    assert baseline["sha256"] != specials_changed["sha256"]


def test_same_config_different_source_weights_change_provenance() -> None:
    baseline_model = _SourceModel()
    changed_model = _SourceModel()
    with torch.no_grad():
        changed_model.marker.fill_(9.5)

    baseline, _model, _source, _candidate = _build(source_model=baseline_model)
    changed, _model, _source, _candidate = _build(source_model=changed_model)

    assert baseline_model.config.to_dict() == changed_model.config.to_dict()
    assert baseline["stepwise_replay_bank"] == changed["stepwise_replay_bank"]
    assert baseline["on_policy_replay_bank"] == changed["on_policy_replay_bank"]
    assert (
        baseline["precomputed_teacher_generated_texts"]
        == (changed["precomputed_teacher_generated_texts"])
    )
    baseline_state = baseline["source_model_fingerprint"]["model_state_fingerprint"]
    changed_state = changed["source_model_fingerprint"]["model_state_fingerprint"]
    assert baseline_state["state_dict_sha256"] != changed_state["state_dict_sha256"]
    assert (
        baseline["provenance_sha256"]["source_model_state"]
        != (changed["provenance_sha256"]["source_model_state"])
    )
    assert (
        baseline["provenance_sha256"]["source_model"]
        != (changed["provenance_sha256"]["source_model"])
    )
    assert baseline["payload_sha256"] != changed["payload_sha256"]


def test_teacher_text_bank_uses_all_prompts_own_budget_and_recovery_dedup() -> None:
    calibration_texts = ("d", "c", "d", "e")
    payload, model, source, candidate = _build(
        calibration_texts=calibration_texts,
        steps=1,
        batch_size=2,
        on_policy_max_new_tokens=1,
        teacher_max_new_tokens=3,
    )

    # The scheduled on-policy event sees only its own first batch. The teacher
    # LM bank independently generates over every calibration prompt, then uses
    # recovery's source-decoded, first-occurrence deduplication semantics.
    assert payload["on_policy_replay_bank"] == [["dalibration", "calibration"]]
    assert payload["precomputed_teacher_generated_texts"] == [
        "dalibration",
        "calibration",
        "ealibration",
    ]
    assert [call["batch_size"] for call in model.calls] == [2, 2, 2, 2]
    assert [call["max_new_tokens"] for call in model.calls] == [1, 1, 3, 3]
    assert model.calls[-2]["input_ids"] == [[2, 31, 32], [0, 2, 30]]
    assert model.calls[-1]["input_ids"] == [[2, 31, 32], [0, 2, 33]]

    expected_model = _SourceModel()
    expected_source = _SourceTokenizer()
    expected = hf_recovery._teacher_generated_lm_texts(
        torch,
        expected_model,
        expected_source,
        calibration_texts,
        "none",
        3,
        8,
        2,
        "cpu",
    )
    assert payload["precomputed_teacher_generated_texts"] == expected
    assert source.padding_side == "right"
    assert candidate.padding_side == "right"
    assert expected_source.padding_side == "right"
    assert expected_model.training is True


def test_domain_balanced_order_matches_recovery(monkeypatch) -> None:
    domains = {"c": "general", "d": "math", "e": "general"}
    monkeypatch.setattr(
        replay.hf_recovery,
        "_recovery_text_domain",
        lambda text: domains[text],
    )
    original = ("c", "e", "d")
    expected = hf_recovery._order_recovery_texts(original, "domain-balanced")

    payload, _model, _source, _candidate = _build(
        calibration_texts=original,
        calibration_order="domain-balanced",
        steps=3,
        batch_size=1,
    )

    assert expected == ["c", "d", "e"]
    assert payload["calibration_texts"] == expected
    assert payload["parameters"]["calibration_order"] == "domain-balanced"
    assert [
        event["rows"][0]["prompt_input_ids"]
        for event in payload["stepwise_replay_bank"]
    ] == [[2, 10], [2, 11, 12], [2, 13]]
    assert payload["precomputed_teacher_generated_texts"] == [
        "calibration",
        "dalibration",
        "ealibration",
    ]
    assert tuple(
        replay.verify_teacher_token_replay_payload(payload)["calibration_texts"]
    ) == tuple(expected)


def test_model_free_verifier_returns_deeply_immutable_normalized_json() -> None:
    payload, _model, _source, _candidate = _build()

    verified = replay.verify_teacher_token_replay_payload(payload)

    assert isinstance(verified, MappingProxyType)
    assert isinstance(verified["parameters"], MappingProxyType)
    assert isinstance(verified["stepwise_replay_bank"], tuple)
    assert isinstance(verified["stepwise_replay_bank"][0], MappingProxyType)
    assert isinstance(verified["stepwise_replay_bank"][0]["rows"], tuple)
    assert verified["payload_sha256"] == payload["payload_sha256"]
    assert (
        replay.verify_teacher_token_replay_payload(verified)["payload_sha256"]
        == payload["payload_sha256"]
    )
    with pytest.raises(TypeError):
        verified["status"] = "tampered"
    with pytest.raises(TypeError):
        verified["parameters"]["steps"] = 99
    with pytest.raises(TypeError):
        verified["stepwise_replay_bank"][0]["rows"][0]["rollout_input_ids"][0] = 7


def test_model_free_verifier_rejects_exact_fields_self_hashes_and_content_drift() -> (
    None
):
    payload, _model, _source, _candidate = _build()

    extra = _json_clone(payload)
    extra["unexpected"] = True
    with pytest.raises(replay.TeacherTokenReplayIntegrityError, match="exactly"):
        replay.verify_teacher_token_replay_payload(extra)

    top_hash = _json_clone(payload)
    top_hash["on_policy_replay_bank"][0][0] = "tampered source text"
    with pytest.raises(replay.TeacherTokenReplayIntegrityError, match="self-hash"):
        replay.verify_teacher_token_replay_payload(top_hash)

    tokenizer_hash = _json_clone(payload)
    tokenizer_hash["tokenizer_fingerprints"]["source"]["class"] = "tampered"
    _reseal_payload(tokenizer_hash)
    with pytest.raises(
        replay.TeacherTokenReplayIntegrityError,
        match="tokenizer_fingerprints.source self-hash",
    ):
        replay.verify_teacher_token_replay_payload(tokenizer_hash)

    model_state_hash = _json_clone(payload)
    model_state_hash["source_model_fingerprint"]["model_state_fingerprint"][
        "state_dict_sha256"
    ] = "0" * 64
    model_fingerprint_body = dict(model_state_hash["source_model_fingerprint"])
    model_fingerprint_body.pop("sha256")
    model_state_hash["source_model_fingerprint"]["sha256"] = replay._canonical_sha256(
        model_fingerprint_body
    )
    _reseal_payload(model_state_hash)
    with pytest.raises(
        replay.TeacherTokenReplayIntegrityError,
        match="model-state fingerprint self-hash",
    ):
        replay.verify_teacher_token_replay_payload(model_state_hash)

    schedule = _json_clone(payload)
    schedule["stepwise_scheduled_steps"] = [1, 3, 2]
    _reseal_payload(schedule)
    with pytest.raises(
        replay.TeacherTokenReplayIntegrityError, match="parameters.steps"
    ):
        replay.verify_teacher_token_replay_payload(schedule)

    bank = _json_clone(payload)
    bank["stepwise_replay_bank"][0]["rows"][0]["rollout_input_ids"][-1] = 21
    _reseal_payload(bank)
    with pytest.raises(
        replay.TeacherTokenReplayIntegrityError, match="provenance hashes"
    ):
        replay.verify_teacher_token_replay_payload(bank)

    blank = _json_clone(payload)
    blank["precomputed_teacher_generated_texts"][0] = " "
    _reseal_payload(blank)
    with pytest.raises(replay.TeacherTokenReplayIntegrityError, match="malformed"):
        replay.verify_teacher_token_replay_payload(blank)


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("source-empty", "no continuation"),
        ("source-blank", "blank text"),
        ("candidate-empty", "at least one token"),
        ("candidate-invalid-vocabulary", "outside the candidate vocabulary"),
        ("candidate-too-many", "exceeds max_new_tokens"),
        ("candidate-too-long", "exceeds the exact runtime prompt maximum"),
    ],
)
def test_failures_reject_unsafe_replay_without_state_drift(
    failure: str, match: str
) -> None:
    source_model = _SourceModel(
        continuation_id=52 if failure == "source-blank" else 50,
        emit_continuation=failure != "source-empty",
    )
    source_model.train(False)
    source = _SourceTokenizer()
    candidate = _CandidateTokenizer(
        continuation_mode={
            "candidate-empty": "empty",
            "candidate-invalid-vocabulary": "invalid-vocabulary",
            "candidate-too-many": "too-many",
        }.get(failure, "normal"),
        ignore_prompt_truncation=failure == "candidate-too-long",
        long_prompt=failure == "candidate-too-long",
    )
    before = {
        name: value.detach().clone()
        for name, value in source_model.state_dict().items()
    }

    with pytest.raises(ValueError, match=match):
        _build(
            source_model=source_model,
            source_tokenizer=source,
            candidate_tokenizer=candidate,
            calibration_texts=("c",),
            steps=1,
            batch_size=1,
            max_length=6 if failure == "candidate-too-long" else 8,
            stepwise_max_new_tokens=2 if failure == "candidate-too-long" else 1,
            on_policy_max_new_tokens=1,
        )

    assert source_model.training is False
    assert source.padding_side == "right"
    assert candidate.padding_side == "right"
    assert candidate.generate_calls == 0
    assert all(
        torch.equal(value, before[name])
        for name, value in source_model.state_dict().items()
    )


def test_teacher_bank_failure_restores_model_and_both_tokenizers() -> None:
    source_model = _SourceModel(fail_on_max_new_tokens=3)
    source_model.train(False)
    source = _SourceTokenizer()
    candidate = _CandidateTokenizer()
    before = {
        name: value.detach().clone()
        for name, value in source_model.state_dict().items()
    }

    with pytest.raises(RuntimeError, match="deliberate teacher generation failure"):
        _build(
            source_model=source_model,
            source_tokenizer=source,
            candidate_tokenizer=candidate,
            calibration_texts=("c", "d", "e"),
            steps=1,
            batch_size=2,
            on_policy_max_new_tokens=1,
            teacher_max_new_tokens=3,
        )

    assert [call["max_new_tokens"] for call in source_model.calls] == [1, 1, 3]
    assert source_model.training is False
    assert source.padding_side == "right"
    assert candidate.padding_side == "right"
    assert candidate.generate_calls == 0
    assert all(
        torch.equal(value, before[name])
        for name, value in source_model.state_dict().items()
    )


def test_blank_calibration_is_rejected_before_generation() -> None:
    model = _SourceModel()
    with pytest.raises(ValueError, match="must not be blank"):
        _build(source_model=model, calibration_texts=(" ",))
    assert model.calls == []
