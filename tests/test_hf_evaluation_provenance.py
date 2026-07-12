from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_evaluation_provenance import (  # noqa: E402
    HFEvaluationCountError,
    HFEvaluationPathOverlapError,
    assert_evaluation_report_count_contract,
    assert_strict_evaluation_path_disjointness,
    bind_loaded_text_evaluation_provenance,
    bind_prompt_path_domain,
)
from neural_forge.hf_tools import evaluate_loaded_text_model  # noqa: E402


class _Batch(dict[str, torch.Tensor]):
    def to(self, device: str) -> "_Batch":
        return _Batch({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "right"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any], str]] = []

    @staticmethod
    def _encode(text: str, add_special_tokens: bool = True) -> list[int]:
        # Boundary-sensitive and deterministic: the exact concatenated choice
        # string matters, just as it does for a real subword tokenizer.
        values = [3 + (ord(character) % 89) for character in text]
        return ([1] + values) if add_special_tokens else values

    def __call__(
        self,
        value: str | list[str],
        *,
        return_tensors: str,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> _Batch:
        self.calls.append(
            (
                value if isinstance(value, str) else tuple(value),
                {
                    "return_tensors": return_tensors,
                    "padding": padding,
                    "truncation": truncation,
                    "max_length": max_length,
                    "add_special_tokens": add_special_tokens,
                },
                self.padding_side,
            )
        )
        texts = [value] if isinstance(value, str) else list(value)
        rows = [self._encode(text, add_special_tokens=add_special_tokens) for text in texts]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        width = max(len(row) for row in rows)
        if not padding and len({len(row) for row in rows}) > 1:
            raise AssertionError("ragged batch without padding")
        ids: list[list[int]] = []
        masks: list[list[int]] = []
        for row in rows:
            pad = width - len(row)
            if self.padding_side == "left":
                ids.append(([self.pad_token_id] * pad) + row)
                masks.append(([0] * pad) + ([1] * len(row)))
            else:
                ids.append(row + ([self.pad_token_id] * pad))
                masks.append(([1] * len(row)) + ([0] * pad))
        return _Batch(
            {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        )

    @staticmethod
    def decode(token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "decoded " + " ".join(str(int(item)) for item in token_ids)


class _CaptureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.forward_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.generate_calls: list[dict[str, torch.Tensor]] = []

    def forward(
        self, *, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> SimpleNamespace:
        self.forward_calls.append((input_ids.detach().cpu(), labels.detach().cpu()))
        return SimpleNamespace(loss=self.weight * 0.0 + torch.tensor(0.25))

    def generate(self, **inputs: Any) -> torch.Tensor:
        self.generate_calls.append(
            {
                key: value.detach().cpu()
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
        )
        input_ids = inputs["input_ids"]
        suffix = torch.full(
            (int(input_ids.shape[0]), int(inputs["max_new_tokens"])),
            7,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, suffix), dim=1)


def _suite(*, generation_count: int = 2) -> dict[str, Any]:
    return {
        "source": "exact-unit-suite",
        "text_evals": ["ppl text"],
        "choice_probes": [
            {
                "id": "choice-1",
                "category": "math",
                "prompt": "Q:",
                "choices": [" A", " B"],
                "answer": 0,
            }
        ],
        "generation_prompts": [
            {
                "id": f"generation-{index}",
                "category": "code",
                "prompt": "shared prompt" if index == 0 else f"generation {index}",
                "relevance_rules": [],
            }
            for index in range(generation_count)
        ],
        "behavior_probes": [
            {
                "id": "behavior-1",
                "category": "reasoning",
                "prompt": "shared prompt",
                "kind": "pattern_groups",
                "pattern_groups": [],
            }
        ],
    }


def _report_for(provenance: Any) -> dict[str, Any]:
    contract = provenance.count_contract
    suite = _suite()
    choice_items = []
    for probe in suite["choice_probes"][: contract.choice_items]:
        scores = [float(len(probe["choices"]) - index) for index in range(len(probe["choices"]))]
        prediction = max(range(len(scores)), key=lambda index: scores[index])
        answer = int(probe["answer"])
        choice_items.append(
            {
                "id": probe["id"],
                "prediction": prediction,
                "answer": answer,
                "correct": prediction == answer,
                "scores": scores,
                "choices": list(probe["choices"]),
            }
        )
    correct = sum(int(item["correct"]) for item in choice_items)
    generation_items = [
        {"id": item["id"], "prompt": item["prompt"]}
        for item in suite["generation_prompts"][: contract.generation_items]
    ]
    generation_diagnostics = [
        {"id": item["id"], "prompt": item["prompt"], "passed": True}
        for item in suite["generation_prompts"][: contract.generation_items]
    ]
    behavior_items = [
        {"id": item["id"], "prompt": item["prompt"], "passed": True}
        for item in suite["behavior_probes"][: contract.behavior_items]
    ]
    return {
        "identity_binding": provenance.evaluation_identity_binding(),
        "eval_suite": {
            "source": provenance.suite_source,
            "text_eval_count": contract.perplexity_items,
            "choice_probe_count": contract.choice_items,
            "generation_prompt_count": contract.generation_items,
            "behavior_probe_count": contract.behavior_items,
        },
        "perplexity": {"items": [1.0] * contract.perplexity_items},
        "choice_accuracy": {
            "correct": correct,
            "total": len(choice_items),
            "accuracy": correct / max(1, len(choice_items)),
            "items": choice_items,
        },
        "generations": generation_items,
        "generation_diagnostics": generation_diagnostics,
        "generation_health": {
            "correct": len(generation_diagnostics),
            "total": len(generation_diagnostics),
            "pass_rate": len(generation_diagnostics)
            / max(1, len(generation_diagnostics)),
        },
        "behavior_suite": {
            "correct": len(behavior_items),
            "total": len(behavior_items),
            "pass_rate": len(behavior_items) / max(1, len(behavior_items)),
            "items": behavior_items,
        },
    }


def test_binding_matches_literal_evaluator_tokenizer_and_label_paths(
    tmp_path: Any,
) -> None:
    suite = _suite()
    bound_tokenizer = _Tokenizer()
    provenance = bind_loaded_text_evaluation_provenance(
        bound_tokenizer,
        suite,
        max_new_tokens=2,
    )
    evaluated_tokenizer = _Tokenizer()
    model = _CaptureModel()
    report = evaluate_loaded_text_model(
        model,
        evaluated_tokenizer,
        tmp_path / "evaluation.json",
        device="cpu",
        max_new_tokens=2,
        eval_suite=suite,
        identity_binding=provenance.evaluation_identity_binding(),
    )

    # The helper makes the same calls, in the same order, with evaluator
    # defaults (no truncation/max_length) and restores generation padding side.
    assert bound_tokenizer.calls == evaluated_tokenizer.calls
    assert bound_tokenizer.padding_side == "right"
    assert provenance.tokenizer_truncation == "omitted"
    assert provenance.tokenizer_max_length == "omitted"

    ppl = provenance.scored_items[0]
    choice = provenance.scored_items[1]
    generation = provenance.scored_items[2]
    behavior = provenance.scored_items[-1]
    assert ppl.domain == "perplexity"
    assert ppl.input_token_count == int(model.forward_calls[0][0].shape[1])
    assert ppl.scored_label_count == ppl.input_token_count - 1
    assert torch.equal(model.forward_calls[0][0], model.forward_calls[0][1])

    assert choice.domain == "choice"
    assert choice.category == "math"
    assert len(choice.choice_sequences) == 2
    prompt_length = len(_Tokenizer._encode("Q:"))
    assert all(
        item.label_mask_boundary == prompt_length
        for item in choice.choice_sequences
    )
    for offset, sequence in enumerate(choice.choice_sequences, start=1):
        observed_labels = model.forward_calls[offset][1]
        assert torch.all(observed_labels[:, :prompt_length] == -100)
        assert sequence.loss_scale_token_count == int(
            (observed_labels != -100).sum().item()
        )
        assert sequence.scored_label_count == int(
            (observed_labels[:, 1:] != -100).sum().item()
        )

    # Generation and behavior share one cached model-input path when their
    # prompts are identical, but remain separate immutable metric records.
    assert generation.domain == "generation"
    assert behavior.domain == "behavior"
    assert provenance.generation_do_sample is False
    assert provenance.generation_pad_token_id == 0
    assert provenance.generation_batch_size == 8
    assert generation.input_sha256 == behavior.input_sha256
    assert generation.exact_path_sha256s == behavior.exact_path_sha256s
    assert len(model.generate_calls) == 1
    assert len(provenance.scored_items) == 5
    verification = assert_evaluation_report_count_contract(provenance, report)
    assert verification.observed_total == 5


def test_generation_batching_is_exactly_eight_and_records_every_metric_item() -> None:
    tokenizer = _Tokenizer()
    provenance = bind_loaded_text_evaluation_provenance(
        tokenizer,
        _suite(generation_count=10),
        max_new_tokens=3,
    )
    generation_calls = [
        call
        for call in tokenizer.calls
        if isinstance(call[0], tuple)
    ]
    assert [len(call[0]) for call in generation_calls] == [8, 2]
    assert all(call[1]["padding"] is True for call in generation_calls)
    assert all(call[2] == "left" for call in generation_calls)
    generation_items = [
        item for item in provenance.scored_items if item.domain == "generation"
    ]
    assert len(generation_items) == 10
    assert [item.generation_batch_index for item in generation_items] == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
    ]
    assert provenance.count_contract.choice_scoring_sequences == 2
    assert provenance.count_contract.total_metric_items == 13


def test_limits_normalization_determinism_and_records_are_immutable() -> None:
    suite = _suite(generation_count=3)
    first = bind_loaded_text_evaluation_provenance(
        _Tokenizer(),
        suite,
        max_text_evals=1,
        max_choice_probes=1,
        max_generation_prompts=1,
        max_behavior_probes=0,
    )
    second = bind_loaded_text_evaluation_provenance(
        _Tokenizer(),
        suite,
        max_text_evals=1,
        max_choice_probes=1,
        max_generation_prompts=1,
        max_behavior_probes=0,
    )
    assert first.aggregate_sha256 == second.aggregate_sha256
    assert first.scored_items_sha256 == second.scored_items_sha256
    assert first.count_contract.total_metric_items == 3
    with pytest.raises(FrozenInstanceError):
        first.scored_items[0].category = "changed"  # type: ignore[misc]

    changed = _suite(generation_count=3)
    changed["choice_probes"][0]["choices"][1] = " C"
    third = bind_loaded_text_evaluation_provenance(
        _Tokenizer(),
        changed,
        max_text_evals=1,
        max_choice_probes=1,
        max_generation_prompts=1,
        max_behavior_probes=0,
    )
    assert third.aggregate_sha256 != first.aggregate_sha256


def test_strict_four_domain_disjointness_passes_and_binds_six_pairs() -> None:
    tokenizer = _Tokenizer()
    calibration = bind_prompt_path_domain(
        tokenizer, ["calibration only"], name="collector-a", max_length=128
    )
    sensitivity = bind_prompt_path_domain(
        tokenizer, ["sensitivity only"], name="collector-b", max_length=128
    )
    candidate = bind_loaded_text_evaluation_provenance(
        tokenizer,
        {
            "text_evals": ["candidate ppl"],
            "choice_probes": [],
            "generation_prompts": [],
            "behavior_probes": [],
        },
    )
    transaction = bind_loaded_text_evaluation_provenance(
        tokenizer,
        {
            "text_evals": ["transaction ppl"],
            "choice_probes": [],
            "generation_prompts": [],
            "behavior_probes": [],
        },
    )
    result = assert_strict_evaluation_path_disjointness(
        calibration=calibration,
        sensitivity=sensitivity,
        candidate=candidate,
        transaction=transaction,
    )
    assert result.all_paths_disjoint is True
    assert len(result.pairwise_checks) == 6
    assert [item.name for item in result.domains] == [
        "calibration",
        "sensitivity",
        "candidate",
        "transaction",
    ]


@pytest.mark.parametrize("overlap_domain", ["candidate", "transaction"])
def test_strict_disjointness_detects_same_active_tokens_across_different_calls(
    overlap_domain: str,
) -> None:
    tokenizer = _Tokenizer()
    shared = "same path"
    calibration = bind_prompt_path_domain(
        tokenizer, [shared], name="calibration-source", max_length=128
    )
    sensitivity = bind_prompt_path_domain(
        tokenizer, ["sensitivity only"], name="sensitivity-source", max_length=128
    )

    def evaluation(text: str) -> Any:
        return bind_loaded_text_evaluation_provenance(
            tokenizer,
            {
                "text_evals": [text],
                "choice_probes": [],
                "generation_prompts": [],
                "behavior_probes": [],
            },
        )

    candidate = evaluation(shared if overlap_domain == "candidate" else "candidate")
    transaction = evaluation(
        shared if overlap_domain == "transaction" else "transaction"
    )
    with pytest.raises(HFEvaluationPathOverlapError, match="token_content_overlap_count"):
        assert_strict_evaluation_path_disjointness(
            calibration=calibration,
            sensitivity=sensitivity,
            candidate=candidate,
            transaction=transaction,
        )


def test_truncated_collector_prefix_cannot_leak_into_untruncated_evaluation() -> None:
    tokenizer = _Tokenizer()
    collector_prompt = "abcdefghijklmno"
    evaluation_prompt = collector_prompt + " has an untruncated held-out suffix"
    calibration = bind_prompt_path_domain(
        tokenizer,
        [collector_prompt],
        name="calibration-source",
        max_length=10,
    )
    sensitivity = bind_prompt_path_domain(
        tokenizer,
        ["completely separate sensitivity"],
        name="sensitivity-source",
        max_length=128,
    )
    candidate = bind_loaded_text_evaluation_provenance(
        tokenizer,
        {
            "text_evals": [evaluation_prompt],
            "choice_probes": [],
            "generation_prompts": [],
            "behavior_probes": [],
        },
    )
    transaction = bind_loaded_text_evaluation_provenance(
        tokenizer,
        {
            "text_evals": ["independent transaction"],
            "choice_probes": [],
            "generation_prompts": [],
            "behavior_probes": [],
        },
    )

    # Raw texts and full active-token hashes differ.  The meaningful prefix
    # binding is what exposes the truncated/untruncated data reuse.
    assert not (
        set(calibration.raw_source_text_sha256s)
        & set(candidate.to_path_domain("candidate").raw_source_text_sha256s)
    )
    assert not (
        set(calibration.token_content_sha256s)
        & set(candidate.to_path_domain("candidate").token_content_sha256s)
    )
    assert (
        set(calibration.token_content_sha256s)
        & set(candidate.to_path_domain("candidate").token_prefix_sha256s)
    )
    with pytest.raises(HFEvaluationPathOverlapError, match="token_prefix_overlap_count"):
        assert_strict_evaluation_path_disjointness(
            calibration=calibration,
            sensitivity=sensitivity,
            candidate=candidate,
            transaction=transaction,
        )


def test_shared_boilerplate_prefixes_do_not_make_independent_paths_overlap() -> None:
    tokenizer = _Tokenizer()
    calibration = bind_prompt_path_domain(
        tokenizer,
        ["shared boilerplate calibration unique ending"],
        name="calibration-source",
        max_length=128,
    )
    sensitivity = bind_prompt_path_domain(
        tokenizer,
        ["shared boilerplate sensitivity unique ending"],
        name="sensitivity-source",
        max_length=128,
    )

    def evaluation(text: str) -> Any:
        return bind_loaded_text_evaluation_provenance(
            tokenizer,
            {
                "text_evals": [text],
                "choice_probes": [],
                "generation_prompts": [],
                "behavior_probes": [],
            },
        )

    candidate = evaluation("shared boilerplate candidate unique ending")
    transaction = evaluation("shared boilerplate transaction unique ending")
    # The domains share many >=8-token prefixes, but neither complete token
    # path is a prefix of another complete path.
    assert (
        set(calibration.token_prefix_sha256s)
        & set(candidate.to_path_domain("candidate").token_prefix_sha256s)
    )
    report = assert_strict_evaluation_path_disjointness(
        calibration=calibration,
        sensitivity=sensitivity,
        candidate=candidate,
        transaction=transaction,
    )
    assert report.all_paths_disjoint is True
    assert all(
        check[-2:] == (0, 0)
        for check in report.pairwise_checks
    )


def test_count_contract_rejects_missing_items_and_reordered_ids() -> None:
    provenance = bind_loaded_text_evaluation_provenance(_Tokenizer(), _suite())
    report = _report_for(provenance)
    assert_evaluation_report_count_contract(provenance, report)

    missing = _report_for(provenance)
    missing["generations"].pop()
    with pytest.raises(HFEvaluationCountError, match="item-count contract"):
        assert_evaluation_report_count_contract(provenance, missing)

    wrong_id = _report_for(provenance)
    wrong_id["choice_accuracy"]["items"][0]["id"] = "wrong"
    with pytest.raises(HFEvaluationCountError, match="ordered item IDs"):
        assert_evaluation_report_count_contract(provenance, wrong_id)


def test_same_counts_and_ids_cannot_forge_evaluation_identity() -> None:
    provenance = bind_loaded_text_evaluation_provenance(_Tokenizer(), _suite())
    report = _report_for(provenance)

    counts_and_ids_only = copy.deepcopy(report)
    counts_and_ids_only.pop("identity_binding")
    with pytest.raises(HFEvaluationCountError, match="identity binding"):
        assert_evaluation_report_count_contract(provenance, counts_and_ids_only)

    forged_prompt = copy.deepcopy(report)
    forged_prompt["generations"][0]["prompt"] = "forged held-out prompt"
    with pytest.raises(HFEvaluationCountError, match="generation prompt changed"):
        assert_evaluation_report_count_contract(provenance, forged_prompt)

    forged_choice = copy.deepcopy(report)
    forged_choice["choice_accuracy"]["items"][0]["choices"][1] = " forged"
    with pytest.raises(HFEvaluationCountError, match="Choice text hashes changed"):
        assert_evaluation_report_count_contract(provenance, forged_choice)

    missing_score = copy.deepcopy(report)
    missing_score["choice_accuracy"]["items"][0]["scores"].pop()
    with pytest.raises(HFEvaluationCountError, match="Choice score count changed"):
        assert_evaluation_report_count_contract(provenance, missing_score)

    forged_generation_summary = copy.deepcopy(report)
    forged_generation_summary["generation_health"]["correct"] = 0
    with pytest.raises(HFEvaluationCountError, match="Generation summary counts"):
        assert_evaluation_report_count_contract(provenance, forged_generation_summary)

    forged_behavior_summary = copy.deepcopy(report)
    forged_behavior_summary["behavior_suite"]["pass_rate"] = 0.0
    with pytest.raises(HFEvaluationCountError, match="Behavior summary pass rate"):
        assert_evaluation_report_count_contract(provenance, forged_behavior_summary)

    forged_answer = copy.deepcopy(report)
    forged_answer["choice_accuracy"]["items"][0]["answer"] = 1
    with pytest.raises(HFEvaluationCountError, match="Choice answer changed"):
        assert_evaluation_report_count_contract(provenance, forged_answer)
