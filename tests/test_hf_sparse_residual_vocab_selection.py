from __future__ import annotations

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_factorized_vocab import (  # noqa: E402
    analyze_tied_vocabulary_modules,
)
from neural_forge.hf_sparse_residual_vocab import (  # noqa: E402
    SparseResidualVocabularyCore,
    build_sparse_residual_vocabulary_pair,
)
from neural_forge.hf_sparse_residual_vocab_selection import (  # noqa: E402
    SparseResidualSelectionError,
    VocabularyCalibrationCase,
    evaluate_sparse_residual_logit_validation,
    residual_output_energy,
    select_sparse_residual_token_rows,
    split_recovery_calibration_cases,
)


def _source():
    weight = torch.tensor(
        [
            [10.0, 0.0],
            [9.0, 0.0],
            [8.0, 0.0],
            [7.0, 0.0],
            [6.0, 0.0],
            [5.0, 0.0],
            [4.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    embedding = torch.nn.Embedding(8, 2, dtype=torch.float64)
    with torch.no_grad():
        embedding.weight.copy_(weight)
    head = torch.nn.Linear(2, 8, bias=False, dtype=torch.float64)
    head.weight = embedding.weight
    return embedding, head


def _cases(weight):
    hidden = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    logits = hidden @ weight.transpose(0, 1)
    provisional = []
    for category in ("math", "code"):
        for index in range(8):
            provisional.append(
                VocabularyCalibrationCase.create(
                    case_id=f"recovery_{category}_{index:05d}",
                    category=category,
                    prompt_token_ids=[0],
                    parent_output_token_ids=[7],
                    hidden_rows=hidden,
                    parent_logits=logits,
                    vocab_size=8,
                    hidden_size=2,
                )
            )
    split = split_recovery_calibration_cases(provisional)
    validation = set(split.validation_case_ids)
    cases = tuple(
        VocabularyCalibrationCase.create(
            case_id=case.case_id,
            category=case.category,
            prompt_token_ids=[6] if case.case_id in validation else [0],
            parent_output_token_ids=[7],
            hidden_rows=hidden,
            parent_logits=logits,
            vocab_size=8,
            hidden_size=2,
        )
        for case in provisional
    )
    return cases, split_recovery_calibration_cases(cases)


def test_fit_only_selection_beats_raw_rank_on_disjoint_validation() -> None:
    embedding, head = _source()
    analysis = analyze_tied_vocabulary_modules(embedding, head)
    base_pair = analysis.factorize(1)
    cases, split = _cases(embedding.weight.detach())
    fit_hidden = torch.cat(
        [
            case.hidden_rows
            for case in cases
            if case.case_id in set(split.fit_case_ids)
        ],
        dim=0,
    )
    energy = residual_output_energy(
        source_weight=embedding.weight,
        vocab_factor=base_pair.embedding.vocab_factor,
        hidden_factor=base_pair.embedding.hidden_factor,
        hidden_rows=fit_hidden,
        chunk_rows=3,
    )
    selection = select_sparse_residual_token_rows(
        cases,
        split,
        selected_row_budget=3,
        special_token_ids=[1],
        residual_energy=energy,
        top_k=4,
    )
    sparse = build_sparse_residual_vocabulary_pair(
        analysis,
        embedding,
        head,
        rank=1,
        selected_token_ids=torch.tensor(
            selection.selected_token_ids,
            dtype=torch.long,
        ),
    )
    raw = SparseResidualVocabularyCore(
        base_pair.embedding.vocab_factor,
        base_pair.embedding.hidden_factor,
        torch.tensor([0], dtype=torch.long),
        torch.zeros(1, 2, dtype=torch.float64),
    )

    raw_report = evaluate_sparse_residual_logit_validation(
        raw,
        cases,
        split,
        top_k=4,
    )
    sparse_report = evaluate_sparse_residual_logit_validation(
        sparse.core,
        cases,
        split,
        top_k=4,
    )

    assert split.as_dict()["case_disjoint"] is True
    assert len(split.fit_case_ids) == 12
    assert len(split.validation_case_ids) == 4
    assert selection.selected_token_ids == (0, 1, 7)
    assert 6 not in selection.mandatory_token_ids
    assert 6 not in selection.selected_token_ids
    assert energy[7] > energy[:7].max()
    assert raw_report["top1_agreement"] == 0.0
    assert sparse_report["top1_agreement"] == 1.0
    assert sparse_report["minimum_candidate_parent_winner_margin"] > 0.0


def test_split_tamper_and_manual_case_injection_fail_closed() -> None:
    embedding, _head = _source()
    cases, split = _cases(embedding.weight.detach())
    tampered = replace(
        split,
        validation_case_ids=split.validation_case_ids[:-1]
        + (split.fit_case_ids[0],),
    )

    with pytest.raises(SparseResidualSelectionError, match="modified"):
        select_sparse_residual_token_rows(
            cases,
            tampered,
            selected_row_budget=3,
            special_token_ids=[1],
            residual_energy=torch.ones(8),
            top_k=4,
        )
    with pytest.raises(SparseResidualSelectionError, match="recovery"):
        VocabularyCalibrationCase.create(
            case_id="manual_math_00",
            category="math",
            prompt_token_ids=[0],
            parent_output_token_ids=[7],
            hidden_rows=torch.tensor([[0.0, 2.0]]),
            parent_logits=torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]]
            ),
            vocab_size=8,
            hidden_size=2,
        )


def test_teacher_forced_ids_and_parent_logit_winners_remain_distinct() -> None:
    case = VocabularyCalibrationCase.create(
        case_id="recovery_math_00000",
        category="math",
        prompt_token_ids=[0],
        parent_output_token_ids=[6],
        hidden_rows=torch.tensor([[0.0, 2.0]]),
        parent_logits=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]]
        ),
        vocab_size=8,
        hidden_size=2,
    )

    descriptor = case.descriptor()

    assert case.parent_output_token_ids == (6,)
    assert descriptor["forced_vs_parent_top1_agreement_fraction"] == 0.0
    assert descriptor["teacher_forced_rows_allowed"] is True
