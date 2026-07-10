from __future__ import annotations

import hashlib

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_canonical_graph import (  # noqa: E402
    CanonicalChildSpec,
    CanonicalGraphError,
    CanonicalParentLogits,
    CanonicalTokenContract,
    build_canonical_execution_graph,
)
from neural_forge.hf_canonical_kernel import (  # noqa: E402
    CanonicalParentTrace,
    CanonicalRowIdentity,
    build_proportional_depth_cover,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(name: str, split: str, rows: int) -> CanonicalRowIdentity:
    return CanonicalRowIdentity(
        split_name=split,
        pooling="last_valid",
        row_ids=tuple(f"{split}-{index}" for index in range(rows)),
        raw_prompt_sha256=tuple(
            _digest(f"{split}-prompt-{index}") for index in range(rows)
        ),
        corpus_sha256=_digest(f"{split}-corpus"),
        tokenizer_fingerprint=_digest(f"{name}-tokenizer"),
        cache_fingerprint=_digest(f"{name}-{split}-cache"),
    )


def _states(seed: int, rows: int, width: int, depth: int):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(rows, width, generator=generator, dtype=torch.float64)
    return tuple(
        base + 0.1 * layer * torch.sin(base * (layer + 1))
        for layer in range(depth + 1)
    )


def _parent(
    name: str,
    *,
    seed: int,
    rows: int,
    width: int,
    depth: int,
    target_depth: int,
    split: str,
    weight: float,
) -> CanonicalParentTrace:
    return CanonicalParentTrace(
        name=name,
        layer_states=_states(seed, rows, width, depth),
        row_identity=_identity(name, split, rows),
        depth_cover=build_proportional_depth_cover(depth, target_depth),
        weight=weight,
    )


def _child() -> CanonicalChildSpec:
    return CanonicalChildSpec(
        family="blank-llama",
        layer_count=2,
        hidden_size=4,
        parameter_budget=12345,
        vocabulary_size=5,
        tokenizer_fingerprint=_digest("child-tokenizer"),
        tie_embeddings=True,
        position_policy="rope",
        architecture_fingerprint=_digest("blank-llama-shape"),
    )


def _contract(name: str, *, child_hash: str | None = None):
    return CanonicalTokenContract(
        parent_name=name,
        parent_tokenizer_fingerprint=_digest(f"{name}-tokenizer"),
        child_tokenizer_fingerprint=child_hash or _digest("child-tokenizer"),
        method="byte-span-likelihood",
        mapping_fingerprint=_digest(f"{name}-to-child-map"),
    )


def _fixture():
    child = _child()
    calibration = (
        _parent(
            "gpt",
            seed=11,
            rows=12,
            width=6,
            depth=3,
            target_depth=2,
            split="calibration",
            weight=0.25,
        ),
        _parent(
            "llama",
            seed=12,
            rows=12,
            width=8,
            depth=4,
            target_depth=2,
            split="calibration",
            weight=0.75,
        ),
    )
    validation = (
        _parent(
            "gpt",
            seed=13,
            rows=7,
            width=6,
            depth=3,
            target_depth=2,
            split="validation",
            weight=0.25,
        ),
        _parent(
            "llama",
            seed=14,
            rows=7,
            width=8,
            depth=4,
            target_depth=2,
            split="validation",
            weight=0.75,
        ),
    )
    generator = torch.Generator().manual_seed(15)
    gpt_cal = torch.randn(12, 5, generator=generator, dtype=torch.float64)
    llama_cal = torch.randn(12, 5, generator=generator, dtype=torch.float64)
    gpt_val = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    llama_val = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    logits = (
        CanonicalParentLogits(
            "gpt", gpt_cal, gpt_val, _contract("gpt")
        ),
        CanonicalParentLogits(
            "llama", llama_cal, llama_val, _contract("llama")
        ),
    )
    return child, calibration, validation, logits


def test_multi_parent_graph_is_simultaneous_complete_and_order_invariant() -> None:
    child, calibration, validation, logits = _fixture()
    first = build_canonical_execution_graph(
        calibration,
        validation,
        logits,
        child_spec=child,
        ridge=1e-8,
    )
    reversed_graph = build_canonical_execution_graph(
        tuple(reversed(calibration)),
        tuple(reversed(validation)),
        tuple(reversed(logits)),
        child_spec=child,
        ridge=1e-8,
    )

    assert first.fingerprint == reversed_graph.fingerprint
    assert first.boundary_splits.calibration_plan.parent_names == ("gpt", "llama")
    assert len(first.calibration_boundaries) == 3
    assert len(first.edges) == 2
    assert first.edges[0].calibration_input_sha256 != (
        first.edges[0].calibration_output_sha256
    )
    assert all(
        tuple(name for name, _ in edge.parent_source_spans) == ("gpt", "llama")
        for edge in first.edges
    )
    torch.testing.assert_close(
        first.calibration_target_log_probs,
        reversed_graph.calibration_target_log_probs,
        atol=0.0,
        rtol=0.0,
    )
    report = first.report()
    assert report["boundary_zero_consumed"] is True
    assert report["single_authoritative_parent"] is False
    assert report["parent_application"] == "simultaneous-not-sequential"
    assert report["training_steps"] == 0
    assert report["gradients_used"] is False
    assert all(edge["all_source_spans_represented"] for edge in report["edges"])


def test_logits_are_weighted_probability_mixture_not_weight_average() -> None:
    child, calibration, validation, logits = _fixture()
    graph = build_canonical_execution_graph(
        calibration,
        validation,
        logits,
        child_spec=child,
        ridge=1e-8,
    )
    gpt = torch.log_softmax(logits[0].calibration_logits, dim=-1)
    llama = torch.log_softmax(logits[1].calibration_logits, dim=-1)
    expected = torch.logsumexp(
        torch.stack(
            (
                gpt + torch.log(torch.tensor(0.25, dtype=torch.float64)),
                llama + torch.log(torch.tensor(0.75, dtype=torch.float64)),
            )
        ),
        dim=0,
    )
    expected = expected - torch.logsumexp(expected, dim=-1, keepdim=True)

    torch.testing.assert_close(
        graph.calibration_target_log_probs,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        graph.calibration_target_log_probs.exp().sum(dim=-1),
        torch.ones(12, dtype=torch.float64),
    )


def test_graph_rejects_missing_or_inexact_token_coverage() -> None:
    child, calibration, validation, logits = _fixture()
    with pytest.raises(CanonicalGraphError, match="exactly one aligned logit"):
        build_canonical_execution_graph(
            calibration,
            validation,
            logits[:1],
            child_spec=child,
        )

    bad_contract = CanonicalTokenContract(
        parent_name="gpt",
        parent_tokenizer_fingerprint=_digest("gpt-tokenizer"),
        child_tokenizer_fingerprint=_digest("child-tokenizer"),
        method="byte-span-likelihood",
        mapping_fingerprint=_digest("partial"),
        coverage_fraction=0.9,
        exact=False,
    )
    bad_logits = (
        CanonicalParentLogits(
            "gpt",
            logits[0].calibration_logits,
            logits[0].validation_logits,
            bad_contract,
        ),
        logits[1],
    )
    with pytest.raises(CanonicalGraphError, match="exact full child-token coverage"):
        build_canonical_execution_graph(
            calibration,
            validation,
            bad_logits,
            child_spec=child,
        )


def test_graph_rejects_wrong_logit_rows_and_child_tokenizer() -> None:
    child, calibration, validation, logits = _fixture()
    wrong_rows = (
        CanonicalParentLogits(
            "gpt",
            logits[0].calibration_logits[:-1],
            logits[0].validation_logits,
            logits[0].token_contract,
        ),
        logits[1],
    )
    with pytest.raises(CanonicalGraphError, match=r"shape \(12, 5\)"):
        build_canonical_execution_graph(
            calibration,
            validation,
            wrong_rows,
            child_spec=child,
        )

    wrong_contract = _contract("gpt", child_hash=_digest("other-child"))
    wrong_child = (
        CanonicalParentLogits(
            "gpt",
            logits[0].calibration_logits,
            logits[0].validation_logits,
            wrong_contract,
        ),
        logits[1],
    )
    with pytest.raises(CanonicalGraphError, match="different child tokenizer"):
        build_canonical_execution_graph(
            calibration,
            validation,
            wrong_child,
            child_spec=child,
        )
