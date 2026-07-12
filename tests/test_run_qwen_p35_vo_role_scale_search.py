from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip(
        "Installed Transformers has no Qwen3 implementation.",
        allow_module_level=True,
    )

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qwen_p35_vo_role_scale_search as search  # noqa: E402
import run_qwen_residual_attention_probe as base  # noqa: E402
from neural_forge.hf_attention_codebook import (  # noqa: E402
    RaggedAttentionCodebookRegistry,
)
from neural_forge.hf_functional_decode_trace import (  # noqa: E402
    FunctionalDecodeLimits,
)
from neural_forge.hf_qwen_attention_surgery import (  # noqa: E402
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    prepare_qwen3_codebook_model,
)
from neural_forge.hf_residual_attention_role_scale import (  # noqa: E402
    ResidualAttentionRoleScaleError,
)


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self) -> int:
        return 24

    def get_vocab(self) -> dict[str, int]:
        return {f"token-{index}": index for index in range(len(self))}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"token-{token_id}"


def _runtime():
    torch.manual_seed(9898)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=24,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    ).eval()
    tokenizer = _Tokenizer()
    return prepare_qwen3_codebook_model(
        model,
        tokenizer,
        tokenizer,
        attention_prototypes_per_group=3,
        neuron_prototypes_per_neuron=3,
    )


def _limits() -> FunctionalDecodeLimits:
    return FunctionalDecodeLimits(
        max_horizon=2,
        max_prompt_tokens=16,
        max_vocabulary=32,
        max_logit_bytes=100_000,
        max_model_devices=1,
    )


def _run(runtime, pairs=()):
    return search.run_qwen_p35_vo_role_scale_search(
        runtime,
        (torch.tensor([[1, 4, 6]]),),
        (torch.tensor([[1, 5, 7]]),),
        pairs,
        redirect=base.AttentionRedirect(2, 0, 0, 0),
        require_exact_p35=False,
        horizon=2,
        top_k=5,
        max_candidates=5,
        max_abs_scale=4.0,
        max_prompts_per_split=1,
        trace_limits=_limits(),
    )


def _mapping_registry() -> RaggedAttentionCodebookRegistry:
    torch.manual_seed(4545)
    return RaggedAttentionCodebookRegistry(
        torch.randn(4, 2, 4),
        torch.randn(4, 2, 4),
        torch.randn(4, 2, 4),
        torch.randn(4, 4, 2),
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([[1, 0], [0, 1]], dtype=torch.int64),
        role_scales=torch.ones(2, 2, 4),
    )


def test_physical_scale_target_resolves_inverse_permutation_to_canonical_row() -> None:
    registry = _mapping_registry()

    assert search._canonical_assignment(
        registry,
        layer_index=0,
        physical_group=1,
    ) == 0
    assert search._canonical_assignment(
        registry,
        layer_index=0,
        physical_group=0,
    ) == 1


def test_candidate_order_is_input_order_independent_and_incumbent_is_first() -> None:
    incumbent = search.VOScalePair(-0.001249, -0.000539)
    pairs = (
        search.VOScalePair(1.0, 1.0),
        search.VOScalePair(0.5, -0.5),
        search.VOScalePair(1.0, 1.0),
    )
    forward = search._prepare_scale_candidates(
        pairs,
        incumbent,
        dtype=torch.float32,
        max_candidates=4,
        max_abs_scale=4.0,
    )
    reverse = search._prepare_scale_candidates(
        tuple(reversed(pairs)),
        incumbent,
        dtype=torch.float32,
        max_candidates=4,
        max_abs_scale=4.0,
    )

    assert [value.candidate_id for value in forward] == [
        value.candidate_id for value in reverse
    ]
    assert forward[0].tentative_incumbent is True
    assert sum(value.tentative_incumbent for value in forward) == 1
    assert len(forward) == 3


@dataclass(frozen=True)
class _Step:
    step_index: int
    selected_token_id: int
    greedy_token_id: int
    action_logits: object


@dataclass(frozen=True)
class _Trace:
    sequence_id: str
    lane: str
    prompt_token_ids: tuple[int, ...]
    consumed_token_ids: tuple[int, ...]
    selected_token_ids: tuple[int, ...]
    steps: tuple[_Step, ...]


def _row(top: int, *, vocab: int = 6) -> np.ndarray:
    row = np.linspace(-2.0, -3.0, vocab, dtype=np.float64)
    row[top] = 5.0
    return row


def _trace(
    lane: str,
    selected: tuple[int, ...],
    tops: tuple[int, ...],
) -> _Trace:
    return _Trace(
        sequence_id="prompt-0",
        lane=lane,
        prompt_token_ids=(0, 1),
        consumed_token_ids=selected[:-1],
        selected_token_ids=selected,
        steps=tuple(
            _Step(index, token, top, _row(top))
            for index, (token, top) in enumerate(
                zip(selected, tops, strict=True)
            )
        ),
    )


def _functional_fixture():
    parent = {
        "forced": _trace("forced", (1, 2, 3), (1, 2, 3)),
        "greedy": _trace("greedy", (1, 2, 3), (1, 2, 3)),
    }
    perfect = dict(parent)
    weak = {
        "forced": _trace("forced", (1, 2, 3), (4, 4, 4)),
        "greedy": _trace("greedy", (4, 4, 4), (4, 4, 4)),
    }
    return parent, perfect, weak


def test_heldout_inversion_cannot_change_calibration_winner_or_name_one() -> None:
    parent, perfect, weak = _functional_fixture()
    comparison = search._build_functional_comparison(
        parent,
        parent,
        {"calibration-best": perfect, "heldout-best": weak},
        {"heldout-best": perfect, "calibration-best": weak},
        top_k=3,
    )

    assert (
        comparison["calibration_selection"]["winner_candidate_id"]
        == "calibration-best"
    )
    assert comparison["calibration_selection"]["heldout_used_for_selection"] is False
    assert comparison["heldout_evaluation"]["winner_named"] is False
    assert [
        value["candidate_id"]
        for value in comparison["heldout_evaluation"]["ordering"]["candidates"]
    ] == ["heldout-best", "calibration-best"]


def test_invalid_parent_forced_driver_is_rejected_before_selection() -> None:
    parent, _perfect, _weak = _functional_fixture()
    forced = parent["forced"]
    parent["forced"] = replace(
        forced,
        steps=(
            replace(forced.steps[0], greedy_token_id=5),
            *forced.steps[1:],
        ),
    )
    with pytest.raises(
        search.QwenP35VORoleScaleSearchError,
        match="not all parent top-1",
    ):
        search._assert_parent_driver_top1(parent, split="calibration")


def test_real_search_uses_only_functional_collector_and_restores_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("internal cached-attention trace path was called")

    monkeypatch.setattr(base, "_collect_parent_traces", forbidden)
    monkeypatch.setattr(base, "_collect_candidate_traces", forbidden)
    monkeypatch.setattr(base, "_variant_report", forbidden)
    report = _run(
        runtime,
        (
            search.VOScalePair(1.0, 1.0),
            search.VOScalePair(0.5, -0.5),
        ),
    )

    assert qwen3_attention_runtime_snapshot(runtime) == before
    assert report["incumbent_runtime_restored_exact"] is True
    assert report["tentative_state_reset_exact"] is True
    assert report["lineage_audit"]["prototype_count_removed"] == 1
    assert report["candidate_count"] == 3
    assert report["candidates"][report["candidate_evaluation_order"][0]][
        "tentative_incumbent"
    ]
    winner = report["functional_comparison"]["calibration_selection"][
        "winner_candidate_id"
    ]
    assert winner in report["candidates"]
    assert report["functional_comparison"]["heldout_evaluation"][
        "winner_named"
    ] is False
    assert report["selection_contract"]["internal_tensor_similarity_used"] is False
    for candidate in report["candidates"].values():
        assert candidate["tentative_reset_exact"] is True
        assert candidate["scale_change_audit"][
            "only_target_v_o_scales_changed"
        ]
        assert candidate["scale_change_audit"]["target_q_k_scales_unchanged"]
        assert candidate["installation_audit"]["runtime_parameters_added"] == 0
        assert set(candidate["functional"]) == {
            "calibration_selection",
            "heldout_evaluation",
        }


def test_candidate_tamper_fails_closed_but_restores_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)
    original = search.collect_functional_decode_trace
    calls = 0

    def tamper_after_first_candidate_trace(*args, **kwargs):
        nonlocal calls
        trace = original(*args, **kwargs)
        calls += 1
        if calls == 5:
            registry = base._registry(args[0])
            assert registry.role_scales is not None
            with torch.no_grad():
                registry.role_scales[1, 0, 0].add_(0.25)
        return trace

    monkeypatch.setattr(
        search,
        "collect_functional_decode_trace",
        tamper_after_first_candidate_trace,
    )
    with pytest.raises(ResidualAttentionRoleScaleError, match="no longer matches"):
        _run(runtime)

    assert calls >= 5
    assert qwen3_attention_runtime_snapshot(runtime) == before


def test_prompt_leakage_and_candidate_bound_fail_closed() -> None:
    runtime = _runtime()
    prompt = torch.tensor([[1, 4, 6]])
    with pytest.raises(
        search.QwenP35VORoleScaleSearchError,
        match="must be disjoint",
    ):
        search.run_qwen_p35_vo_role_scale_search(
            runtime,
            (prompt,),
            (prompt.clone(),),
            redirect=base.AttentionRedirect(2, 0, 0, 0),
            require_exact_p35=False,
            horizon=2,
            trace_limits=_limits(),
        )

    with pytest.raises(
        search.QwenP35VORoleScaleSearchError,
        match="exceed 2",
    ):
        search._prepare_scale_candidates(
            (
                search.VOScalePair(1.0, 1.0),
                search.VOScalePair(0.5, 0.5),
            ),
            search.VOScalePair(-0.1, -0.2),
            dtype=torch.float32,
            max_candidates=2,
            max_abs_scale=4.0,
        )


def test_cli_is_hard_wired_and_accepts_repeated_explicit_pairs() -> None:
    assert search.EXACT_REDIRECT == base.AttentionRedirect(19, 4, 21, 6)
    assert search.EXACT_SCALE_SITE == (19, 4)
    args = search.build_parser().parse_args(
        [
            "--codebook-checkpoint",
            "p35",
            "--tokenizer",
            "qwen",
            "--vo-scale",
            "-0.001249:-0.000539",
            "--vo-scale",
            "1:1",
            "--calibration-prompt",
            "alpha",
            "--heldout-prompt",
            "beta",
            "--output",
            "report.json",
        ]
    )
    assert args.vo_scale == [
        search.VOScalePair(-0.001249, -0.000539),
        search.VOScalePair(1.0, 1.0),
    ]
    assert not hasattr(args, "candidate")
