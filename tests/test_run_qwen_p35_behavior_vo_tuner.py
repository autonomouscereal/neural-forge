from __future__ import annotations

import shutil
import sys
from pathlib import Path

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

import run_qwen_p35_behavior_vo_tuner as runner  # noqa: E402
import run_qwen_p35_vo_role_scale_search as role_search  # noqa: E402
import run_qwen_residual_attention_probe as base  # noqa: E402
from neural_forge.hf_functional_decode_trace import (  # noqa: E402
    FunctionalDecodeLimits,
)
from neural_forge.hf_qwen_attention_surgery import (  # noqa: E402
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    prepare_qwen3_codebook_model,
)
from neural_forge.hf_residual_attention_behavior_tuner import (  # noqa: E402
    BEHAVIOR_VO_TUNER_SCHEMA,
    BehaviorVOTuningControls,
    BehaviorVOTuningResult,
)
from neural_forge.hf_residual_attention_role_scale import (  # noqa: E402
    AttentionRoleScales,
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
    torch.manual_seed(1155)
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


def _controls() -> BehaviorVOTuningControls:
    return BehaviorVOTuningControls(
        steps=1,
        max_prompts_per_split=2,
        max_horizon=2,
        max_total_fit_actions=8,
    )


def _fake_result() -> BehaviorVOTuningResult:
    scales = AttentionRoleScales(1.0, 1.0, 0.25, -0.5)
    report = {
        "schema": BEHAVIOR_VO_TUNER_SCHEMA,
        "exact_fold_lineage": {
            "parent_prototype_count": 280,
            "child_prototype_count": 279,
        },
        "accounting": {
            "prototype_count_removed": 1,
            "this_report_covers_one_redirect_only": True,
        },
        "transaction": {
            "full_tentative_role_scales_restored_exactly": True,
            "selected_candidate_installed_after_return": False,
        },
        "checkpoint_saved": False,
        "fold_committed": False,
    }
    return BehaviorVOTuningResult(scales, "fake-calibration-winner", report)


def test_exact_fixture_provenance_rederives_disjoint_fit_rows() -> None:
    fixtures = runner.load_verified_prompt_splits(
        runner.DEFAULT_FIT_PROMPTS,
        runner.DEFAULT_CALIBRATION_PROMPTS,
        runner.DEFAULT_HELDOUT_PROMPTS,
    )

    assert (len(fixtures.fit), len(fixtures.calibration), len(fixtures.heldout)) == (
        8,
        6,
        8,
    )
    assert fixtures.provenance["verified_exact_fixture_provenance"] is True
    assert fixtures.provenance["fit_rows_match_transition_inventory_exactly"]
    assert fixtures.provenance["fit_rows_are_deterministic_quota_winners"]
    assert fixtures.provenance["fit_case_ids"] == [
        "generation_python_add_0835",
        "generation_python_add_0254",
        "generation_python_add_0541",
        "generation_tokenizer_mismatch_0297",
        "generation_tokenizer_mismatch_0059",
        "generation_tokenizer_mismatch_0318",
        "generation_model_merge_risk_0400",
        "generation_model_merge_risk_0302",
    ]
    assert set(fixtures.fit).isdisjoint(fixtures.calibration)
    assert set(fixtures.fit).isdisjoint(fixtures.heldout)
    assert set(fixtures.calibration).isdisjoint(fixtures.heldout)
    assert not fixtures.provenance["broad_promotion_suite_modified"]
    assert not fixtures.provenance["broad_promotion_suite_used_as_live_fit_input"]


def test_any_prompt_fixture_byte_change_fails_closed(tmp_path: Path) -> None:
    tampered = tmp_path / "fit.json"
    shutil.copyfile(runner.DEFAULT_FIT_PROMPTS, tampered)
    tampered.write_text(
        tampered.read_text(encoding="utf-8").replace("0835", "0836", 1),
        encoding="utf-8",
    )

    with pytest.raises(runner.QwenP35BehaviorVOTunerError, match="SHA-256 changed"):
        runner.load_verified_prompt_splits(
            tampered,
            runner.DEFAULT_CALIBRATION_PROMPTS,
            runner.DEFAULT_HELDOUT_PROMPTS,
        )


@pytest.mark.parametrize(
    "fit,calibration,heldout",
    (
        (((1, 4),), ((1, 4),), ((1, 6),)),
        (((1, 4),), ((1, 5),), ((1, 4),)),
        (((1, 4),), ((1, 5),), ((1, 5),)),
    ),
)
def test_every_tokenized_split_pair_must_be_disjoint(
    fit: tuple[tuple[int, ...], ...],
    calibration: tuple[tuple[int, ...], ...],
    heldout: tuple[tuple[int, ...], ...],
) -> None:
    def rows(values: tuple[tuple[int, ...], ...]) -> tuple[torch.Tensor, ...]:
        return tuple(torch.tensor([value]) for value in values)

    with pytest.raises(
        runner.QwenP35BehaviorVOTunerError, match="pairwise disjoint"
    ):
        runner._validate_token_splits(
            rows(fit),
            rows(calibration),
            rows(heldout),
            max_prompts_per_split=2,
            require_exact_p35=False,
        )


def test_runner_collects_parent_outputs_once_and_restores_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)
    calls: list[tuple[str, int]] = []
    original_collect = role_search._collect_parent_split

    def collect_once(model, prompts, *, split, horizon, limits):
        calls.append((split, len(prompts)))
        return original_collect(
            model,
            prompts,
            split=split,
            horizon=horizon,
            limits=limits,
        )

    def behavior_only_tuner(
        model,
        registry,
        fit,
        calibration,
        heldout,
        *,
        controls,
        trace_limits,
    ):
        del controls, trace_limits
        assert model is runtime
        assert registry.prototype_count == 2
        assert {value.lane for value in fit.values()} == {"forced", "greedy"}
        assert {value.lane for value in calibration.values()} == {
            "forced",
            "greedy",
        }
        assert {value.lane for value in heldout.values()} == {"forced", "greedy"}
        return _fake_result()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("internal cached-attention collector was called")

    monkeypatch.setattr(role_search, "_collect_parent_split", collect_once)
    monkeypatch.setattr(runner, "tune_exact_p35_residual_attention_vo", behavior_only_tuner)
    monkeypatch.setattr(base, "_collect_parent_traces", forbidden)
    monkeypatch.setattr(base, "_collect_candidate_traces", forbidden)
    monkeypatch.setattr(base, "_variant_report", forbidden)

    report = runner.run_qwen_p35_behavior_vo_tuner(
        runtime,
        (torch.tensor([[1, 3, 4]]),),
        (torch.tensor([[1, 5, 6]]),),
        (torch.tensor([[1, 7, 8]]),),
        split_provenance={"test": True},
        controls=_controls(),
        redirect=base.AttentionRedirect(2, 0, 0, 0),
        require_exact_p35=False,
        horizon=2,
        trace_limits=_limits(),
    )

    assert calls == [
        ("fit", 1),
        ("calibration-selection", 1),
        ("heldout-evaluation", 1),
    ]
    assert qwen3_attention_runtime_snapshot(runtime) == before
    assert report["transaction"]["incumbent_p35_runtime_restored_exactly"]
    assert report["transaction"][
        "tentative_child_restored_exactly_after_tuning"
    ]
    assert report["objective_contract"]["causal_output_action_space_only"]
    assert not report["objective_contract"]["weight_similarity_used"]
    assert not report["objective_contract"]["activation_similarity_used"]
    assert not report["objective_contract"]["cache_value_similarity_used"]
    assert report["selected_candidate_id"] == "fake-calibration-winner"
    assert report["checkpoint_saved"] is False
    assert report["fold_committed"] is False


def test_tuner_state_tamper_fails_but_parent_still_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)

    def tampering_tuner(
        _model,
        registry,
        _fit,
        _calibration,
        _heldout,
        **_kwargs,
    ):
        assert registry.role_scales is not None
        with torch.no_grad():
            registry.role_scales[0, 0, 0].add_(0.25)
        return _fake_result()

    monkeypatch.setattr(runner, "tune_exact_p35_residual_attention_vo", tampering_tuner)
    with pytest.raises(
        runner.QwenP35BehaviorVOTunerError,
        match="did not restore the tentative",
    ):
        runner.run_qwen_p35_behavior_vo_tuner(
            runtime,
            (torch.tensor([[1, 3, 4]]),),
            (torch.tensor([[1, 5, 6]]),),
            (torch.tensor([[1, 7, 8]]),),
            split_provenance={"test": True},
            controls=_controls(),
            redirect=base.AttentionRedirect(2, 0, 0, 0),
            require_exact_p35=False,
            horizon=2,
            trace_limits=_limits(),
        )

    assert qwen3_attention_runtime_snapshot(runtime) == before


def test_cli_requires_all_three_json_roles_and_has_no_commit_option() -> None:
    args = runner.build_parser().parse_args(
        [
            "--codebook-checkpoint",
            "p35",
            "--tokenizer",
            "qwen",
            "--fit-prompts-json",
            str(runner.DEFAULT_FIT_PROMPTS),
            "--calibration-selection-prompts-json",
            str(runner.DEFAULT_CALIBRATION_PROMPTS),
            "--heldout-evaluation-prompts-json",
            str(runner.DEFAULT_HELDOUT_PROMPTS),
            "--output",
            "report.json",
        ]
    )

    assert runner.EXACT_REDIRECT == base.AttentionRedirect(19, 4, 21, 6)
    assert args.fit_prompts_json == runner.DEFAULT_FIT_PROMPTS
    assert args.calibration_selection_prompts_json == runner.DEFAULT_CALIBRATION_PROMPTS
    assert args.heldout_evaluation_prompts_json == runner.DEFAULT_HELDOUT_PROMPTS
    assert not hasattr(args, "save")
    assert not hasattr(args, "commit")
