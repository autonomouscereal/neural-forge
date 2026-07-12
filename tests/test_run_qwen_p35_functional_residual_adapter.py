from __future__ import annotations

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

import run_qwen_p35_functional_residual_adapter as runner  # noqa: E402
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
    torch.manual_seed(5150)
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
        max_horizon=1,
        max_prompt_tokens=16,
        max_vocabulary=32,
        max_logit_bytes=100_000,
        max_model_devices=1,
    )


def _run(runtime, **kwargs):
    return runner.run_qwen_p35_functional_residual_adapter(
        runtime,
        (torch.tensor([[1, 3, 4]]),),
        (torch.tensor([[1, 5, 6]]),),
        (torch.tensor([[1, 7, 8]]),),
        split_provenance={"test": True},
        redirect=base.AttentionRedirect(2, 0, 0, 0),
        require_exact_p35=False,
        rank=1,
        horizon=1,
        steps=2,
        micro_batch_size=1,
        learning_rate=1e-3,
        calibration_steps=(0, 1, 2),
        trace_limits=_limits(),
        **kwargs,
    )


def test_exact_fixture_loader_reuses_verified_disjoint_8_6_8_sets() -> None:
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
    assert fixtures.provenance["all_three_text_sets_pairwise_disjoint"] is True


def test_action_batch_builder_uses_parent_top1_on_declared_prefix_lane() -> None:
    runtime = _runtime()
    traces, _drivers = runner._collect_greedy_split(
        runtime,
        (torch.tensor([[1, 3, 4]]),),
        split="fit",
        horizon=1,
        limits=_limits(),
    )

    batches = runner.build_functional_action_batches(
        traces,
        split="fit",
        prefix_source="parent-greedy",
        micro_batch_size=1,
    )

    assert len(batches) == 1
    assert batches[0].split == "fit"
    assert batches[0].prefix_source == "parent-greedy"
    assert torch.equal(batches[0].input_ids, torch.tensor([[1, 3, 4]]))
    assert torch.equal(
        batches[0].parent_action_ids,
        batches[0].parent_action_logits.argmax(dim=-1),
    )


def test_runner_locks_before_heldout_and_restores_parent_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)
    identity_before = role_search._runtime_state_identity_sha256(runtime)
    locked = False
    original_train = runner.train_functional_residual_adapter_disjoint
    original_collect = runner._collect_greedy_split

    def train_then_lock(*args, **kwargs):
        nonlocal locked
        result = original_train(*args, **kwargs)
        locked = True
        return result

    def reject_early_heldout(*args, split, **kwargs):
        if split == "heldout-evaluation":
            assert locked
        return original_collect(*args, split=split, **kwargs)

    monkeypatch.setattr(
        runner, "train_functional_residual_adapter_disjoint", train_then_lock
    )
    monkeypatch.setattr(runner, "_collect_greedy_split", reject_early_heldout)

    report = _run(runtime)

    assert qwen3_attention_runtime_snapshot(runtime) == before
    assert role_search._runtime_state_identity_sha256(runtime) == identity_before
    assert report["transaction"]["incumbent_p35_runtime_restored_exactly"]
    assert report["heldout_access_audit"][
        "heldout_model_forward_only_after_lock"
    ]
    assert (
        report["heldout_access_audit"]["first_heldout_model_forward_event_index"]
        > report["heldout_access_audit"]["winner_lock_event_index"]
    )
    assert report["training"]["gradient_split"] == "fit"
    assert report["training"]["selection_split"] == "calibration-selection"
    assert report["training"]["calibration_selection_steps"] == (0, 1, 2)
    assert report["objective_contract"][
        "weight_values_required_to_match_numerically"
    ] is False
    assert report["checkpoint_saved"] is False
    assert report["fold_committed"] is False


def test_training_failure_still_restores_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)
    identity_before = role_search._runtime_state_identity_sha256(runtime)

    def fail_training(*_args, **_kwargs):
        raise RuntimeError("synthetic fit failure")

    monkeypatch.setattr(
        runner, "train_functional_residual_adapter_disjoint", fail_training
    )
    with pytest.raises(RuntimeError, match="synthetic fit failure"):
        _run(runtime)

    assert qwen3_attention_runtime_snapshot(runtime) == before
    assert role_search._runtime_state_identity_sha256(runtime) == identity_before


def test_cli_fixes_first_run_defaults_and_has_no_save_or_commit_option() -> None:
    args = runner.build_parser().parse_args(
        [
            "--codebook-checkpoint",
            "p35",
            "--tokenizer",
            "qwen",
            "--output",
            "report.json",
        ]
    )

    assert runner.EXACT_REDIRECT == base.AttentionRedirect(19, 4, 21, 6)
    assert args.rank == 8
    assert args.horizon == 4
    assert args.steps == 32
    assert args.micro_batch_size == 2
    assert args.learning_rate == pytest.approx(1.5e-3)
    assert args.fit_prompts_json == runner.DEFAULT_FIT_PROMPTS
    assert args.calibration_selection_prompts_json == runner.DEFAULT_CALIBRATION_PROMPTS
    assert args.heldout_evaluation_prompts_json == runner.DEFAULT_HELDOUT_PROMPTS
    assert not hasattr(args, "save")
    assert not hasattr(args, "commit")
