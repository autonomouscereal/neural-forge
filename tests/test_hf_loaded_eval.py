from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from neural_forge.hf_tools import _generation_health, evaluate_loaded_text_model  # noqa: E402


def _tokenizer():
    words = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + [f"w{i}" for i in range(12)]
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {word: index for index, word in enumerate(words)},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )


def test_loaded_model_evaluation_does_not_require_checkpoint_reload(tmp_path) -> None:
    torch.manual_seed(704)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=16,
            n_positions=32,
            n_embd=8,
            n_layer=2,
            n_head=2,
            n_inner=12,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    suite = {
        "source": "tiny-loaded-model-suite",
        "text_evals": ["w1 w2 w3"],
        "choice_probes": [],
        "generation_prompts": [
            {"id": "generation", "prompt": "w1", "relevance_rules": []}
        ],
        "behavior_probes": [
            {
                "id": "behavior",
                "prompt": "w2",
                "kind": "pattern_groups",
                "pattern_groups": [],
            }
        ],
    }

    report = evaluate_loaded_text_model(
        model,
        _tokenizer(),
        tmp_path / "eval.json",
        model_label="in-memory-ir",
        device="cpu",
        max_new_tokens=2,
        eval_suite=suite,
    )

    assert report["model"] == "in-memory-ir"
    assert report["param_count"] == sum(parameter.numel() for parameter in model.parameters())
    assert report["eval_suite"] == {
        "source": "tiny-loaded-model-suite",
        "text_eval_count": 1,
        "choice_probe_count": 0,
        "generation_prompt_count": 1,
        "behavior_probe_count": 1,
    }
    assert report["perplexity"]["items"]
    assert len(report["generations"]) == 1
    assert len(report["behavior_suite"]["items"]) == 1
    assert report["generation_health"]["contract"] == {
        "min_continuation_chars": 24,
        "min_unique_token_ratio": 0.25,
        "min_distinct_bigram_ratio": 0.50,
        "max_repeated_bigram_share": 0.35,
        "repeated_prompt_allowed": False,
        "exact_expected_behavior_can_satisfy_minimum_length": True,
        "empty_continuation_passes": False,
        "low_information_continuation_passes": False,
    }
    assert (tmp_path / "eval.json").is_file()


@pytest.mark.parametrize(
    "output",
    [
        "prompt",
        "prompt a a a a a a a a a a a a a a a a",
        "prompt one two",
        (
            "prompt This is a longer test prompt repeated exactly "
            "This is a longer test prompt repeated exactly "
            "This is a longer test prompt repeated exactly"
        ),
    ],
)
def test_generation_health_rejects_empty_short_and_low_information_output(
    output: str,
) -> None:
    report = _generation_health(
        {"id": "g", "prompt": "prompt", "output": output},
        min_continuation_chars=24,
        min_unique_token_ratio=0.25,
    )
    assert report["passed"] is False


def test_generation_health_rejects_full_prompt_echo_loop() -> None:
    prompt = "Explain why tokenizer mismatch harms a stitched language model."
    report = _generation_health(
        {
            "id": "echo",
            "prompt": prompt,
            "output": f"{prompt} {prompt} {prompt}",
        },
        min_continuation_chars=24,
        min_unique_token_ratio=0.25,
    )
    assert report["prompt_repeated"] is True
    assert report["passed"] is False
