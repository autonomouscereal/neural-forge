from __future__ import annotations

import hashlib

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.dynamic_fold_executor import (  # noqa: E402
    fingerprint_model_state,
)
from neural_forge.hf_lfm2_cpcf_blank_target import (  # noqa: E402
    Lfm2CpcfBlankTargetError,
    plan_and_synthesize_lfm2_cpcf_blank,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parent():
    torch.manual_seed(1801)
    config = transformers.Lfm2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=[
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
        ],
        conv_L_cache=3,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


@pytest.mark.parametrize("ratio", (4,))
def test_registry_creates_real_exact_dense_zero_blank(ratio: int) -> None:
    parent = _parent()
    parent_before = fingerprint_model_state(parent)

    result = plan_and_synthesize_lfm2_cpcf_blank(
        parent,
        parent_name="parent",
        parent_tokenizer_fingerprint=_sha("tokenizer"),
        evidence_fingerprint=_sha("cache"),
        child_name=f"child-{ratio}x",
        compression_ratio=ratio,
        blank_seed=ratio,
        device="cpu",
        dtype=torch.float32,
    )

    assert result.model is not parent
    assert result.unique_parameters < sum(
        parameter.numel()
        for parameter in {id(value): value for value in parent.parameters()}.values()
    )
    assert result.realized_compression_ratio >= ratio
    assert result.report["compiler_created_blank"] is True
    assert result.report["quantization_used"] is False
    assert result.report["pruning_used"] is False
    assert result.report["training_used"] is False
    assert result.model.get_input_embeddings().weight is (
        result.model.get_output_embeddings().weight
    )
    assert all(
        int(torch.count_nonzero(parameter).item()) == 0
        for parameter in result.model.parameters()
    )
    assert fingerprint_model_state(parent) == parent_before


def test_blank_request_requires_one_real_parameter_budget() -> None:
    parent = _parent()
    with pytest.raises(
        Lfm2CpcfBlankTargetError,
        match="exactly one",
    ):
        plan_and_synthesize_lfm2_cpcf_blank(
            parent,
            parent_name="parent",
            parent_tokenizer_fingerprint=_sha("tokenizer"),
            evidence_fingerprint=_sha("cache"),
            child_name="invalid",
        )
