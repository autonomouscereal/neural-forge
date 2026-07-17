from __future__ import annotations

import copy

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.dynamic_model_inspector import (
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_qwen_adapter import (
    Qwen3CodebookTargetSynthesisAdapter,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesisError,
    DynamicTargetSynthesizer,
)
from neural_forge.hf_qwen_codebook_model import (
    prepare_qwen3_codebook_model,
    qwen3_codebook_parameter_report,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tuple(tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[token_id]


def _model(seed: int):
    torch.manual_seed(seed)
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=12,
            hidden_size=8,
            intermediate_size=6,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    ).eval()


def _tokenizers():
    parent = _Tokenizer([f"token-{index}" for index in range(12)])
    target = _Tokenizer(
        [
            "token-0",
            "token-1",
            "token-2",
            "token-5",
            "token-7",
            "token-9",
            "token-11",
        ]
    )
    return parent, target


def _expected_budget(parent, parent_tokenizer, target_tokenizer):
    blank = type(parent)(copy.deepcopy(parent.config)).eval()
    prepare_qwen3_codebook_model(
        blank,
        parent_tokenizer,
        target_tokenizer,
        attention_prototypes_per_group=1,
        neuron_prototypes_per_neuron=2,
    )
    return qwen3_codebook_parameter_report(blank).unique_parameter_count


def test_qwen_codebook_target_is_synthesized_without_caller_layer_surgery() -> None:
    parent = _model(1)
    parent_tokenizer, target_tokenizer = _tokenizers()
    parent_fingerprint = canonical_sha256({"tokenizer": "parent"})
    target_fingerprint = canonical_sha256({"tokenizer": "target"})
    exact_budget = _expected_budget(
        parent,
        parent_tokenizer,
        target_tokenizer,
    )
    adapter = Qwen3CodebookTargetSynthesisAdapter(
        parent_tokenizer=parent_tokenizer,
        target_tokenizer=target_tokenizer,
        attention_prototypes_per_group=1,
        neuron_prototypes_per_neuron=2,
    )

    result = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((adapter,)),
    ).synthesize(
        request=DynamicTargetRequest(
            child_name="qwen-codebook-child",
            target_family="qwen3_codebook",
            target_modalities=("text",),
            exact_unique_parameters=exact_budget,
            tokenizer_fingerprint=target_fingerprint,
            blank_seed=2,
        ),
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={
            "parent": parent_fingerprint,
        },
    )

    report = result.adapter_payload
    assert result.adapter_id == adapter.adapter_id
    assert result.graph.family == "qwen3_codebook"
    assert result.graph.unique_parameter_count == exact_budget
    assert result.graph.layer_indices == (0, 1)
    assert report.parent_runtime_reference_count == 0
    assert result.as_dict()["pre_fold_state_independently_fingerprinted"] is True
    assert result.blank_state_fingerprint == fingerprint_model_state(result.model)


def test_qwen_codebook_target_fails_closed_on_wrong_exact_budget() -> None:
    parent = _model(3)
    parent_tokenizer, target_tokenizer = _tokenizers()
    target_fingerprint = canonical_sha256({"tokenizer": "target"})
    exact_budget = _expected_budget(
        parent,
        parent_tokenizer,
        target_tokenizer,
    )
    adapter = Qwen3CodebookTargetSynthesisAdapter(
        parent_tokenizer=parent_tokenizer,
        target_tokenizer=target_tokenizer,
        attention_prototypes_per_group=1,
        neuron_prototypes_per_neuron=2,
    )

    with pytest.raises(
        DynamicTargetSynthesisError,
        match="exact unique-parameter budget",
    ):
        DynamicTargetSynthesizer(
            inspectors=default_inspector_registry(),
            adapters=DynamicTargetAdapterRegistry((adapter,)),
        ).synthesize(
            request=DynamicTargetRequest(
                child_name="wrong-budget",
                target_family="qwen3_codebook",
                target_modalities=("text",),
                exact_unique_parameters=exact_budget - 1,
                tokenizer_fingerprint=target_fingerprint,
                blank_seed=4,
            ),
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={
                "parent": canonical_sha256({"tokenizer": "parent"}),
            },
        )


def test_dense_qwen_inspector_remains_selected_for_unconverted_parent() -> None:
    parent = _model(5)
    graph = inspect_model_circuits(
        parent,
        model_name="dense-parent",
        tokenizer_fingerprint=canonical_sha256({"tokenizer": "parent"}),
    )

    assert graph.family == "qwen3"
