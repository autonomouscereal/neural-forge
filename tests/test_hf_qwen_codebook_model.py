from __future__ import annotations

import copy
import gc
import hashlib
import weakref
from dataclasses import dataclass

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)

from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    LEGACY_CHECKPOINT_SCHEMAS,
    Qwen3CodebookError,
    Qwen3CodebookStateError,
    load_qwen3_codebook_checkpoint,
    plan_qwen3_codebook_artifact,
    prepare_qwen3_codebook_model,
    qwen3_codebook_parameter_report,
    save_qwen3_codebook_checkpoint,
)
import neural_forge.hf_qwen_codebook_model as codebook_model  # noqa: E402


class _Tokenizer:
    def __init__(
        self,
        tokens: list[str],
        *,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> None:
        self.tokens = tuple(tokens)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def __len__(self) -> int:
        return len(self.tokens)

    def get_vocab(self) -> dict[str, int]:
        return {token: token_id for token_id, token in enumerate(self.tokens)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[token_id]


@dataclass
class _PreparedRuntime:
    runtime: torch.nn.Module
    dense_reference: torch.nn.Module
    parent_rows: torch.Tensor
    target_input_ids: torch.Tensor
    residual_blocks: tuple[torch.nn.Module, ...]
    layer_norms: tuple[torch.nn.Module, ...]
    replaced_parameter_ids: frozenset[int]
    source_parameters: tuple[torch.nn.Parameter, ...]
    neuron_permutations: torch.Tensor
    group_permutations: torch.Tensor
    parent_tokenizer: _Tokenizer
    target_tokenizer: _Tokenizer


def _tiny_qwen3(seed: int = 641) -> torch.nn.Module:
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(
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
    return transformers.Qwen3ForCausalLM(config).eval()


def _norms(model: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    return tuple(
        norm
        for layer in model.model.layers
        for norm in (
            layer.input_layernorm,
            layer.post_attention_layernorm,
            layer.self_attn.q_norm,
            layer.self_attn.k_norm,
        )
    )


@torch.no_grad()
def _make_depth_equivalent_under_permutations(
    model: torch.nn.Module,
    neuron_permutations: torch.Tensor,
    group_permutations: torch.Tensor,
) -> None:
    source_attention = model.model.layers[0].self_attn
    target_attention = model.model.layers[1].self_attn
    head_dim = int(model.config.head_dim)
    query_rows_per_group = (
        int(model.config.num_attention_heads)
        // int(model.config.num_key_value_heads)
        * head_dim
    )
    for canonical_group, physical_group in enumerate(group_permutations[1].tolist()):
        canonical_q = slice(
            canonical_group * query_rows_per_group,
            (canonical_group + 1) * query_rows_per_group,
        )
        physical_q = slice(
            physical_group * query_rows_per_group,
            (physical_group + 1) * query_rows_per_group,
        )
        canonical_kv = slice(
            canonical_group * head_dim,
            (canonical_group + 1) * head_dim,
        )
        physical_kv = slice(
            physical_group * head_dim,
            (physical_group + 1) * head_dim,
        )
        target_attention.q_proj.weight[physical_q].copy_(
            source_attention.q_proj.weight[canonical_q]
        )
        target_attention.k_proj.weight[physical_kv].copy_(
            source_attention.k_proj.weight[canonical_kv]
        )
        target_attention.v_proj.weight[physical_kv].copy_(
            source_attention.v_proj.weight[canonical_kv]
        )
        target_attention.o_proj.weight[:, physical_q].copy_(
            source_attention.o_proj.weight[:, canonical_q]
        )

    source_mlp = model.model.layers[0].mlp
    target_mlp = model.model.layers[1].mlp
    for canonical_neuron, physical_neuron in enumerate(neuron_permutations[1].tolist()):
        target_mlp.gate_proj.weight[physical_neuron].copy_(
            source_mlp.gate_proj.weight[canonical_neuron]
        )
        target_mlp.up_proj.weight[physical_neuron].copy_(
            source_mlp.up_proj.weight[canonical_neuron]
        )
        target_mlp.down_proj.weight[:, physical_neuron].copy_(
            source_mlp.down_proj.weight[:, canonical_neuron]
        )


def _prepare(seed: int = 641) -> _PreparedRuntime:
    model = _tiny_qwen3(seed)
    neuron_permutations = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [2, 0, 5, 1, 4, 3]],
        dtype=torch.int64,
    )
    group_permutations = torch.tensor(
        [[0, 1], [1, 0]],
        dtype=torch.int64,
    )
    _make_depth_equivalent_under_permutations(
        model,
        neuron_permutations,
        group_permutations,
    )
    parent_tokenizer = _Tokenizer([f"token-{index}" for index in range(12)])
    target_tokenizer = _Tokenizer(
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
    parent_rows = torch.tensor([0, 1, 2, 5, 7, 9, 11], dtype=torch.long)
    target_input_ids = torch.tensor([[1, 3, 4, 6]], dtype=torch.long)
    dense_reference = copy.deepcopy(model)
    residual_blocks = tuple(model.model.layers)
    layer_norms = _norms(model)
    source_parameters = tuple(model.parameters())
    replaced_parameter_ids = frozenset(
        id(parameter)
        for layer in model.model.layers
        for owner in (layer.self_attn, layer.mlp)
        for name in (
            ("q_proj", "k_proj", "v_proj", "o_proj")
            if owner is layer.self_attn
            else ("gate_proj", "up_proj", "down_proj")
        )
        for parameter in getattr(owner, name).parameters()
    ) | {id(model.get_input_embeddings().weight)}

    runtime = prepare_qwen3_codebook_model(
        model,
        parent_tokenizer,
        target_tokenizer,
        neuron_permutations=neuron_permutations,
        group_permutations=group_permutations,
        attention_prototypes_per_group=1,
    )
    return _PreparedRuntime(
        runtime=runtime,
        dense_reference=dense_reference,
        parent_rows=parent_rows,
        target_input_ids=target_input_ids,
        residual_blocks=residual_blocks,
        layer_norms=layer_norms,
        replaced_parameter_ids=replaced_parameter_ids,
        source_parameters=source_parameters,
        neuron_permutations=neuron_permutations,
        group_permutations=group_permutations,
        parent_tokenizer=parent_tokenizer,
        target_tokenizer=target_tokenizer,
    )


def _checkpoint_contract(
    runtime: torch.nn.Module,
    tokenizer: _Tokenizer,
    *,
    name: str = "unit-test-tokenizer",
) -> dict[str, object]:
    parent_rows = runtime.parent_token_rows.detach().to(device="cpu").tolist()
    return {
        "tokenizer_name_or_path": name,
        "vocab_size": len(tokenizer),
        "vocab_sha256": hashlib.sha256(
            codebook_model._canonical_json_bytes(tokenizer.get_vocab())
        ).hexdigest(),
        "special_token_ids": {
            key: getattr(runtime.config, key, None)
            for key in ("bos_token_id", "eos_token_id", "pad_token_id")
        },
        "parent_token_rows_sha256": hashlib.sha256(
            codebook_model._canonical_json_bytes(
                [int(value) for value in parent_rows]
            )
        ).hexdigest(),
    }


@pytest.fixture(scope="module")
def prepared() -> _PreparedRuntime:
    return _prepare()


def test_forward_matches_dense_qwen_with_exact_target_rows_and_permutations(
    prepared: _PreparedRuntime,
) -> None:
    runtime = prepared.runtime
    parent_input_ids = prepared.parent_rows[prepared.target_input_ids]
    with torch.no_grad():
        dense_logits = prepared.dense_reference(
            parent_input_ids,
            use_cache=False,
        ).logits.index_select(-1, prepared.parent_rows)
        runtime_output = runtime(
            prepared.target_input_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    torch.testing.assert_close(runtime_output.logits, dense_logits)
    assert runtime_output.hidden_states is not None
    assert runtime_output.logits.shape == (1, 4, 7)
    assert tuple(runtime.model.layers) == prepared.residual_blocks
    assert _norms(runtime) == prepared.layer_norms
    assert all(parameter.grad is None for parameter in prepared.source_parameters)
    assert runtime.attention_codebook.diagnostics.fitting_requires_grad is False
    assert runtime.neuron_codebook.diagnostics.fitting_requires_grad is False


def test_generate_uses_standard_qwen_cache_and_target_vocabulary(
    prepared: _PreparedRuntime,
) -> None:
    prompt = prepared.target_input_ids[:, :2]
    with torch.no_grad():
        generated = prepared.runtime.generate(
            prompt,
            do_sample=False,
            min_new_tokens=2,
            max_new_tokens=2,
            return_dict_in_generate=True,
            output_scores=True,
        )

    assert generated.sequences.shape == (1, 4)
    assert torch.equal(generated.sequences[:, :2], prompt)
    assert len(generated.scores) == 2
    assert all(score.shape == (1, 7) for score in generated.scores)
    assert int(generated.sequences.max()) < 7


def test_parameter_report_separates_unique_logical_and_index_storage(
    prepared: _PreparedRuntime,
) -> None:
    report = qwen3_codebook_parameter_report(prepared.runtime)

    assert report.source_unique_parameter_count == 816
    assert report.source_logical_parameter_count == 912
    assert report.unique_parameter_count == 440
    assert report.logical_parameter_count == 832
    assert report.registered_logical_parameter_count == 496
    assert report.attention_unique_parameter_count == 192
    assert report.attention_logical_parameter_count == 384
    assert report.neuron_prototypes_per_neuron == 1
    assert report.neuron_prototype_count == 6
    assert report.neuron_unique_parameter_count == 144
    assert report.neuron_logical_parameter_count == 288
    assert report.retained_fixed_parameter_count == 48
    assert report.retained_layer_specific_norm_parameter_count == 40
    assert report.vocabulary_unique_parameter_count == 56
    assert report.vocabulary_logical_parameter_count == 112
    assert report.attention_index_bytes == 48
    assert report.neuron_index_bytes == 72
    assert report.token_row_index_bytes == 28
    assert report.index_buffer_bytes == 148
    assert report.parameters_saved == 392
    assert report.logical_layers == report.residual_block_count == 2
    assert report.layer_specific_norm_count == 8
    assert sum(parameter.numel() for parameter in prepared.runtime.parameters()) == 440
    assert report.as_dict()["index_bytes"] == 148


def test_codebooks_have_one_owner_parameterless_aliases_and_no_parent_runtime(
    prepared: _PreparedRuntime,
) -> None:
    runtime = prepared.runtime
    attention = runtime.attention_codebook
    neurons = runtime.neuron_codebook
    assert (
        runtime.get_input_embeddings().weight is runtime.get_output_embeddings().weight
    )
    torch.testing.assert_close(
        runtime.get_input_embeddings().weight,
        prepared.dense_reference.get_input_embeddings().weight.index_select(
            0,
            prepared.parent_rows,
        ),
    )
    assert torch.equal(runtime.parent_token_rows, prepared.parent_rows.to(torch.int32))

    for layer_index, layer in enumerate(runtime.model.layers):
        attention_view = attention[layer_index]
        assert layer.self_attn.q_proj is attention_view.q_proj
        assert layer.self_attn.k_proj is attention_view.k_proj
        assert layer.self_attn.v_proj is attention_view.v_proj
        assert layer.self_attn.o_proj is attention_view.o_proj
        assert layer.mlp is neurons[layer_index]
        assert attention_view.registry is attention.registry
        assert neurons[layer_index].registry is neurons.registry
        assert not list(attention_view.parameters())
        assert not list(neurons[layer_index].parameters())
        assert not attention_view.state_dict()
        assert not neurons[layer_index].state_dict()

    attention_owners = [
        name
        for name, module in runtime.named_modules(remove_duplicate=False)
        if module is attention.registry
    ]
    neuron_owners = [
        name
        for name, module in runtime.named_modules(remove_duplicate=False)
        if module is neurons.registry
    ]
    assert attention_owners == ["attention_codebook.registry"]
    assert neuron_owners == ["neuron_codebook.registry"]
    runtime_parameter_ids = {id(parameter) for parameter in runtime.parameters()}
    assert runtime_parameter_ids.isdisjoint(prepared.replaced_parameter_ids)
    assert not hasattr(runtime, "parent_model")
    assert not hasattr(runtime, "source_model")
    assert qwen3_codebook_parameter_report(runtime).parent_runtime_reference_count == 0

    state_names = set(runtime.state_dict())
    assert not any(".self_attn.q_proj.weight" in name for name in state_names)
    assert not any(".mlp.gate_proj.weight" in name for name in state_names)
    assert any(name.startswith("attention_codebook.registry.") for name in state_names)
    assert any(name.startswith("neuron_codebook.registry.") for name in state_names)


def test_tensor_state_roundtrip_restores_runtime_exactly(
    prepared: _PreparedRuntime,
) -> None:
    runtime = prepared.runtime
    state = {
        name: value.detach().clone() for name, value in runtime.state_dict().items()
    }
    with torch.no_grad():
        expected = runtime(prepared.target_input_ids, use_cache=False).logits.clone()
        runtime.attention_codebook.registry.prototype_q_weight.add_(0.25)
        changed = runtime(prepared.target_input_ids, use_cache=False).logits
    assert not torch.equal(changed, expected)

    incompatible = runtime.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    for name, value in runtime.state_dict().items():
        assert torch.equal(value, state[name])
    with torch.no_grad():
        restored = runtime(prepared.target_input_ids, use_cache=False).logits
    torch.testing.assert_close(restored, expected, rtol=0.0, atol=0.0)
    assert (
        runtime.get_input_embeddings().weight is runtime.get_output_embeddings().weight
    )


def test_construction_is_deterministic_from_dense_state_and_permutations(
    prepared: _PreparedRuntime,
) -> None:
    repeated = _prepare()
    first_state = prepared.runtime.state_dict()
    second_state = repeated.runtime.state_dict()

    assert first_state.keys() == second_state.keys()
    for name, value in first_state.items():
        assert torch.equal(value, second_state[name]), name
    assert qwen3_codebook_parameter_report(
        prepared.runtime
    ) == qwen3_codebook_parameter_report(repeated.runtime)
    assert all(parameter.grad is None for parameter in repeated.source_parameters)


def test_non_exact_target_tokenizer_is_rejected_before_model_mutation() -> None:
    model = _tiny_qwen3(977)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    parent = _Tokenizer([f"token-{index}" for index in range(12)])
    target = _Tokenizer(["token-0", "token-1", "missing"])

    with pytest.raises(Qwen3CodebookError, match="no exact donor row"):
        prepare_qwen3_codebook_model(
            model,
            parent,
            target,
            neuron_permutations=torch.arange(6).repeat(2, 1),
            group_permutations=torch.arange(2).repeat(2, 1),
        )

    assert not hasattr(model, "attention_codebook")
    assert not hasattr(model, "neuron_codebook")
    assert model.state_dict().keys() == before.keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_replaced_dense_modules_and_parameters_are_process_released() -> None:
    def convert_with_weakrefs() -> tuple[
        torch.nn.Module,
        list[weakref.ReferenceType[torch.nn.Module]],
        list[weakref.ReferenceType[torch.nn.Parameter]],
    ]:
        model = _tiny_qwen3(991)
        modules = [
            module
            for layer in model.model.layers
            for module in (
                layer.mlp,
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
            )
        ]
        parameters = [
            parameter
            for module in modules
            for parameter in module.parameters()
        ] + [model.get_input_embeddings().weight]
        module_refs = [weakref.ref(module) for module in modules]
        parameter_refs = [weakref.ref(parameter) for parameter in parameters]
        tokenizer = _Tokenizer([f"token-{index}" for index in range(12)])
        runtime = prepare_qwen3_codebook_model(
            model,
            tokenizer,
            tokenizer,
            neuron_permutations=torch.arange(6).repeat(2, 1),
            group_permutations=torch.arange(2).repeat(2, 1),
            attention_prototypes_per_group=2,
            neuron_prototypes_per_neuron=2,
        )
        return runtime, module_refs, parameter_refs

    runtime, module_refs, parameter_refs = convert_with_weakrefs()
    gc.collect()
    assert runtime.config.model_type == "qwen3"
    assert all(reference() is None for reference in module_refs)
    assert all(reference() is None for reference in parameter_refs)


def test_compact_checkpoint_reloads_without_dense_parent(tmp_path) -> None:
    pytest.importorskip("safetensors")
    prepared_runtime = _prepare(997)
    runtime = prepared_runtime.runtime
    checkpoint = tmp_path / "compact-child"
    manifest = save_qwen3_codebook_checkpoint(
        runtime,
        checkpoint,
        tokenizer_contract=_checkpoint_contract(
            runtime, prepared_runtime.target_tokenizer
        ),
    )
    assert manifest["format"] == "safetensors"
    assert manifest["assignment_hashes"]
    assert manifest["prototype_source_hashes"]
    assert (checkpoint / "model.safetensors").is_file()
    assert (checkpoint / "config.json").is_file()
    assert (checkpoint / "tokenizer_contract.json").is_file()

    reloaded = load_qwen3_codebook_checkpoint(checkpoint)
    assert reloaded.neural_forge_tokenizer_contract == (
        codebook_model._load_checkpoint_json(
            checkpoint / "tokenizer_contract.json", "tokenizer contract"
        )
    )
    assert qwen3_codebook_parameter_report(reloaded) == qwen3_codebook_parameter_report(
        runtime
    )
    assert not any(
        ".self_attn.q_proj.weight" in name or ".mlp.gate_proj.weight" in name
        for name in reloaded.state_dict()
    )
    assert reloaded.get_input_embeddings().weight is reloaded.get_output_embeddings().weight
    with torch.no_grad():
        expected = runtime(prepared_runtime.target_input_ids, use_cache=False).logits
        actual = reloaded(prepared_runtime.target_input_ids, use_cache=False).logits
        expected_generation = runtime.generate(
            prepared_runtime.target_input_ids[:, :2],
            do_sample=False,
            max_new_tokens=1,
        )
        actual_generation = reloaded.generate(
            prepared_runtime.target_input_ids[:, :2],
            do_sample=False,
            max_new_tokens=1,
        )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.equal(actual_generation, expected_generation)


def test_checkpoint_finiteness_validation_is_detached_from_autograd(monkeypatch) -> None:
    runtime = _prepare(999).runtime
    original = torch.isfinite
    observed_requires_grad: list[bool] = []

    def audited_isfinite(value):
        observed_requires_grad.append(bool(value.requires_grad))
        return original(value)

    monkeypatch.setattr(codebook_model.torch, "isfinite", audited_isfinite)
    state = codebook_model._checkpoint_state(runtime)

    assert state
    assert observed_requires_grad
    assert not any(observed_requires_grad)
    assert all(not value.requires_grad for value in state.values())


def test_artifact_plan_serializes_tied_embedding_once_and_matches_tree(tmp_path) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    runtime = _prepare(1001).runtime
    tokenizer = _Tokenizer(
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
    contract = _checkpoint_contract(runtime, tokenizer)
    plan = plan_qwen3_codebook_artifact(runtime, tokenizer_contract=contract)
    checkpoint = tmp_path / "canonical"
    manifest = save_qwen3_codebook_checkpoint(
        runtime,
        checkpoint,
        tokenizer_contract=contract,
    )
    state = safetensors.load_file(str(checkpoint / "model.safetensors"))

    assert "model.embed_tokens.weight" in state
    assert "lm_head.weight" not in state
    embedding_bytes = (
        runtime.get_input_embeddings().weight.numel()
        * runtime.get_input_embeddings().weight.element_size()
    )
    assert (
        plan.logical_state_tensor_bytes - plan.safetensors_payload_bytes
        == embedding_bytes
    )
    assert manifest["artifact_plan"] == plan.as_dict()
    assert manifest["state_tensor_count"] == len(state)
    assert manifest["canonical_tied_state"] == {
        "serialized_key": "model.embed_tokens.weight",
        "reconstructed_keys": ["lm_head.weight"],
    }
    assert sum(path.stat().st_size for path in checkpoint.iterdir()) == (
        plan.artifact_total_bytes
    )


def test_artifact_plan_marks_unresolved_tokenizer_provenance() -> None:
    prepared = _prepare(1003)
    runtime = prepared.runtime
    unresolved = plan_qwen3_codebook_artifact(runtime)
    short = plan_qwen3_codebook_artifact(
        runtime,
        tokenizer_contract=_checkpoint_contract(
            runtime, prepared.target_tokenizer, name="x"
        ),
    )
    long = plan_qwen3_codebook_artifact(
        runtime,
        tokenizer_contract=_checkpoint_contract(
            runtime, prepared.target_tokenizer, name="a" * 10_000
        ),
    )

    assert unresolved.tokenizer_contract_resolved is False
    assert short.tokenizer_contract_resolved is True
    assert long.tokenizer_contract_resolved is True
    assert unresolved.tokenizer_contract_sha256 != short.tokenizer_contract_sha256
    assert short.tokenizer_contract_file_bytes < long.tokenizer_contract_file_bytes
    assert short.artifact_total_bytes < long.artifact_total_bytes
    assert unresolved.artifact_scope == "four-file-checkpoint-only"
    assert unresolved.artifact_file_count == 4
    assert unresolved.artifact_filenames == (
        "model.safetensors",
        "config.json",
        "tokenizer_contract.json",
        "manifest.json",
    )
    assert unresolved.includes_tokenizer_assets is False
    assert unresolved.includes_runtime_dependencies is False
    assert unresolved.standalone_deployable_bytes is None


def test_checkpoint_save_requires_truthful_tokenizer_contract(tmp_path) -> None:
    prepared = _prepare(1004)
    runtime = prepared.runtime
    valid = _checkpoint_contract(runtime, prepared.target_tokenizer)
    invalid_contracts = []

    missing_digest = dict(valid)
    missing_digest.pop("vocab_sha256")
    invalid_contracts.append(missing_digest)
    invalid_contracts.append({**valid, "vocab_size": len(prepared.target_tokenizer) + 1})
    invalid_contracts.append({**valid, "vocab_sha256": "A" * 64})
    invalid_contracts.append({**valid, "tokenizer_name_or_path": "   "})
    invalid_contracts.append({**valid, "parent_token_rows_sha256": "f" * 64})
    wrong_special = copy.deepcopy(valid)
    wrong_special["special_token_ids"]["bos_token_id"] = 9
    invalid_contracts.append(wrong_special)
    invalid_contracts.append({**valid, "unvalidated_extra": True})

    for index, contract in enumerate(invalid_contracts):
        destination = tmp_path / f"invalid-{index}"
        with pytest.raises(Qwen3CodebookError, match="tokenizer_contract"):
            save_qwen3_codebook_checkpoint(
                runtime,
                destination,
                tokenizer_contract=contract,
            )
        assert not destination.exists()


def test_checkpoint_save_refuses_nonempty_destination(tmp_path) -> None:
    prepared = _prepare(10041)
    destination = tmp_path / "occupied"
    destination.mkdir()
    marker = destination / "user-owned.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(Qwen3CodebookError, match="empty directory"):
        save_qwen3_codebook_checkpoint(
            prepared.runtime,
            destination,
            tokenizer_contract=_checkpoint_contract(
                prepared.runtime, prepared.target_tokenizer
            ),
        )
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert {path.name for path in destination.iterdir()} == {marker.name}


def test_loader_rejects_extra_checkpoint_files(tmp_path) -> None:
    prepared = _prepare(10042)
    destination = tmp_path / "extra-file"
    save_qwen3_codebook_checkpoint(
        prepared.runtime,
        destination,
        tokenizer_contract=_checkpoint_contract(
            prepared.runtime, prepared.target_tokenizer
        ),
    )
    (destination / "unplanned.bin").write_bytes(b"not part of the checkpoint")

    with pytest.raises(Qwen3CodebookError, match="exactly its four planned files"):
        load_qwen3_codebook_checkpoint(destination)


def test_loader_rejects_tokenizer_contract_tamper_with_rehashed_plan(tmp_path) -> None:
    prepared = _prepare(10043)
    destination = tmp_path / "contract-tamper"
    save_qwen3_codebook_checkpoint(
        prepared.runtime,
        destination,
        tokenizer_contract=_checkpoint_contract(
            prepared.runtime, prepared.target_tokenizer
        ),
    )
    contract_path = destination / "tokenizer_contract.json"
    manifest_path = destination / "manifest.json"
    contract = codebook_model._load_checkpoint_json(contract_path, "contract")
    contract["special_token_ids"]["bos_token_id"] = 9
    codebook_model._atomic_write_json(contract_path, contract)
    manifest = codebook_model._load_checkpoint_json(manifest_path, "manifest")
    manifest["files"]["tokenizer_contract.json"] = codebook_model._sha256_file(
        contract_path
    )
    artifact_plan = manifest["artifact_plan"]
    artifact_plan["tokenizer_contract_sha256"] = hashlib.sha256(
        codebook_model._canonical_json_bytes(contract)
    ).hexdigest()
    digest_payload = dict(artifact_plan)
    digest_payload.pop("artifact_plan_sha256")
    artifact_plan["artifact_plan_sha256"] = hashlib.sha256(
        codebook_model._canonical_json_bytes(digest_payload)
    ).hexdigest()
    codebook_model._atomic_write_json(manifest_path, manifest)

    with pytest.raises(Qwen3CodebookError, match="special_token_ids"):
        load_qwen3_codebook_checkpoint(destination)


def test_loader_recomputes_semantic_plan_after_valid_self_rehash(tmp_path) -> None:
    prepared = _prepare(10044)
    destination = tmp_path / "semantic-plan-tamper"
    save_qwen3_codebook_checkpoint(
        prepared.runtime,
        destination,
        tokenizer_contract=_checkpoint_contract(
            prepared.runtime, prepared.target_tokenizer
        ),
    )
    manifest_path = destination / "manifest.json"
    manifest = codebook_model._load_checkpoint_json(manifest_path, "manifest")
    artifact_plan = manifest["artifact_plan"]
    artifact_plan["logical_state_tensor_bytes"] += 1
    digest_payload = dict(artifact_plan)
    digest_payload.pop("artifact_plan_sha256")
    artifact_plan["artifact_plan_sha256"] = hashlib.sha256(
        codebook_model._canonical_json_bytes(digest_payload)
    ).hexdigest()
    original_size = manifest_path.stat().st_size
    codebook_model._atomic_write_json(manifest_path, manifest)
    assert manifest_path.stat().st_size == original_size

    with pytest.raises(Qwen3CodebookStateError, match="semantic artifact plan"):
        load_qwen3_codebook_checkpoint(destination)


def test_checkpoint_save_fails_closed_on_artifact_plan_drift(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("safetensors")
    prepared = _prepare(1005)
    runtime = prepared.runtime
    load_file, real_save_file = codebook_model._safetensors_io()

    def drifting_save_file(state, path, *, metadata):
        real_save_file(state, path, metadata=metadata)
        with open(path, "ab") as handle:
            handle.write(b"unexpected-byte")

    monkeypatch.setattr(
        codebook_model,
        "_safetensors_io",
        lambda: (load_file, drifting_save_file),
    )
    checkpoint = tmp_path / "drift"
    with pytest.raises(Qwen3CodebookStateError, match="drifted from the artifact plan"):
        save_qwen3_codebook_checkpoint(
            runtime,
            checkpoint,
            tokenizer_contract=_checkpoint_contract(
                runtime, prepared.target_tokenizer
            ),
        )
    assert not (checkpoint / "manifest.json").exists()


def test_artifact_plan_matches_variable_prototype_shapes(tmp_path) -> None:
    pytest.importorskip("safetensors")
    tokenizer = _Tokenizer([f"token-{index}" for index in range(12)])
    runtimes = (
        prepare_qwen3_codebook_model(
            _tiny_qwen3(1007),
            tokenizer,
            tokenizer,
            attention_prototypes_per_group=1,
            neuron_prototypes_per_neuron=1,
        ),
        prepare_qwen3_codebook_model(
            _tiny_qwen3(1009),
            tokenizer,
            tokenizer,
            attention_prototypes_per_group=2,
            neuron_prototypes_per_neuron=2,
        ).to(dtype=torch.bfloat16),
    )
    plans = []
    for index, runtime in enumerate(runtimes):
        contract = _checkpoint_contract(
            runtime, tokenizer, name=f"shape-{index}"
        )
        plan = plan_qwen3_codebook_artifact(
            runtime,
            tokenizer_contract=contract,
        )
        checkpoint = tmp_path / f"shape-{index}"
        manifest = save_qwen3_codebook_checkpoint(
            runtime,
            checkpoint,
            tokenizer_contract=contract,
        )
        assert manifest["artifact_plan"] == plan.as_dict()
        assert sum(path.stat().st_size for path in checkpoint.iterdir()) == (
            plan.artifact_total_bytes
        )
        plans.append(plan)

    assert plans[0].safetensors_payload_bytes != plans[1].safetensors_payload_bytes
    assert plans[0].safetensors_header_sha256 != plans[1].safetensors_header_sha256


@pytest.mark.parametrize("legacy_schema", sorted(LEGACY_CHECKPOINT_SCHEMAS))
def test_loader_accepts_legacy_dual_tied_embedding_checkpoint(
    tmp_path, legacy_schema: str
) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    prepared = _prepare(1011)
    runtime = prepared.runtime
    checkpoint = tmp_path / "legacy"
    manifest = save_qwen3_codebook_checkpoint(
        runtime,
        checkpoint,
        tokenizer_contract=_checkpoint_contract(
            runtime, prepared.target_tokenizer, name="legacy"
        ),
    )
    state = safetensors.load_file(str(checkpoint / "model.safetensors"))
    state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
    if legacy_schema == codebook_model.LEGACY_V1_CHECKPOINT_SCHEMA:
        metadata_bytes = sum(
            int(state.pop(name).numel()) * torch.tensor(0, dtype=torch.int32).element_size()
            for name in (
                "attention_codebook.registry._registry_schema_version",
                "attention_codebook.registry._registry_layout",
            )
        )
        manifest["parameter_report"]["attention_index_bytes"] -= metadata_bytes
        manifest["parameter_report"]["index_buffer_bytes"] -= metadata_bytes
        manifest["parameter_report"]["index_bytes"] -= metadata_bytes
    safetensors.save_file(
        state,
        str(checkpoint / "model.safetensors"),
        metadata={"format": legacy_schema},
    )
    manifest["schema"] = legacy_schema
    manifest.pop("artifact_plan")
    manifest.pop("canonical_tied_state")
    if legacy_schema == codebook_model.LEGACY_V1_CHECKPOINT_SCHEMA:
        manifest.pop("attention_registry")
    manifest["files"]["model.safetensors"] = codebook_model._sha256_file(
        checkpoint / "model.safetensors"
    )
    manifest["state_key_sha256"] = hashlib.sha256(
        codebook_model._canonical_json_bytes(sorted(state))
    ).hexdigest()
    manifest["state_tensor_count"] = len(state)
    manifest["state_groups"] = codebook_model._checkpoint_grouped_keys(state)
    codebook_model._atomic_write_json(checkpoint / "manifest.json", manifest)

    restored = load_qwen3_codebook_checkpoint(checkpoint)
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    with torch.no_grad():
        expected = runtime(torch.tensor([[1, 2, 3]]), use_cache=False).logits
        actual = restored(torch.tensor([[1, 2, 3]]), use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert CHECKPOINT_SCHEMA != legacy_schema


def test_full_depth_neuron_prototypes_reconstruct_and_reload_qwen_runtime() -> None:
    seed = 983
    model = _tiny_qwen3(seed)
    dense_reference = copy.deepcopy(model)
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
    parent_rows = torch.tensor([0, 1, 2, 5, 7, 9, 11], dtype=torch.long)
    target_input_ids = torch.tensor([[1, 3, 4, 6]], dtype=torch.long)
    neuron_permutations = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [3, 0, 5, 1, 4, 2]],
        dtype=torch.int64,
    )
    group_permutations = torch.tensor(
        [[0, 1], [1, 0]],
        dtype=torch.int64,
    )
    prepare_kwargs = {
        "neuron_permutations": neuron_permutations,
        "group_permutations": group_permutations,
        "attention_prototypes_per_group": 2,
        "neuron_prototypes_per_neuron": 2,
    }

    runtime = prepare_qwen3_codebook_model(
        model,
        parent,
        target,
        **prepare_kwargs,
    )
    report = qwen3_codebook_parameter_report(runtime)
    assert report.attention_prototypes_per_group == 2
    assert report.neuron_prototypes_per_neuron == 2
    assert report.neuron_prototype_count == 12
    assert report.attention_unique_parameter_count == 384
    assert report.neuron_unique_parameter_count == 288
    assert report.neuron_logical_parameter_count == 288
    assert report.neuron_index_bytes == 96
    assert report.unique_parameter_count == 776
    assert report.logical_parameter_count == 832
    assert report.registered_logical_parameter_count == 832
    assert report.index_buffer_bytes == 180
    assert runtime.neuron_codebook.registry.neuron_scales is None

    for layer_index, dense_layer in enumerate(dense_reference.model.layers):
        materialized = runtime.neuron_codebook.registry.materialize_layer(layer_index)
        torch.testing.assert_close(
            materialized.gate_weight,
            dense_layer.mlp.gate_proj.weight,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            materialized.up_weight,
            dense_layer.mlp.up_proj.weight,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            materialized.down_weight,
            dense_layer.mlp.down_proj.weight,
            rtol=0.0,
            atol=0.0,
        )

    with torch.no_grad():
        expected = dense_reference(
            parent_rows[target_input_ids],
            use_cache=False,
        ).logits.index_select(-1, parent_rows)
        actual = runtime(target_input_ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected)

    state = {
        name: value.detach().clone() for name, value in runtime.state_dict().items()
    }
    restored = prepare_qwen3_codebook_model(
        _tiny_qwen3(seed),
        parent,
        target,
        **prepare_kwargs,
    )
    incompatible = restored.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    assert qwen3_codebook_parameter_report(restored) == report
    with torch.no_grad():
        restored_logits = restored(target_input_ids, use_cache=False).logits
    torch.testing.assert_close(restored_logits, expected)
