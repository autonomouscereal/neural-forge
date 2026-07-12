from __future__ import annotations

import hashlib
import json
import shutil

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)

import neural_forge.hf_qwen_tied_depth_checkpoint as checkpoint_module  # noqa: E402
from neural_forge.hf_qwen_tied_depth_checkpoint import (  # noqa: E402
    MANIFEST_FILENAME,
    Qwen3TiedDepthBudgetError,
    Qwen3TiedDepthCheckpointError,
    Qwen3TiedDepthFormatError,
    count_qwen3_tied_depth_parameters,
    export_qwen3_tied_depth_checkpoint,
    load_qwen3_tied_depth_checkpoint,
    plan_qwen3_tied_depth_checkpoint,
)
from neural_forge.hf_tied_depth import (  # noqa: E402
    ATTENTION_PROJECTIONS,
    MLP_PROJECTIONS,
)


class _Tokenizer:
    def __init__(self, tokens, *, bos=1, eos=2, pad=0):
        self._tokens = tuple(tokens)
        self._vocab = {token: index for index, token in enumerate(self._tokens)}
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.pad_token_id = pad

    def __len__(self):
        return len(self._tokens)

    def get_vocab(self):
        return dict(self._vocab)

    def convert_ids_to_tokens(self, token_id):
        if 0 <= token_id < len(self._tokens):
            return self._tokens[token_id]
        return None


def _tiny_qwen3():
    torch.manual_seed(2421)
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=31,
            hidden_size=12,
            intermediate_size=16,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
            use_cache=False,
        )
    ).eval()


def _tokenizers():
    donor_tokens = [f"token-{index}" for index in range(31)]
    selected = (0, 3, 7, 11, 15, 19, 23, 27)
    target_tokens = [donor_tokens[index] for index in selected]
    return _Tokenizer(donor_tokens), _Tokenizer(target_tokens), selected


def _logical_parameter_count(model) -> int:
    return sum(
        parameter.numel()
        for _, parameter in model.named_parameters(remove_duplicate=False)
    )


def _canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _rewrite_manifest(path, mutate) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    path.write_bytes(_canonical_json_bytes(manifest))
    return manifest


def test_closed_form_counts_and_export_budget_fail_before_mutation(tmp_path) -> None:
    model = _tiny_qwen3()
    donor, target, _ = _tokenizers()
    plan = plan_qwen3_tied_depth_checkpoint(model.config, len(target))

    assert plan.unique_parameter_count == 1_356
    assert plan.logical_parameter_count == 3_756
    assert plan.dense_source_unique_parameter_count == 3_936
    assert plan.dense_source_logical_parameter_count == 4_308
    assert plan.shared_attention_parameter_count == 576
    assert plan.shared_mlp_parameter_count == 576
    assert plan.retained_norm_parameter_count == 108
    assert count_qwen3_tied_depth_parameters(model.config, len(target)) == 1_356

    destination = tmp_path / "over-budget"
    with pytest.raises(Qwen3TiedDepthBudgetError, match="exceeding"):
        export_qwen3_tied_depth_checkpoint(
            model,
            destination,
            donor,
            target,
            attention_source_layer=1,
            mlp_source_layer=2,
            parameter_budget=plan.unique_parameter_count - 1,
            tensor_format="raw",
        )
    assert not destination.exists()
    assert model.config.vocab_size == 31
    assert len(
        {id(layer.self_attn.q_proj.weight) for layer in model.model.layers}
    ) == len(model.model.layers)


def test_safetensors_roundtrip_rows_aliases_counts_and_forward(tmp_path) -> None:
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    model = _tiny_qwen3()
    donor, target, selected = _tokenizers()
    source_embedding = model.get_input_embeddings().weight.detach().clone()
    source_attention = {
        role: getattr(model.model.layers[1].self_attn, role).weight.detach().clone()
        for role in ATTENTION_PROJECTIONS
    }
    source_mlp = {
        role: getattr(model.model.layers[2].mlp, role).weight.detach().clone()
        for role in MLP_PROJECTIONS
    }
    checkpoint = tmp_path / "safe"

    manifest = export_qwen3_tied_depth_checkpoint(
        model,
        checkpoint,
        donor,
        target,
        attention_source_layer=1,
        mlp_source_layer=2,
        parameter_budget=1_356,
        tensor_format="safetensors",
    )
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        expected_logits = model(input_ids=input_ids, use_cache=False).logits

    torch.testing.assert_close(
        model.get_input_embeddings().weight,
        source_embedding[list(selected)],
        rtol=0,
        atol=0,
    )
    for layer in model.model.layers:
        for role, expected in source_attention.items():
            torch.testing.assert_close(
                getattr(layer.self_attn, role).weight,
                expected,
                rtol=0,
                atol=0,
            )
        for role, expected in source_mlp.items():
            torch.testing.assert_close(
                getattr(layer.mlp, role).weight,
                expected,
                rtol=0,
                atol=0,
            )

    assert {path.name for path in checkpoint.iterdir()} == {
        "config.json",
        "generation_config.json",
        MANIFEST_FILENAME,
        "model.safetensors",
    }
    assert manifest["vocabulary"]["donor_row_indices"] == list(selected)
    assert manifest["topology"]["attention"]["source_layer"] == 1
    assert manifest["topology"]["mlp"]["source_layer"] == 2
    assert manifest["serialization"] == {
        "python_pickle": False,
        "serialized_modules": False,
        "serialized_weakrefs": False,
        "tensor_format": "safetensors",
        "byte_order": "little",
    }
    state = manifest["state_dict"]
    assert state["logical_tensor_count"] == 36
    assert state["stored_tensor_count"] == 21
    assert state["projection_alias_count"] == 14
    assert state["vocabulary_alias_count"] == 1
    assert state["alias_count"] == 15
    tensor_path = checkpoint / manifest["artifacts"]["tensors"]["file"]
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(manifest["tensors"])

    loaded = load_qwen3_tied_depth_checkpoint(checkpoint)
    with torch.no_grad():
        actual_logits = loaded(input_ids=input_ids, use_cache=False).logits
    assert torch.equal(actual_logits, expected_logits)
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
    for role in ATTENTION_PROJECTIONS:
        parameters = [
            getattr(layer.self_attn, role).weight for layer in loaded.model.layers
        ]
        assert all(parameter is parameters[0] for parameter in parameters)
    for role in MLP_PROJECTIONS:
        parameters = [getattr(layer.mlp, role).weight for layer in loaded.model.layers]
        assert all(parameter is parameters[0] for parameter in parameters)
    norm_parameters = []
    for layer in loaded.model.layers:
        norm_parameters.extend(
            (
                layer.self_attn.q_norm.weight,
                layer.self_attn.k_norm.weight,
                layer.input_layernorm.weight,
                layer.post_attention_layernorm.weight,
            )
        )
    assert len({id(parameter) for parameter in norm_parameters}) == 12
    assert sum(parameter.numel() for parameter in loaded.parameters()) == 1_356
    assert _logical_parameter_count(loaded) == 3_756
    assert loaded.neural_forge_load_report["constructed_on_meta"] is True
    assert loaded.neural_forge_load_report["streamed_physical_tensors"] is True
    assert not any(
        marker in name.lower()
        for name, _ in loaded.named_modules()
        for marker in ("adapter", "teacher", "parent")
    )


def test_raw_payload_is_pickle_free_streamed_and_exact(tmp_path, monkeypatch) -> None:
    model = _tiny_qwen3()
    donor, target, _ = _tokenizers()
    checkpoint = tmp_path / "raw"
    manifest = export_qwen3_tied_depth_checkpoint(
        model,
        checkpoint,
        donor,
        target,
        attention_source_layer=0,
        mlp_source_layer=1,
        tensor_format="raw",
    )
    input_ids = torch.tensor([[1, 4, 6]], dtype=torch.long)
    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False).logits

    def unsafe_loader(*_args, **_kwargs):
        raise AssertionError("pickle/object loader must not be called")

    monkeypatch.setattr(torch, "load", unsafe_loader)
    try:
        import safetensors.torch
    except ImportError:
        pass
    else:
        monkeypatch.setattr(safetensors.torch, "load_file", unsafe_loader)
    loaded = load_qwen3_tied_depth_checkpoint(checkpoint)
    with torch.no_grad():
        actual = loaded(input_ids=input_ids, use_cache=False).logits

    assert torch.equal(actual, expected)
    assert (checkpoint / "model.nfraw").is_file()
    assert not any(
        path.suffix.lower() in {".bin", ".pkl", ".pickle", ".pt", ".pth"}
        for path in checkpoint.iterdir()
    )
    memory = manifest["reload_memory"]
    assert memory["topology_construction_device"] == "meta"
    assert memory["streamed_physical_tensors"] is True
    assert memory["largest_staging_tensor_bytes"] < memory["model_parameter_bytes"]


def test_manifest_payload_alias_and_raw_range_corruption_fail_closed(tmp_path) -> None:
    model = _tiny_qwen3()
    donor, target, _ = _tokenizers()
    original = tmp_path / "original"
    export_qwen3_tied_depth_checkpoint(
        model,
        original,
        donor,
        target,
        attention_source_layer=1,
        mlp_source_layer=2,
        tensor_format="raw",
    )

    bad_manifest = tmp_path / "bad-manifest"
    shutil.copytree(original, bad_manifest)
    manifest_path = bad_manifest / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["standalone"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(Qwen3TiedDepthFormatError, match="Manifest SHA-256"):
        load_qwen3_tied_depth_checkpoint(bad_manifest)

    bad_payload = tmp_path / "bad-payload"
    shutil.copytree(original, bad_payload)
    payload_path = bad_payload / "model.nfraw"
    payload = bytearray(payload_path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    payload_path.write_bytes(payload)
    with pytest.raises(Qwen3TiedDepthFormatError, match="failed SHA-256"):
        load_qwen3_tied_depth_checkpoint(bad_payload)

    bad_alias = tmp_path / "bad-alias"
    shutil.copytree(original, bad_alias)
    _rewrite_manifest(
        bad_alias / MANIFEST_FILENAME,
        lambda value: value["state_dict"].__setitem__("alias_count", 14),
    )
    with pytest.raises(Qwen3TiedDepthFormatError, match="counts are inconsistent"):
        load_qwen3_tied_depth_checkpoint(bad_alias)

    bad_range = tmp_path / "bad-range"
    shutil.copytree(original, bad_range)

    def shift_second_range(value):
        second = sorted(value["tensors"])[1]
        value["tensors"][second]["byte_offset"] += 1

    _rewrite_manifest(bad_range / MANIFEST_FILENAME, shift_second_range)
    with pytest.raises(Qwen3TiedDepthFormatError, match="ranges must be sorted"):
        load_qwen3_tied_depth_checkpoint(bad_range)


def test_reload_budget_caps_fail_before_meta_model_construction(
    tmp_path,
    monkeypatch,
) -> None:
    model = _tiny_qwen3()
    donor, target, _ = _tokenizers()
    checkpoint = tmp_path / "budgeted"
    manifest = export_qwen3_tied_depth_checkpoint(
        model,
        checkpoint,
        donor,
        target,
        attention_source_layer=0,
        mlp_source_layer=0,
        tensor_format="raw",
    )
    counts = manifest["parameter_counts"]
    memory = manifest["reload_memory"]

    def meta_construction_must_not_run(*_args, **_kwargs):
        raise AssertionError("budget checks must precede meta model construction")

    monkeypatch.setattr(
        checkpoint_module,
        "_build_meta_topology",
        meta_construction_must_not_run,
    )
    limits = (
        {"max_parameter_count": counts["unique_parameter_count"] - 1},
        {"max_model_bytes": memory["model_parameter_bytes"] - 1},
        {"max_tensor_bytes": memory["largest_staging_tensor_bytes"] - 1},
        {"max_memory_bytes": (memory["bounded_peak_parameter_and_staging_bytes"] - 1)},
    )
    for kwargs in limits:
        with pytest.raises(Qwen3TiedDepthBudgetError, match="exceeding"):
            load_qwen3_tied_depth_checkpoint(checkpoint, **kwargs)


def test_atomic_failures_leave_no_partial_checkpoint(tmp_path, monkeypatch) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("keep", encoding="ascii")
    donor, target, _ = _tokenizers()
    with pytest.raises(Qwen3TiedDepthCheckpointError, match="Refusing to replace"):
        export_qwen3_tied_depth_checkpoint(
            _tiny_qwen3(),
            existing,
            donor,
            target,
            attention_source_layer=0,
            mlp_source_layer=0,
            tensor_format="raw",
        )
    assert sentinel.read_text(encoding="ascii") == "keep"

    def fail_payload(*_args, **_kwargs):
        raise RuntimeError("injected payload failure")

    monkeypatch.setattr(checkpoint_module, "_save_tensor_payload", fail_payload)
    destination = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected payload failure"):
        export_qwen3_tied_depth_checkpoint(
            _tiny_qwen3(),
            destination,
            donor,
            target,
            attention_source_layer=0,
            mlp_source_layer=0,
            tensor_format="raw",
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))
