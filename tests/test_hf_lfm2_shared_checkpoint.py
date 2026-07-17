from __future__ import annotations

import hashlib
import json
import shutil

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Lfm2Config"):
    pytest.skip("Transformers build has no LFM2 support", allow_module_level=True)

import neural_forge.hf_lfm2_shared_checkpoint as checkpoint_module  # noqa: E402
from neural_forge.hf_lfm2_shared_checkpoint import (  # noqa: E402
    MANIFEST_FILENAME,
    Lfm2SharedCheckpointBudgetError,
    Lfm2SharedCheckpointError,
    Lfm2SharedCheckpointFormatError,
    export_lfm2_shared_checkpoint,
    load_lfm2_shared_checkpoint,
    plan_lfm2_shared_checkpoint,
)


def _tiny_lfm2(seed: int = 813):
    torch.manual_seed(seed)
    config = transformers.Lfm2Config(
        vocab_size=29,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv", "full_attention", "conv", "full_attention", "conv"],
        conv_L_cache=3,
        conv_bias=True,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return transformers.Lfm2ForCausalLM(config).eval()


def _tie_groups(model, groups=((0, 2), (1, 4))):
    for group in groups:
        source = model.model.layers[group[0]].feed_forward
        for role in ("w1", "w3", "w2"):
            parameter = getattr(source, role).weight
            for index in group[1:]:
                getattr(model.model.layers[index].feed_forward, role).weight = parameter
    return model


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


def test_raw_roundtrip_preserves_arbitrary_groups_counts_bytes_and_forward(
    tmp_path,
    monkeypatch,
) -> None:
    model = _tie_groups(_tiny_lfm2())
    model.model.layers[0].feed_forward.w1.weight.requires_grad_(False)
    groups = ((0, 2), (1, 4))
    plan = plan_lfm2_shared_checkpoint(model, groups)
    dense = _tiny_lfm2(seed=814)
    vocabulary_savings = model.get_input_embeddings().weight.numel()
    per_ffn = sum(
        getattr(model.model.layers[0].feed_forward, role).weight.numel()
        for role in ("w1", "w3", "w2")
    )

    assert plan.logical_layer_count == 5
    assert plan.shared_ffn_layer_groups == groups
    assert plan.ffn_shared_parameter_savings == 2 * per_ffn
    assert plan.vocabulary_shared_parameter_savings == vocabulary_savings
    assert plan.logical_parameter_count - plan.unique_parameter_count == (
        2 * per_ffn + vocabulary_savings
    )
    assert _logical_parameter_count(model) == plan.logical_parameter_count
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        plan.unique_parameter_count
    )
    assert sum(parameter.numel() for parameter in dense.parameters()) > (
        plan.unique_parameter_count
    )

    input_ids = torch.tensor([[1, 3, 5, 7]], dtype=torch.long)
    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False).logits
    checkpoint = tmp_path / "raw"
    manifest = export_lfm2_shared_checkpoint(
        model,
        checkpoint,
        shared_ffn_layer_groups=groups,
        parameter_budget=plan.unique_parameter_count,
        tensor_format="raw",
    )

    payload_path = checkpoint / "model.nfraw"
    assert payload_path.stat().st_size == manifest["storage"][
        "physical_tensor_bytes"
    ]
    assert payload_path.stat().st_size == sum(
        entry["nbytes"] for entry in manifest["tensors"].values()
    )
    assert manifest["storage"]["physical_parameter_bytes"] == sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    assert manifest["state_dict"]["logical_tensor_count"] == len(model.state_dict())
    assert manifest["state_dict"]["stored_tensor_count"] == len(manifest["tensors"])
    assert manifest["state_dict"]["ffn_alias_count"] == 6
    assert manifest["state_dict"]["vocabulary_alias_count"] == 1
    assert manifest["serialization"]["python_pickle"] is False
    assert manifest["parent_model_required"] is False
    assert manifest["tokenizer_required_for_reload"] is False
    assert manifest["external_weight_dependencies"] == []
    assert {path.name for path in checkpoint.iterdir()} == {
        "config.json",
        "generation_config.json",
        MANIFEST_FILENAME,
        "model.nfraw",
    }

    def unsafe_loader(*_args, **_kwargs):
        raise AssertionError("pickle/object loader must not be called")

    monkeypatch.setattr(torch, "load", unsafe_loader)
    loaded = load_lfm2_shared_checkpoint(checkpoint)
    with torch.no_grad():
        actual = loaded(input_ids=input_ids, use_cache=False).logits
    assert torch.equal(actual, expected)
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
    for group in groups:
        for role in ("w1", "w3", "w2"):
            parameters = [
                getattr(loaded.model.layers[index].feed_forward, role).weight
                for index in group
            ]
            assert all(parameter is parameters[0] for parameter in parameters)
    assert loaded.model.layers[0].feed_forward.w1.weight.requires_grad is False
    assert loaded.model.layers[2].feed_forward.w1.weight.requires_grad is False
    assert (
        loaded.model.layers[3].feed_forward.w1.weight
        is not loaded.model.layers[0].feed_forward.w1.weight
    )
    assert (
        loaded.model.layers[3].feed_forward.w1.weight
        is not loaded.model.layers[1].feed_forward.w1.weight
    )
    assert sum(parameter.numel() for parameter in loaded.parameters()) == (
        plan.unique_parameter_count
    )
    assert _logical_parameter_count(loaded) == plan.logical_parameter_count
    assert loaded.neural_forge_load_report["constructed_on_meta"] is True
    assert loaded.neural_forge_load_report["streamed_physical_tensors"] is True
    assert not any(
        marker in name.lower()
        for name, _ in loaded.named_modules()
        for marker in ("adapter", "teacher", "parent")
    )


def test_safetensors_stores_only_physical_tensors_and_reloads(tmp_path) -> None:
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    model = _tie_groups(_tiny_lfm2(seed=912), groups=((0, 4),))
    checkpoint = tmp_path / "safe"
    manifest = export_lfm2_shared_checkpoint(
        model,
        checkpoint,
        tensor_format="safetensors",
    )
    with safe_open(
        str(checkpoint / "model.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        assert set(handle.keys()) == set(manifest["tensors"])
        assert handle.metadata() == {
            "format": "neural-forge-lfm2-shared-ffn-checkpoint-v1"
        }
    assert manifest["topology"]["shared_ffn_layer_groups"] == [[0, 4]]
    assert manifest["state_dict"]["logical_to_stored_tensor"][
        "model.layers.4.feed_forward.w2.weight"
    ] == "model.layers.0.feed_forward.w2.weight"
    loaded = load_lfm2_shared_checkpoint(checkpoint)
    for role in ("w1", "w3", "w2"):
        assert (
            getattr(loaded.model.layers[0].feed_forward, role).weight
            is getattr(loaded.model.layers[4].feed_forward, role).weight
        )


def test_reload_preserves_bfloat16_rotary_runtime_buffers(tmp_path) -> None:
    model = _tie_groups(_tiny_lfm2()).to(dtype=torch.bfloat16)
    checkpoint = tmp_path / "bf16-rotary"
    export_lfm2_shared_checkpoint(
        model,
        checkpoint,
        tensor_format="raw",
    )
    loaded = load_lfm2_shared_checkpoint(checkpoint)
    source_buffers = dict(model.named_buffers(remove_duplicate=False))
    loaded_buffers = dict(loaded.named_buffers(remove_duplicate=False))
    for name in (
        "model.rotary_emb.inv_freq",
        "model.rotary_emb.original_inv_freq",
    ):
        assert loaded_buffers[name].dtype == torch.bfloat16
        assert loaded_buffers[name].dtype == source_buffers[name].dtype
        assert torch.equal(loaded_buffers[name], source_buffers[name])


def test_partial_or_undeclared_aliases_are_rejected_without_mutation(tmp_path) -> None:
    partial = _tiny_lfm2()
    partial.model.layers[2].feed_forward.w1.weight = (
        partial.model.layers[0].feed_forward.w1.weight
    )
    before = tuple(id(parameter) for parameter in partial.parameters())
    with pytest.raises(Lfm2SharedCheckpointError, match="Partial FFN aliasing"):
        export_lfm2_shared_checkpoint(partial, tmp_path / "partial", tensor_format="raw")
    assert tuple(id(parameter) for parameter in partial.parameters()) == before
    assert not (tmp_path / "partial").exists()

    tied = _tie_groups(_tiny_lfm2(), groups=((0, 2),))
    with pytest.raises(Lfm2SharedCheckpointError, match="differ from exact live"):
        export_lfm2_shared_checkpoint(
            tied,
            tmp_path / "wrong-declaration",
            shared_ffn_layer_groups=((0, 3),),
            tensor_format="raw",
        )
    assert not (tmp_path / "wrong-declaration").exists()


def test_manifest_payload_and_partial_alias_corruption_fail_closed(tmp_path) -> None:
    original = tmp_path / "original"
    export_lfm2_shared_checkpoint(
        _tie_groups(_tiny_lfm2(), groups=((0, 2),)),
        original,
        tensor_format="raw",
    )

    bad_manifest = tmp_path / "bad-manifest"
    shutil.copytree(original, bad_manifest)
    manifest_path = bad_manifest / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["standalone"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(Lfm2SharedCheckpointFormatError, match="Manifest SHA-256"):
        load_lfm2_shared_checkpoint(bad_manifest)

    bad_payload = tmp_path / "bad-payload"
    shutil.copytree(original, bad_payload)
    payload_path = bad_payload / "model.nfraw"
    payload = bytearray(payload_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    payload_path.write_bytes(payload)
    with pytest.raises(Lfm2SharedCheckpointFormatError, match="SHA-256"):
        load_lfm2_shared_checkpoint(bad_payload)

    partial_alias = tmp_path / "partial-alias"
    shutil.copytree(original, partial_alias)

    def break_coupled_alias(value):
        aliases = value["state_dict"]["logical_to_stored_tensor"]
        aliases["model.layers.2.feed_forward.w2.weight"] = (
            "model.layers.2.feed_forward.w2.weight"
        )
        value["state_dict"]["stored_tensor_count"] += 1
        value["state_dict"]["alias_count"] -= 1
        value["state_dict"]["ffn_alias_count"] -= 1

    _rewrite_manifest(partial_alias / MANIFEST_FILENAME, break_coupled_alias)
    with pytest.raises(Lfm2SharedCheckpointFormatError, match="complete declared"):
        load_lfm2_shared_checkpoint(partial_alias)

    wrong_shape = tmp_path / "wrong-shape"
    shutil.copytree(original, wrong_shape)

    def transpose_manifest_shape(value):
        entry = value["tensors"]["model.layers.0.feed_forward.w1.weight"]
        entry["shape"] = list(reversed(entry["shape"]))

    _rewrite_manifest(wrong_shape / MANIFEST_FILENAME, transpose_manifest_shape)
    with pytest.raises(Lfm2SharedCheckpointFormatError, match="fresh LFM2 topology"):
        load_lfm2_shared_checkpoint(wrong_shape)


def test_reload_budgets_fail_before_meta_construction(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "budget"
    manifest = export_lfm2_shared_checkpoint(
        _tie_groups(_tiny_lfm2(), groups=((0, 2),)),
        checkpoint,
        tensor_format="raw",
    )

    def meta_construction_must_not_run(*_args, **_kwargs):
        raise AssertionError("budget checks must precede meta construction")

    monkeypatch.setattr(
        checkpoint_module,
        "_build_meta_topology",
        meta_construction_must_not_run,
    )
    counts = manifest["parameter_counts"]
    memory = manifest["reload_memory"]
    limits = (
        {"max_parameter_count": counts["unique_parameter_count"] - 1},
        {"max_model_bytes": memory["model_parameter_bytes"] - 1},
        {"max_tensor_bytes": memory["largest_staging_tensor_bytes"] - 1},
        {
            "max_memory_bytes": (
                memory["bounded_peak_parameter_and_staging_bytes"] - 1
            )
        },
    )
    for kwargs in limits:
        with pytest.raises(Lfm2SharedCheckpointBudgetError, match="exceeding"):
            load_lfm2_shared_checkpoint(checkpoint, **kwargs)


def test_atomic_failures_leave_no_partial_checkpoint(tmp_path, monkeypatch) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("keep", encoding="ascii")
    with pytest.raises(Lfm2SharedCheckpointError, match="Refusing to replace"):
        export_lfm2_shared_checkpoint(
            _tie_groups(_tiny_lfm2()),
            existing,
            tensor_format="raw",
        )
    assert sentinel.read_text(encoding="ascii") == "keep"

    def fail_payload(*_args, **_kwargs):
        raise RuntimeError("injected payload failure")

    monkeypatch.setattr(checkpoint_module, "_save_tensor_payload", fail_payload)
    destination = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected payload failure"):
        export_lfm2_shared_checkpoint(
            _tie_groups(_tiny_lfm2()),
            destination,
            tensor_format="raw",
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))
