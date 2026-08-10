from __future__ import annotations

import inspect
import json
import socket
from pathlib import Path

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_direct_local import (
    BRIDGE_REFERENCE_FILENAME,
    Lfm2DirectLocalCheckpointIO,
    Lfm2DirectLocalError,
    PORTABLE_LOAD_RECEIPT_SCHEMA,
    RUNTIME_TENSOR_ALIAS_MANIFEST_SCHEMA,
    build_lfm2_direct_local_runtime_alias_manifest,
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_config,
    load_lfm2_direct_local_model,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_dynamic_blank_materializer import (
    materialize_lfm2_dynamic_blank_child,
)
from neural_forge.hf_lfm2_dynamic_physical_layout import (
    build_lfm2_dynamic_physical_layout,
)
from neural_forge.hf_tokenizer_bridge import (
    fingerprint_hf_tokenizer_semantics,
    seal_strict_tokenizer_bridge_artifact,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
tokenizers = pytest.importorskip("tokenizers")


def _tokenizer_package(path: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    path.mkdir()
    vocabulary = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "hello": 4,
        "world": 5,
        "small": 6,
        "model": 7,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    return load_lfm2_direct_local_tokenizer(path)


def _model():
    config = transformers.Lfm2Config(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    torch.manual_seed(8675309)
    return transformers.Lfm2ForCausalLM(config).eval()


class _AliasAdversary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        partial_storage = torch.arange(12, dtype=torch.float32)
        self.partial_left = torch.nn.Parameter(partial_storage[:8])
        self.partial_right = torch.nn.Parameter(partial_storage[4:])

        tied = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.tied_primary = tied
        self.tied_alias = tied
        self.equal_but_distinct = torch.nn.Parameter(tied.detach().clone())

        disjoint_storage = torch.arange(12, dtype=torch.float32)
        self.disjoint_left = torch.nn.Parameter(disjoint_storage[:3])
        self.disjoint_right = torch.nn.Parameter(disjoint_storage[9:])

        buffer_storage = torch.arange(10, dtype=torch.float64)
        self.register_buffer("persistent_window", buffer_storage[1:6])
        self.register_buffer(
            "transient_window",
            buffer_storage[4:9],
            persistent=False,
        )


class _EmptyAliasAdversary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared_storage = torch.empty(0, dtype=torch.float32)
        self.shared_left = torch.nn.Parameter(shared_storage[:])
        self.shared_right = torch.nn.Parameter(shared_storage.view(0))
        self.independent = torch.nn.Parameter(torch.empty(0, dtype=torch.float32))


def _entry_map(manifest):
    return {
        (entry["kind"], entry["name"]): entry for entry in manifest["state_entries"]
    }


def _overlap_map(manifest):
    return {
        frozenset(
            {
                (pair["left"]["kind"], pair["left"]["name"]),
                (pair["right"]["kind"], pair["right"]["name"]),
            }
        ): pair
        for pair in manifest["overlapping_byte_span_pairs"]
    }


def test_runtime_alias_manifest_seals_partial_tied_and_buffer_views() -> None:
    model = _AliasAdversary()
    manifest = build_lfm2_direct_local_runtime_alias_manifest(model)
    body = dict(manifest)
    seal = body.pop("manifest_sha256")
    entries = _entry_map(manifest)
    overlaps = _overlap_map(manifest)

    assert manifest["schema"] == RUNTIME_TENSOR_ALIAS_MANIFEST_SCHEMA
    assert seal == canonical_sha256(body)
    assert manifest["state_entry_count"] == len(entries)
    assert set(entries) == {
        *{
            ("parameter", name)
            for name, _value in model.named_parameters(remove_duplicate=False)
        },
        *{
            ("buffer", name)
            for name, _value in model.named_buffers(remove_duplicate=False)
        },
    }
    assert manifest["storage_addresses_serialized"] is False

    left = entries[("parameter", "partial_left")]
    right = entries[("parameter", "partial_right")]
    assert left["storage_group_id"] == right["storage_group_id"]
    assert left["storage_byte_span"] == {
        "start": 0,
        "end_exclusive": 32,
        "bounding_byte_count": 32,
        "empty": False,
    }
    assert right["storage_byte_span"] == {
        "start": 16,
        "end_exclusive": 48,
        "bounding_byte_count": 32,
        "empty": False,
    }
    partial_pair = overlaps[
        frozenset(
            {
                ("parameter", "partial_left"),
                ("parameter", "partial_right"),
            }
        )
    ]
    assert partial_pair["overlap_kind"] == "partial_span"
    assert partial_pair["overlap_byte_count"] == 16

    tied_primary = entries[("parameter", "tied_primary")]
    tied_alias = entries[("parameter", "tied_alias")]
    assert tied_primary["storage_group_id"] == tied_alias["storage_group_id"]
    tied_pair = overlaps[
        frozenset(
            {
                ("parameter", "tied_primary"),
                ("parameter", "tied_alias"),
            }
        )
    ]
    assert tied_pair["overlap_kind"] == "exact_span"

    persistent = entries[("buffer", "persistent_window")]
    transient = entries[("buffer", "transient_window")]
    assert persistent["persistent"] is True
    assert transient["persistent"] is False
    assert persistent["dtype"] == "torch.float64"
    assert persistent["storage_group_id"] == transient["storage_group_id"]


def test_runtime_alias_manifest_distinguishes_shared_disjoint_and_nonaliases() -> None:
    manifest = build_lfm2_direct_local_runtime_alias_manifest(_AliasAdversary())
    entries = _entry_map(manifest)
    overlaps = _overlap_map(manifest)

    tied = entries[("parameter", "tied_primary")]
    distinct = entries[("parameter", "equal_but_distinct")]
    assert tied["storage_group_id"] != distinct["storage_group_id"]

    disjoint_left = entries[("parameter", "disjoint_left")]
    disjoint_right = entries[("parameter", "disjoint_right")]
    assert disjoint_left["storage_group_id"] == disjoint_right["storage_group_id"]
    assert (
        frozenset(
            {
                ("parameter", "disjoint_left"),
                ("parameter", "disjoint_right"),
            }
        )
        not in overlaps
    )


def test_runtime_alias_manifest_preserves_zero_byte_storage_groups() -> None:
    manifest = build_lfm2_direct_local_runtime_alias_manifest(_EmptyAliasAdversary())
    entries = _entry_map(manifest)
    overlaps = _overlap_map(manifest)

    shared_left = entries[("parameter", "shared_left")]
    shared_right = entries[("parameter", "shared_right")]
    independent = entries[("parameter", "independent")]
    assert shared_left["storage_group_id"] == shared_right["storage_group_id"]
    assert shared_left["storage_group_id"] != independent["storage_group_id"]
    assert shared_left["storage_byte_span"]["empty"] is True
    assert shared_right["storage_byte_span"]["empty"] is True
    assert independent["storage_byte_span"]["empty"] is True
    assert overlaps == {}


def test_runtime_alias_manifest_is_deterministic_for_unchanged_state() -> None:
    model = _AliasAdversary()

    first = build_lfm2_direct_local_runtime_alias_manifest(model)
    second = build_lfm2_direct_local_runtime_alias_manifest(model)
    independently_allocated = build_lfm2_direct_local_runtime_alias_manifest(
        _AliasAdversary()
    )

    assert first == second == independently_allocated
    assert first["manifest_sha256"] == second["manifest_sha256"]


def _dynamic_untied_model():
    config = transformers.Lfm2Config(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
    )
    layout = build_lfm2_dynamic_physical_layout(
        hidden_size=8,
        vocab_size=8,
        head_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        conv_kernel_size=3,
        rope_theta=1_000_000.0,
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        attention_implementation="sdpa",
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=1,
        block_auto_adjust_ff_dim=False,
        layer_types=("conv", "full_attention"),
        mlp_intermediate_sizes=(5, 6),
        short_conv_internal_widths=(10, None),
        parent_parameter_elements=2_000,
        requested_dense_parameter_budget=1_200,
    )
    torch.manual_seed(104729)
    model = transformers.Lfm2ForCausalLM(config)
    materialize_lfm2_dynamic_blank_child(
        model,
        config,
        layout,
        expected_descriptor_sha256=layout.descriptor_sha256,
    )
    return model.eval(), layout


def _bridge(path: Path):
    tokenizer = _tokenizer_package(path)
    bridge = seal_strict_tokenizer_bridge_artifact(
        tokenizer,
        tokenizer,
        path,
        donor_row_capacity=8,
        target_row_capacity=8,
    )
    return tokenizer, bridge


def test_direct_local_checkpoint_roundtrip_never_uses_hf_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    tokenizer, bridge = _bridge(tmp_path / "tokenizer")
    model = _model()
    expected_state = fingerprint_lfm2_direct_local_state(model)
    io = Lfm2DirectLocalCheckpointIO(
        bridge,
        device="cpu",
        dtype=torch.float32,
    )
    child = tmp_path / "child"
    child.mkdir()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("HF/hub/network loader must never be called")

    monkeypatch.setattr(
        transformers.PreTrainedModel,
        "from_pretrained",
        classmethod(forbidden),
    )
    monkeypatch.setattr(
        transformers.PreTrainedTokenizerBase,
        "from_pretrained",
        classmethod(forbidden),
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)

    io.save(model, child)
    reloaded = io.load(child)
    reloaded_tokenizer = load_lfm2_direct_local_tokenizer(child)

    assert fingerprint_lfm2_direct_local_state(reloaded) == expected_state
    assert reloaded is not model
    assert reloaded.get_input_embeddings().weight is (
        reloaded.get_output_embeddings().weight
    )
    assert fingerprint_hf_tokenizer_semantics(reloaded_tokenizer) == (
        fingerprint_hf_tokenizer_semantics(tokenizer)
    )
    assert (child / "model.safetensors").is_file()
    assert (child / "tokenizer.json").is_file()
    assert (child / BRIDGE_REFERENCE_FILENAME).is_file()
    source = inspect.getsource(
        __import__(
            "neural_forge.hf_lfm2_direct_local",
            fromlist=["ignored"],
        )
    )
    assert ".from_pretrained" not in source
    assert "from transformers import Auto" not in source
    assert "AutoModel." not in source
    assert "AutoTokenizer." not in source


def test_portable_load_authenticates_save_policy_before_runtime_cast(
    tmp_path: Path,
) -> None:
    _tokenizer, bridge = _bridge(tmp_path / "tokenizer")
    model = _model()
    saved_state_sha256 = fingerprint_lfm2_direct_local_state(model)
    provenance_io = Lfm2DirectLocalCheckpointIO(
        bridge,
        device="cpu",
        dtype=torch.float32,
    )
    runtime_io = Lfm2DirectLocalCheckpointIO(
        bridge,
        device="cpu",
        dtype=torch.bfloat16,
    )
    child = tmp_path / "portable-child"
    child.mkdir()
    provenance_io.save(model, child)

    with pytest.raises(Lfm2DirectLocalError, match="reference.*stale"):
        runtime_io.load(child)

    loaded, receipt = provenance_io.load_authenticated_portable(
        child,
        runtime_device="cpu",
        runtime_dtype=torch.bfloat16,
    )
    body = receipt.as_dict()

    assert body["schema"] == PORTABLE_LOAD_RECEIPT_SCHEMA
    assert body["provenance_resource_fingerprint"] == (
        provenance_io.resource_fingerprint
    )
    assert body["provenance_device"] == "cpu"
    assert body["provenance_dtype"] == "torch.float32"
    assert body["runtime_device"] == "cpu"
    assert body["runtime_dtype"] == "torch.bfloat16"
    assert body["staged_state_sha256"] == saved_state_sha256
    assert body["runtime_state_sha256"] == (
        fingerprint_lfm2_direct_local_state(loaded)
    )
    assert body["runtime_state_sha256"] != saved_state_sha256
    assert body["dtype_conversion_performed"] is True
    assert body["device_relocation_performed"] is False
    assert body["runtime_policy_used_to_authenticate_legacy_reference"] is False
    assert body["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    assert {value.dtype for value in loaded.state_dict().values()} == {
        torch.bfloat16
    }


def test_portable_load_rejects_incorrect_declared_provenance(
    tmp_path: Path,
) -> None:
    _tokenizer, bridge = _bridge(tmp_path / "tokenizer")
    authentic = Lfm2DirectLocalCheckpointIO(
        bridge,
        device="cpu",
        dtype=torch.float32,
    )
    child = tmp_path / "portable-child"
    child.mkdir()
    authentic.save(_model(), child)
    incorrect = Lfm2DirectLocalCheckpointIO(
        bridge,
        device="cpu",
        dtype=torch.bfloat16,
    )

    with pytest.raises(Lfm2DirectLocalError, match="reference.*stale"):
        incorrect.load_authenticated_portable(
            child,
            runtime_device="cpu",
            runtime_dtype=torch.float32,
        )


def test_portable_load_same_dtype_preserves_state_bytes(
    tmp_path: Path,
) -> None:
    _tokenizer, bridge = _bridge(tmp_path / "tokenizer")
    io = Lfm2DirectLocalCheckpointIO(
        bridge,
        device="cpu",
        dtype=torch.float32,
    )
    child = tmp_path / "portable-child"
    child.mkdir()
    io.save(_model(), child)

    _loaded, receipt = io.load_authenticated_portable(
        child,
        runtime_device="cpu",
        runtime_dtype=torch.float32,
    )

    assert receipt.staged_state_sha256 == receipt.runtime_state_sha256
    assert receipt.dtype_conversion_performed is False
    assert receipt.device_relocation_performed is False


def test_direct_local_roundtrip_materializes_embedded_dynamic_layout_before_load(
    tmp_path: Path,
) -> None:
    _tokenizer, bridge = _bridge(tmp_path / "tokenizer")
    model, layout = _dynamic_untied_model()
    expected_state = fingerprint_lfm2_direct_local_state(model)
    io = Lfm2DirectLocalCheckpointIO(
        bridge,
        device="cpu",
        dtype=torch.float32,
    )
    child = tmp_path / "dynamic-child"
    child.mkdir()

    io.save(model, child)
    loaded = io.load(child)

    assert loaded is not model
    assert fingerprint_lfm2_direct_local_state(loaded) == expected_state
    assert tuple(loaded.model.layers[0].conv.in_proj.weight.shape) == (30, 8)
    assert tuple(loaded.model.layers[0].conv.out_proj.weight.shape) == (8, 10)
    assert tuple(loaded.model.layers[0].feed_forward.w1.weight.shape) == (5, 8)
    assert tuple(loaded.model.layers[1].feed_forward.w2.weight.shape) == (8, 6)
    assert (
        loaded.config.neural_forge_dynamic_physical_layout_sha256
        == layout.descriptor_sha256
    )
    inputs = torch.tensor([[2, 4, 5, 3]], dtype=torch.long)
    with torch.inference_mode():
        expected = model(inputs, use_cache=False).logits
        observed = loaded(inputs, use_cache=False).logits
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_direct_local_config_preserves_legacy_block_ff_dim_across_resave(
    tmp_path: Path,
) -> None:
    _tokenizer, bridge = _bridge(tmp_path / "tokenizer")
    model = _model()
    model.config.block_ff_dim = model.config.intermediate_size
    io = Lfm2DirectLocalCheckpointIO(
        bridge,
        device="cpu",
        dtype=torch.float32,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    io.save(model, first)
    loaded = io.load(first)
    io.save(loaded, second)

    assert loaded.config.block_ff_dim == loaded.config.intermediate_size == 12
    assert loaded.config.to_dict()["block_ff_dim"] == 12
    assert (second / "config.json").read_bytes() == (
        first / "config.json"
    ).read_bytes()


@pytest.mark.parametrize("block_ff_dim", (True, 0, 11, 12.5, "12"))
def test_direct_local_config_rejects_invalid_or_conflicting_block_ff_dim(
    tmp_path: Path,
    block_ff_dim: object,
) -> None:
    directory = tmp_path / "model"
    directory.mkdir()
    payload = _model().config.to_dict()
    payload["block_ff_dim"] = block_ff_dim
    (directory / "config.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(Lfm2DirectLocalError, match="block_ff_dim"):
        load_lfm2_direct_local_config(directory)


def test_direct_local_loader_rejects_empty_path_and_tokenizer_drift(
    tmp_path: Path,
) -> None:
    _tokenizer, bridge = _bridge(tmp_path / "tokenizer")
    io = Lfm2DirectLocalCheckpointIO(
        bridge,
        device="cpu",
        dtype=torch.float32,
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Lfm2DirectLocalError):
        io.load(empty)
    assert not tuple(empty.iterdir())

    child = tmp_path / "child"
    child.mkdir()
    io.save(_model(), child)
    (child / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(
        Lfm2DirectLocalError,
        match="differs from the bridge",
    ):
        io.load(child)


def test_direct_local_saver_requires_real_empty_destination(
    tmp_path: Path,
) -> None:
    _tokenizer, bridge = _bridge(tmp_path / "tokenizer")
    io = Lfm2DirectLocalCheckpointIO(
        bridge,
        device="cpu",
        dtype=torch.float32,
    )
    child = tmp_path / "child"
    child.mkdir()
    (child / "unrelated").write_text("keep", encoding="utf-8")
    with pytest.raises(
        Lfm2DirectLocalError,
        match="real empty directory",
    ):
        io.save(_model(), child)
    assert (child / "unrelated").read_text(encoding="utf-8") == "keep"


def test_direct_local_model_loader_rejects_nonlocal_source() -> None:
    with pytest.raises(Lfm2DirectLocalError, match="local directory"):
        load_lfm2_direct_local_model(
            "https://example.invalid/model",
            device="cpu",
            dtype=torch.float32,
        )


def _indexed_checkpoint(
    path: Path,
    *,
    nested_shards: bool = True,
) -> tuple[object, Path]:
    from safetensors.torch import save_file

    path.mkdir()
    model = _model()
    (path / "config.json").write_text(
        json.dumps(model.config.to_dict()),
        encoding="utf-8",
    )
    if nested_shards:
        shard_directory = path / "weights"
        shard_directory.mkdir()
        shard_names = (
            "weights/model-00001-of-00002.safetensors",
            "weights/model-00002-of-00002.safetensors",
        )
    else:
        shard_names = (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        )
    states = tuple(
        (name, value.detach().clone())
        for name, value in model.state_dict().items()
    )
    first = dict(states[::2])
    second = dict(states[1::2])
    save_file(first, str(path / shard_names[0]))
    save_file(second, str(path / shard_names[1]))
    weight_map = {
        name: shard_names[index % 2]
        for index, (name, _value) in enumerate(states)
    }
    index_path = path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    return model, index_path


def test_direct_local_model_loader_accepts_safe_nested_index_shards(
    tmp_path: Path,
) -> None:
    model, _index_path = _indexed_checkpoint(tmp_path / "indexed")

    loaded = load_lfm2_direct_local_model(
        tmp_path / "indexed",
        device="cpu",
        dtype=torch.float32,
    )

    assert fingerprint_lfm2_direct_local_state(loaded) == (
        fingerprint_lfm2_direct_local_state(model)
    )


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../external.safetensors",
        "C:\\outside\\model.safetensors",
        "/outside/model.safetensors",
    ),
)
def test_direct_local_model_loader_rejects_unsafe_index_shard_paths(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    _model_value, index_path = _indexed_checkpoint(tmp_path / "indexed")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    first_key = next(iter(index["weight_map"]))
    index["weight_map"][first_key] = unsafe_name
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(Lfm2DirectLocalError, match="unsafe shard path"):
        load_lfm2_direct_local_model(
            tmp_path / "indexed",
            device="cpu",
            dtype=torch.float32,
        )


def test_direct_local_model_loader_rejects_index_shard_symlink_escape(
    tmp_path: Path,
) -> None:
    from safetensors.torch import save_file

    snapshot = (
        tmp_path
        / "models--LiquidAI--LFM2"
        / "snapshots"
        / ("a" * 40)
    )
    snapshot.parent.mkdir(parents=True)
    _model_value, index_path = _indexed_checkpoint(
        snapshot,
        nested_shards=False,
    )
    external = tmp_path / "external.safetensors"
    save_file({"value": torch.ones(1)}, str(external))
    link = snapshot / "escape.safetensors"
    try:
        link.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"Local symlink creation is unavailable: {exc}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    first_key = next(iter(index["weight_map"]))
    index["weight_map"][first_key] = "weights/escape.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(
        Lfm2DirectLocalError,
        match="escapes its checkpoint directory",
    ):
        load_lfm2_direct_local_model(
            snapshot,
            device="cpu",
            dtype=torch.float32,
        )


def test_direct_local_model_loader_accepts_hf_cache_blob_shard_links(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models--LiquidAI--LFM2"
    snapshot = model_root / "snapshots" / ("a" * 40)
    snapshot.parent.mkdir(parents=True)
    model, index_path = _indexed_checkpoint(
        snapshot,
        nested_shards=False,
    )
    blobs = model_root / "blobs"
    blobs.mkdir()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    for offset, shard_name in enumerate(shard_names, start=1):
        blob_name = f"{offset:064x}"
        shard = snapshot / shard_name
        shard.replace(blobs / blob_name)
        try:
            shard.symlink_to(Path("../../blobs") / blob_name)
        except OSError as exc:
            pytest.skip(f"Local symlink creation is unavailable: {exc}")

    loaded = load_lfm2_direct_local_model(
        snapshot,
        device="cpu",
        dtype=torch.float32,
    )

    assert fingerprint_lfm2_direct_local_state(loaded) == (
        fingerprint_lfm2_direct_local_state(model)
    )
