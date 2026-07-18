from __future__ import annotations

import inspect
import json
import socket
from pathlib import Path

import pytest

from neural_forge.hf_lfm2_direct_local import (
    BRIDGE_REFERENCE_FILENAME,
    Lfm2DirectLocalCheckpointIO,
    Lfm2DirectLocalError,
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_model,
    load_lfm2_direct_local_tokenizer,
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
