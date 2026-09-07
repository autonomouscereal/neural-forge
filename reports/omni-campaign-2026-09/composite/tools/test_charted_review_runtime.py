"""Small native split-runtime tests; no tokenizer or real media claims."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from chimera_omni.charted import ChartedBlockSpec, H3_FP32_PARAMETERS
from chimera_omni.config import ChimeraConfig
from chimera_omni.importing.safetensors_io import write_safetensors
from chimera_omni.modeling import ChimeraOmniModel

spec = importlib.util.spec_from_file_location("charted_review_runtime", Path(__file__).with_name("run_omni_frozen_review.py"))
review = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = review
spec.loader.exec_module(review)


def make_model(h3=False):
    source = ChartedBlockSpec(
        layer_index=1, source="h3" if h3 else "fixture", component="fixture.mlp",
        type_ids=(2, 3) if h3 else (1,), width=24, heads=2, kv_heads=2,
        head_dim=12, mlp_width=32, modulation="h3_curve6" if h3 else "none",
        norm_eps=1e-5, h3_curve_grid=4 if h3 else 0,
        h3_curve_dim=2 if h3 else 0, h3_rope_freqs=1 if h3 else 0,
    )
    cfg = ChimeraConfig(
        vocab_size=320, hidden_size=32, intermediate_size=48, num_layers=3,
        num_heads=4, num_kv_heads=2, attention_head_dim=8, gqa_rope_dim=2,
        max_sequence_length=128, block_pattern=("gated_gqa", "charted", "gated_gqa"),
        residual_mode="none", native_feature_dim=16, ledger_kind="fixed_band",
        ledger_base_vocab_size=256,
        ledger_bands=(("control", 8), ("media", 16), ("grounding", 8), ("sync", 8),
                      ("tool", 8), ("music_codes", 16)), charted_blocks=(source,),
    )
    cfg.validate()
    torch.manual_seed(178)
    model = ChimeraOmniModel(cfg).eval()
    if h3:
        with torch.no_grad():
            for name in H3_FP32_PARAMETERS:
                parameter = model.blocks[1].charted.get_parameter(name)
                parameter.copy_(torch.randn_like(parameter) * 0.05)
    return model, cfg


def decoder(model, cfg):
    placement = review.placement_for(cfg, [1, 2], ["cpu", "cpu"])
    return review.SplitDecoder(model, cfg, placement, ledger_band_logit_mask=False)


@pytest.mark.parametrize("h3", [False, True])
def test_charted_text_prefill_and_incremental_decode_equal_reference(h3):
    model, cfg = make_model(h3)
    split = decoder(model, cfg)
    ids = torch.tensor([[2, 5, 8, 12]])
    with torch.no_grad():
        expected = model.forward_causal_with_cache(ids)
        actual = split.forward(ids)
        assert torch.equal(actual["logits"], expected["logits"])
        token = torch.tensor([[13]])
        expected = model.forward_causal_with_cache(token, cache=expected["cache"])
        actual = split.forward(token, actual["cache"])
        assert torch.equal(actual["logits"], expected["logits"])


def media(h3=False):
    return {
        "features": torch.randn(1, 2, 16), "type_ids": torch.tensor([[2, 3] if h3 else [1, 1]]),
        "positions": torch.zeros(1, 2, 4), "indices": torch.tensor([[0, 1]]),
        "valid_mask": torch.ones(1, 2, dtype=torch.bool),
    }


def h3_media(sequence=4):
    m = media(True)
    base = 0.3
    m["noise_levels"] = torch.tensor([[12 * base / (1 + 11 * base), 3 * base / (1 + 2 * base)]])
    m["target_kind"] = "x0"
    m["update_mask"] = torch.ones(1, 2, dtype=torch.bool)
    m["charted_source_positions"] = {"h3": torch.arange(sequence * 3).reshape(1, sequence, 3).double()}
    return m


def whole_model_prefill(model, ids, m):
    return model(
        ids, native_media_features=m["features"], native_media_type_ids=m["type_ids"],
        native_media_positions=m["positions"], native_media_indices=m["indices"],
        native_media_valid_mask=m["valid_mask"], native_media_noise_levels=m.get("noise_levels"),
        native_media_target_kind=m.get("target_kind", "reconstruction"),
        native_media_update_mask=m.get("update_mask"), native_media_bidirectional_refinement=False,
        charted_source_positions=m.get("charted_source_positions"),
    )


@pytest.mark.parametrize("h3", [False, True])
def test_charted_media_prefill_matches_whole_model_and_reaches_block(h3):
    model, cfg = make_model(h3)
    split = decoder(model, cfg)
    m = media(h3)
    ids = torch.tensor([[264, 265, 8, 12]])
    if h3:
        base = 0.3
        m["noise_levels"] = torch.tensor([[12 * base / (1 + 11 * base), 3 * base / (1 + 2 * base)]])
        m["target_kind"] = "x0"
        m["update_mask"] = torch.ones(1, 2, dtype=torch.bool)
        m["charted_source_positions"] = {"h3": torch.arange(12).reshape(1, 4, 3).double()}
    observed = []
    hook = model.blocks[1].charted.register_forward_hook(lambda module, args, out: observed.append(out.clone()))
    with torch.no_grad():
        expected = model(ids, native_media_features=m["features"], native_media_type_ids=m["type_ids"],
                         native_media_positions=m["positions"], native_media_indices=m["indices"],
                         native_media_valid_mask=m["valid_mask"], native_media_noise_levels=m.get("noise_levels"),
                         native_media_target_kind=m.get("target_kind", "reconstruction"),
                         native_media_update_mask=m.get("update_mask"), native_media_bidirectional_refinement=False,
                         charted_source_positions=m.get("charted_source_positions"))
        actual = split.forward(ids, media=m)
    hook.remove()
    assert len(observed) == 2 and observed[1][:, :2].abs().sum() > 0
    assert torch.count_nonzero(observed[1][:, 2:]) == 0
    torch.testing.assert_close(actual["logits"], expected["logits"], atol=0, rtol=0)
    with pytest.raises(review.Refusal, match="initial prefill"):
        split.forward(ids, actual["cache"], media=m)


def test_h3_missing_actual_noise_and_source_coordinates_refuse():
    model, cfg = make_model(True)
    split = decoder(model, cfg)
    ids = torch.tensor([[264, 265, 8, 12]])
    m = media(True)
    with pytest.raises(review.Refusal, match="actual per-type sigma"):
        split.forward(ids, media=m)
    m["noise_levels"] = torch.ones(1, 2)
    m["target_kind"] = "x0"
    m["update_mask"] = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="source-coordinate"):
        split.forward(ids, media=m)


@pytest.mark.parametrize("h3", [False, True])
def test_meta_split_load_materializes_type_buffers_and_preserves_fp32_islands(tmp_path, h3):
    model, cfg = make_model(h3)
    model.bfloat16()
    root = tmp_path / "child"
    write_safetensors(model.state_dict(), root, config=cfg.to_manifest(), ledger=cfg.ledger().to_manifest(), chimera_commit="fixture")
    placement = review.placement_for(cfg, [3], ["cpu"])
    loaded, _, receipt = review.load_chimera_split(root, placement=placement, dtype=torch.bfloat16, verify_tensors=True)
    assert loaded.blocks[1].charted.routed_type_ids.tolist() == list(cfg.charted_blocks[0].type_ids)
    assert not any(t.is_meta for t in loaded.buffers())
    expected_names = review.fp32_source_names(cfg)
    assert set(receipt["fp32_source_coefficients"]) == expected_names
    for name in expected_names:
        actual = loaded.get_parameter(name)
        assert actual.dtype == torch.float32
        assert torch.equal(actual, model.get_parameter(name))
    with torch.no_grad():
        ids = torch.tensor([[1, 2, 3]])
        assert torch.equal(decoder(loaded, cfg).forward(ids)["logits"], model.forward_causal_with_cache(ids)["logits"])


def test_streamed_charted_lane_remains_explicitly_unsupported(tmp_path):
    model, cfg = make_model()
    root = tmp_path / "child"
    write_safetensors(model.state_dict(), root, config=cfg.to_manifest(), ledger=cfg.ledger().to_manifest(), chimera_commit="fixture")
    with pytest.raises(review.Refusal, match="streamed proxy"):
        review.StreamedChimera(root, device="cpu", dtype=torch.float32, ledger_band_logit_mask=False)


@pytest.mark.parametrize("h3", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_padded_media_slots_match_native_validity_and_never_become_text(h3, dtype):
    model, cfg = make_model(h3)
    model.to(dtype=dtype)
    split = decoder(model, cfg)
    m = h3_media() if h3 else media()
    m["valid_mask"][0, 1] = False
    m["type_ids"][0, 1] = 0
    if h3:
        m["update_mask"][0, 1] = False
        m["noise_levels"][0, 1] = 0
    ids = torch.tensor([[264, 265, 8, 12]])
    with torch.no_grad():
        expected = whole_model_prefill(model, ids, m)
        actual = split.forward(ids, media=m)
        torch.testing.assert_close(actual["logits"], expected["logits"], atol=0, rtol=0)
        assert torch.count_nonzero(actual["hidden"][:, 1]) == 0
        changed = ids.clone()
        changed[0, 1] = 14
        # A padded media slot is invalid regardless of its input token embedding.
        another = split.forward(changed, media=m)
        assert torch.equal(actual["logits"], another["logits"])


def test_non_h3_media_prefill_incremental_decode_still_matches_full_prefix():
    model, cfg = make_model(False)
    split = decoder(model, cfg)
    ids, token, m = torch.tensor([[264, 265, 8, 12]]), torch.tensor([[13]]), media()
    with torch.no_grad():
        first = split.forward(ids, media=m)
        assert not isinstance(first["cache"], review.UncacheableMediaPrefill)
        actual = split.forward(token, first["cache"])
        expected = whole_model_prefill(model, torch.cat((ids, token), dim=1), m)
    assert torch.equal(actual["logits"][:, -1], expected["logits"][:, -1])


def test_h3_prefill_returns_tensor_free_sentinel_and_refuses_cached_continuation(monkeypatch):
    model, cfg = make_model(True)
    split = decoder(model, cfg)
    m, ids = h3_media(), torch.tensor([[264, 265, 8, 12]])
    def no_retained_cache(*args):
        pytest.fail("H3 prefill must not retain a reusable KV/history container")
    monkeypatch.setattr(split, "_cache_class", no_retained_cache)
    with torch.no_grad():
        actual = split.forward(ids, media=m)
        expected = whole_model_prefill(model, ids, m)
    assert torch.equal(actual["logits"], expected["logits"])
    marker = actual["cache"]
    assert isinstance(marker, review.UncacheableMediaPrefill)
    assert not hasattr(marker, "__dict__") and not hasattr(marker, "block_caches")
    with pytest.raises(review.Refusal, match="full-context native forward"):
        split.forward(torch.tensor([[13]]), marker)
    # Fresh full-context evaluation remains faithful; no H3 context is frozen.
    extended = torch.cat((ids, torch.tensor([[13]])), dim=1)
    m["charted_source_positions"] = {"h3": torch.arange(15).reshape(1, 5, 3).double()}
    with torch.no_grad():
        rerun = split.forward(extended, media=m)
        reference = whole_model_prefill(model, extended, m)
    assert torch.equal(rerun["logits"], reference["logits"])


def test_h3_model_without_routed_media_does_not_forbid_text_cache():
    model, cfg = make_model(True)
    split = decoder(model, cfg)
    # Image rows are not H3's AV rows and therefore do not activate joint H3 attention.
    with torch.no_grad():
        first = split.forward(torch.tensor([[264, 265, 8, 12]]), media=media(False))
        assert not isinstance(first["cache"], review.UncacheableMediaPrefill)
        split.forward(torch.tensor([[13]]), first["cache"])


def written_fixture(tmp_path, h3=False, *, bad_fp32_island=False):
    model, cfg = make_model(h3)
    model.bfloat16()
    state = model.state_dict()
    if bad_fp32_island:
        name = "blocks.1.charted." + H3_FP32_PARAMETERS[0]
        state[name] = state[name].bfloat16()
    root = tmp_path / "checkpoint"
    write_safetensors(state, root, config=cfg.to_manifest(), ledger=cfg.ledger().to_manifest(), chimera_commit="fixture")
    path = root / "model.safetensors.index.json"
    return cfg, root, path, json.loads(path.read_text())


@pytest.mark.parametrize("mode", ["missing", "partial", "extra", "malformed", "nonstring"])
def test_requested_verification_refuses_incomplete_or_invalid_hashes_before_loading(tmp_path, monkeypatch, mode):
    cfg, root, path, index = written_fixture(tmp_path)
    hashes = index["metadata"]["tensor_sha256"]
    name = next(iter(hashes))
    if mode == "missing":
        del index["metadata"]["tensor_sha256"]
    elif mode == "partial":
        del hashes[name]
    elif mode == "extra":
        hashes["unindexed.tensor"] = "a" * 64
    elif mode == "malformed":
        hashes[name] = "z" * 64
    else:
        hashes[name] = 42
    path.write_text(json.dumps(index))
    import chimera_omni.modeling
    def refuse_allocation(*args, **kwargs):
        pytest.fail("incomplete verification metadata reached model construction")
    monkeypatch.setattr(chimera_omni.modeling, "ChimeraOmniModel", refuse_allocation)
    with pytest.raises(review.Refusal, match="tensor_sha256|SHA256"):
        review.load_chimera_split(root, placement=review.placement_for(cfg, [3], ["cpu"]), dtype=torch.bfloat16, verify_tensors=True)


def test_full_hash_coverage_is_verified_and_counted_including_h3_source_islands(tmp_path):
    cfg, root, _, index = written_fixture(tmp_path, h3=True)
    _, _, receipt = review.load_chimera_split(root, placement=review.placement_for(cfg, [3], ["cpu"]), dtype=torch.bfloat16, verify_tensors=True)
    count = len(index["weight_map"])
    assert receipt["tensor_sha256_verified"] is True
    assert receipt["tensor_sha256_verified_count"] == receipt["tensor_sha256_total_stored"] == count
    assert len(receipt["fp32_source_coefficients"]) == len(H3_FP32_PARAMETERS)


def test_wrong_hash_with_full_coverage_refuses(tmp_path):
    cfg, root, path, index = written_fixture(tmp_path)
    hashes = index["metadata"]["tensor_sha256"]
    hashes[next(iter(hashes))] = "0" * 64
    path.write_text(json.dumps(index))
    with pytest.raises(review.Refusal, match="does not match the sha256"):
        review.load_chimera_split(root, placement=review.placement_for(cfg, [3], ["cpu"]), dtype=torch.bfloat16, verify_tensors=True)


def test_opt_out_never_claims_verification_when_metadata_is_absent(tmp_path):
    cfg, root, path, index = written_fixture(tmp_path)
    del index["metadata"]["tensor_sha256"]
    path.write_text(json.dumps(index))
    _, _, receipt = review.load_chimera_split(root, placement=review.placement_for(cfg, [3], ["cpu"]), dtype=torch.bfloat16, verify_tensors=False)
    assert receipt["tensor_sha256_verified"] is False
    assert receipt["tensor_sha256_verified_count"] == 0
    assert receipt["tensor_sha256_total_stored"] == len(index["weight_map"])


def test_h3_source_coefficients_refuse_bf16_storage_even_with_valid_hashes(tmp_path):
    cfg, root, _, _ = written_fixture(tmp_path, h3=True, bad_fp32_island=True)
    with pytest.raises(review.Refusal, match="must be stored in FP32"):
        review.load_chimera_split(root, placement=review.placement_for(cfg, [3], ["cpu"]), dtype=torch.bfloat16, verify_tensors=True)


def test_tied_alias_cannot_overwrite_a_verified_stored_tensor(tmp_path):
    cfg, root, path, index = written_fixture(tmp_path)
    name = next(iter(index["weight_map"]))
    index["metadata"]["tied_weights"][name] = name
    path.write_text(json.dumps(index))
    with pytest.raises(review.Refusal, match="without overwriting stored bytes"):
        review.load_chimera_split(root, placement=review.placement_for(cfg, [3], ["cpu"]), dtype=torch.bfloat16, verify_tensors=True)


@pytest.mark.parametrize("mode", ["unindexed_stored_tensor", "missing_indexed_tensor"])
def test_declared_hash_coverage_must_also_match_actual_stored_tensors(tmp_path, mode):
    cfg, root, path, index = written_fixture(tmp_path)
    name = next(iter(index["weight_map"]))
    if mode == "unindexed_stored_tensor":
        del index["weight_map"][name]
        del index["metadata"]["tensor_sha256"][name]
        message = "absent from its declared indexed shard"
    else:
        index["weight_map"]["ghost.weight"] = index["weight_map"][name]
        index["metadata"]["tensor_sha256"]["ghost.weight"] = "0" * 64
        message = "indexed tensors were not verified"
    path.write_text(json.dumps(index))
    with pytest.raises(review.Refusal, match=message):
        review.load_chimera_split(root, placement=review.placement_for(cfg, [3], ["cpu"]), dtype=torch.bfloat16, verify_tensors=True)
