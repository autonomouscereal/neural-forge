"""Small native split-runtime tests; no tokenizer or real media claims."""
import importlib.util
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
