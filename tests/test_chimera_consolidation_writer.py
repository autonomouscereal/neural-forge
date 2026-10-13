"""Actual inverse P1 mapping into a complete tiny native Chimera forward."""
import json
from pathlib import Path

import pytest
from experiments.chimera_recovery import consolidation_writer as W
from test_chimera_consolidation_runtime import fixture, replay, backward


def test_true_native_chimera_forward_after_joint_optimizer_and_inverse_writer(tmp_path):
    torch, M, _, construct = fixture(tmp_path)
    union = construct()
    model = union["hybrid"]
    rows, sources = replay(tmp_path, torch, M, model)
    engine = W.R.UnionEngine(union, rows, sources, tmp_path, accumulation=2)
    run = backward(torch, M)
    assert engine.window(run)["applied"]
    engine.checkpoint(tmp_path / "partial")
    with pytest.raises(W.Error, match="complete declared-budget"):
        W.export_candidate(engine, tmp_path / "partial", tmp_path / "premature")
    assert not (tmp_path / "premature").exists()
    assert engine.window(run)["applied"]
    patches = dict(W.iter_replacements(engine))
    reader = model.engine.source.reader
    original = {name: reader._raw(name).detach().clone() for name in reader.shards.names()}
    published = {**original, **patches}
    reference = model._chimera_model(model.cfg).to(torch.float32).eval()
    reference.load_state_dict(published, strict=True)
    for kind in ("ace", "h3", "ocr", "hy3d", "text"):
        segment = M.tiny_segment(model.cfg, kind, rows=4, seed=171)
        a, b = model.forward(segment), M.chimera_reference(reference, segment)
        torch.testing.assert_close(a["normed"], b["hidden"], rtol=5e-5, atol=5e-6)
        if kind != "text":
            torch.testing.assert_close(a["emission"], b["emission"], rtol=5e-5, atol=5e-6)
    ids = torch.tensor([[1, 2, 3, model.music_band_begin + 1, model.music_band_begin + 2]])
    with torch.no_grad():
        actual = model.forward(dict(ids=ids))["normed"]
        result = reference(ids)
        torch.testing.assert_close(actual, result["hidden"], rtol=5e-5, atol=5e-6)
        music = model._recovery_d4_music
        logits = torch.nn.functional.linear(actual.float(), music["head"])
        torch.testing.assert_close(logits, result["logits"][:, :, model.music_band_begin:model.music_band_begin + 10], rtol=5e-5, atol=5e-6)
    # Full fused FFN halves and KDA concatenated Q/K/V change. The genuine D2
    # builder explicitly forbids GQA q/k adapters, so its interleaved query/gate
    # source parts and key projections must remain exactly source-frozen.
    assert any("ffn.up.weight" in name for name in patches)
    assert any("mixer.q_proj.weight" in name for name in patches)
    for index, kind in enumerate(reader.block_kinds):
        if kind != "kda":
            for suffix in ("q_proj.weight", "gate_proj.weight", "k_proj.weight"):
                name = f"blocks.{reader.text_blocks[index]}.mixer.{suffix}"
                assert name not in patches and torch.equal(published[name], original[name]), name
    begin, end = model.music_band_begin, model.music_band_begin + 10
    for name in ("embed.weight", "lm_head.weight"):
        assert torch.equal(patches[name][:begin], original[name][:begin])
        assert torch.equal(patches[name][end:], original[name][end:])
    engine.checkpoint(tmp_path / "completed")
    artifact = W.export_candidate(engine, tmp_path / "completed", tmp_path / "replacements")
    manifest = json.loads(Path(artifact["path"]).read_text())
    assert manifest["fixture"] and manifest["stage_training_complete"] and not manifest["production_ready"]
    assert not manifest["complete_native_checkpoint_composed"]
    from safetensors.torch import load_file
    for row in manifest["updates"]:
        path = W.I.pinned(row["file"], tmp_path / "replacements")
        values = load_file(str(path))
        assert set(values) == {row["target"]}
        assert torch.equal(values[row["target"]], patches[row["target"]])
    with pytest.raises(W.Error, match="preserve"):
        W.export_candidate(engine, tmp_path / "completed", tmp_path / "replacements")
    # Current trained state cannot silently replace the recorded checkpoint.
    with torch.no_grad():
        next(iter(engine.census.values())).add_(.02)
    with pytest.raises(W.Error, match="differs from sealed"):
        W.export_candidate(engine, tmp_path / "completed", tmp_path / "changed")


def test_pinned_actual_model_files_refuse_same_size_identity_mutation(tmp_path):
    torch, M, _, construct = fixture(tmp_path)
    union = construct()
    rows, sources = replay(tmp_path, torch, M, union["hybrid"])
    engine = W.R.UnionEngine(union, rows, sources, tmp_path, accumulation=2)
    pinned = next(ref for name, ref in engine.identity["rebase"]["model_files"].items() if name.endswith(".safetensors"))
    path = Path(pinned["path"])
    # Append within the fixture's own temporary directory: no registered model
    # or currently read tensor bytes are changed. Full file SHA catches it.
    with path.open("ab") as handle:
        handle.write(b"tampered source trailer")
    with pytest.raises(W.I.Error, match="missing or changed"):
        engine.window(backward(torch, M))
    assert engine.cursor == engine.step == 0 and not engine.optimizer.state
