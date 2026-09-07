"""Synthetic CPU mechanics only: native blocks/loaders, not model capability."""
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
import c17_capture_host as tool
import run_omni_frozen_review as native


@pytest.fixture
def corpus():
    evidence = {"schema": "neural-forge-omni-evidence-v1", "rows": [{"batch_index": 20, "ids": [3, 2, 1], "split": "held"},
                         {"batch_index": 7, "ids": [1, 2, 3, 4], "split": "fit"}]}
    boundary = {"schema": "neural-forge-k3-boundary-map-v2", "sequence_length": 8,
                "evidence_manifest_file_sha256": "a" * 64,
                "rows": [{"row_index": i, "file": "k3_input_ids", "source": "evidence", "split": split,
                          "qwen_tokens_real": n, "k3_tokens_real": n + 1, "pairs": [[0, 1], [n - 1, n]]}
                         for i, (split, n) in enumerate([("fit", 4), ("held", 3)])]}
    boundary["boundary_map_sha256"] = tool.canonical_sha(boundary)
    return evidence, boundary


def seal(boundary):
    boundary["boundary_map_sha256"] = tool.canonical_sha({k: v for k, v in boundary.items() if k != "boundary_map_sha256"})


def test_position_and_split_identity(corpus):
    rows = tool.evidence_rows(*corpus, "a" * 64)
    assert [r["batch_index"] for r in rows] == [7, 20]
    assert [r["row_index"] for r in rows] == [0, 1]
    assert [r["split"] for r in rows] == ["fit", "held"]
    assert rows[1]["pairs"] == [(0, 1), (2, 3)]


@pytest.mark.parametrize("fault", ["unsealed", "wrong_manifest", "duplicate_batch", "duplicate_rank", "float_position",
                                  "bool_position", "duplicate_q", "duplicate_k", "padded", "split", "review", "no_held",
                                  "same_sequence", "negative_token", "oversize"])
def test_reject_bad_identity(corpus, fault):
    e, b = copy.deepcopy(corpus)
    if fault == "unsealed":
        b["boundary_map_sha256"] = "0" * 64
    elif fault == "wrong_manifest":
        b["evidence_manifest_file_sha256"] = "b" * 64
    elif fault == "duplicate_batch":
        e["rows"][1]["batch_index"] = 20
    elif fault == "duplicate_rank":
        b["rows"][1]["row_index"] = 0
    elif fault in ("float_position", "bool_position"):
        b["rows"][0]["pairs"][0][0] = 0.0 if fault == "float_position" else False
    elif fault == "duplicate_q":
        b["rows"][0]["pairs"][1][0] = 0
    elif fault == "duplicate_k":
        b["rows"][0]["pairs"][1][1] = 1
    elif fault == "padded":
        b["rows"][0]["pairs"][1][0] = 4
    elif fault == "split":
        b["rows"][0]["split"] = "held"
    elif fault == "review":
        b["rows"][0]["source"] = "bank"
    elif fault == "no_held":
        e["rows"][0]["split"] = b["rows"][1]["split"] = "fit"
    elif fault == "same_sequence":
        e["rows"][0]["ids"] = e["rows"][1]["ids"][:]
    elif fault == "negative_token":
        e["rows"][0]["ids"][0] = -1
    elif fault == "oversize":
        e["rows"][0]["ids"] *= 4
    if fault != "unsealed":
        seal(b)
    with pytest.raises(tool.Refusal):
        tool.evidence_rows(e, b, "a" * 64)


@pytest.fixture(scope="module")
def chimera():
    location = os.environ.get("CHIMERA_OMNI_SRC")
    if not location:
        pytest.fail("set CHIMERA_OMNI_SRC to the exact validated Chimera tree (synthetic test only)")
    native.bind_chimera_tree(location, None)
    from chimera_omni.config import ChimeraConfig
    from chimera_omni.modeling import ChimeraOmniModel
    from chimera_omni.importing.safetensors_io import write_safetensors
    cfg = ChimeraConfig(vocab_size=1024, hidden_size=32, intermediate_size=48, num_layers=3,
                        num_heads=2, num_kv_heads=1, max_sequence_length=16,
                        block_pattern=("kda", "gated_gqa", "local"), residual_mode="none", native_feature_dim=8)
    torch.manual_seed(171)
    model = ChimeraOmniModel(cfg).eval()
    return cfg, model, write_safetensors


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_hooks_native_block_bit_identical(chimera, layer):
    cfg, model, _ = chimera
    block = model.blocks[layer]
    x = torch.randn(1, 4, cfg.hidden_size)
    valid = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        plain = block(x, token_valid_mask=valid)
        hooks = tool.BoundaryHooks(block, torch.tensor([0, 3]))
        hooked = block(x, token_valid_mask=valid)
        hooks.close()
    assert torch.equal(plain, hooked)
    assert set(hooks.rows) == set(tool.BOUNDARIES)
    assert torch.equal(hooks.rows["residual_in"], x[0, [0, 3]])
    assert torch.equal(hooks.rows["residual_out"], plain[0, [0, 3]])
    assert not block._forward_hooks and not block.mixer._forward_hooks and not block.ffn._forward_hooks


def test_hook_failure_removes_handles(chimera):
    cfg, model, _ = chimera
    block = model.blocks[0]
    hooks = tool.BoundaryHooks(block, torch.tensor([0]))
    try:
        with pytest.raises(tool.Refusal, match="nonfinite"):
            block(torch.full((1, 2, cfg.hidden_size), float("nan")))
    finally:
        hooks.close()
    assert not block._forward_hooks and not block.norm2._forward_pre_hooks


@pytest.fixture
def checkpoint(tmp_path, chimera):
    cfg, model, writer = chimera
    path = tmp_path / "model"
    writer(model.state_dict(), path, config=cfg.to_manifest(), ledger=cfg.ledger().to_manifest(),
           source={"import_source": "explicitly-synthetic-test"}, chimera_commit="synthetic")
    return path


def test_capture_native_loader_matches_each_resident_boundary(checkpoint, chimera, corpus, tmp_path):
    cfg, model, _ = chimera
    rows = tool.evidence_rows(*corpus, "a" * 64)
    engine = native.StreamedChimera(checkpoint, device="cpu", dtype=torch.float32,
                                   ledger_band_logit_mask=False, verify_tensors=True)
    # Any accidental full-logits pass fails rather than merely slowing the test.
    engine.model.lm_head.forward = lambda *_a, **_k: pytest.fail("capture must not compute logits")
    out = tmp_path / "capture"
    out.mkdir()
    reports = tool.capture(engine, rows, [0, 1], out)
    assert engine.block_loads == 2
    assert [r["kind"] for r in reports] == ["kda", "gated_gqa"]
    expected = {i: {k: [] for k in tool.BOUNDARIES} for i in (0, 1)}
    with torch.no_grad():
        for row in rows:
            x = model.embed(torch.tensor([row["ids"]]))
            for i in (0, 1):
                h = tool.BoundaryHooks(model.blocks[i], torch.tensor([q for q, _ in row["pairs"]]))
                x = model.blocks[i](x, token_valid_mask=torch.ones(x.shape[:2], dtype=torch.bool))
                h.close()
                for key in tool.BOUNDARIES:
                    expected[i][key].append(h.rows[key])
    for i in (0, 1):
        payload = load_file(str(out / f"L{i:02d}" / "host_boundaries.safetensors"))
        for key in tool.BOUNDARIES:
            assert torch.equal(payload[key], torch.cat(expected[i][key])), (i, key)
        assert payload["held"].tolist() == [0, 0, 1, 1]
        assert payload["evidence_row_index"].tolist() == [0, 0, 1, 1]
        assert payload["batch_index"].tolist() == [7, 7, 20, 20]
        assert payload["host_position"].tolist() == [0, 3, 0, 2]
        assert payload["k3_position"].tolist() == [1, 4, 1, 3]
        assert payload["host_flat_position"].tolist() == [0, 3, 8, 10]
        assert payload["k3_flat_position"].tolist() == [1, 4, 9, 11]
        assert tool.digest(out / reports[i]["path"]) == reports[i]["sha256"]


@pytest.mark.parametrize("layers", [[], [1, 0], [1, 1], [-1], [3]])
def test_invalid_layer_window_refused(chimera, corpus, layers, tmp_path):
    engine = SimpleNamespace(cfg=chimera[0])
    with pytest.raises(tool.Refusal, match="layers"):
        tool.capture(engine, tool.evidence_rows(*corpus, "a" * 64), layers, tmp_path)


@pytest.mark.parametrize("fault", ["missing_digest", "escape", "bad_config_pin", "bad_index_pin"])
def test_checkpoint_seals_refuse(checkpoint, fault):
    config, index = checkpoint / "config.json", checkpoint / "model.safetensors.index.json"
    body = json.loads(index.read_text())
    if fault == "missing_digest":
        body["metadata"]["tensor_sha256"].pop(next(iter(body["weight_map"])))
    if fault == "escape":
        body["weight_map"][next(iter(body["weight_map"]))] = "../escape.safetensors"
    index.write_text(json.dumps(body))
    with pytest.raises(tool.Refusal):
        tool.checkpoint_inputs(checkpoint.resolve(), "0" * 64 if fault == "bad_config_pin" else tool.digest(config),
                               "0" * 64 if fault == "bad_index_pin" else tool.digest(index))


def test_native_load_refuses_changed_tensor_despite_unchanged_index(checkpoint):
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    name = "embed.weight"
    path = checkpoint / index["weight_map"][name]
    payload = load_file(str(path))
    payload[name][0, 0] += 1
    save_file(payload, str(path))
    # The map and its expected hashes still match their pins, but actual tensor
    # data must be verified by the native consumed-tensor loader.
    tool.checkpoint_inputs(checkpoint.resolve(), tool.digest(checkpoint / "config.json"),
                           tool.digest(checkpoint / "model.safetensors.index.json"))
    with pytest.raises(ValueError, match="does not match the sha256"):
        native.StreamedChimera(checkpoint, device="cpu", dtype=torch.float32,
                              ledger_band_logit_mask=False, verify_tensors=True)


@pytest.mark.parametrize("change_evidence", [False, True])
def test_cli_writes_final_receipt_no_capability_claim(checkpoint, corpus, tmp_path, monkeypatch, change_evidence):
    e, b = corpus
    ep, bp, rp = tmp_path / "evidence.json", tmp_path / "boundary.json", tmp_path / "host.json"
    ep.write_text(json.dumps(e))
    b["evidence_manifest_file_sha256"] = tool.digest(ep)
    seal(b)
    bp.write_text(json.dumps(b))
    rp.write_text(json.dumps({"schema": "synthetic-upstream-only"}))
    tree = Path(os.environ["CHIMERA_OMNI_SRC"])
    # Source identity check is covered by the native loader tests; avoid asserting
    # a synthetic commit is the real Chimera commit here.
    original = native.bind_chimera_tree
    monkeypatch.setattr(native, "bind_chimera_tree", lambda tree, commit: original(tree, None))
    args = tool.parser().parse_args(["--model", str(checkpoint), "--config-sha256", tool.digest(checkpoint / "config.json"),
        "--index-sha256", tool.digest(checkpoint / "model.safetensors.index.json"), "--host-receipt", str(rp),
        "--host-receipt-sha256", tool.digest(rp), "--evidence", str(ep), "--evidence-sha256", tool.digest(ep),
        "--boundary-map", str(bp), "--boundary-sha256", tool.digest(bp), "--chimera-tree", str(tree),
        "--chimera-commit", "synthetic", "--layers", "0,1", "--dtype", "float32", "--out", str(tmp_path / "out")])
    args.max_output_bytes = 1
    with pytest.raises(tool.Refusal, match="projected capture tensors"):
        tool.run(args)
    assert not args.out.exists()
    args.max_output_bytes = 16 * 1024**3
    if change_evidence:
        original_capture = tool.capture
        def capture_then_change(*a, **k):
            captured = original_capture(*a, **k)
            ep.write_text(ep.read_text() + "\n")
            return captured
        monkeypatch.setattr(tool, "capture", capture_then_change)
        with pytest.raises(tool.Refusal, match="SHA256 mismatch"):
            tool.run(args)
        assert (args.out / "L00" / "host_boundaries.safetensors").exists()
        assert not (args.out / "capture_receipt.json").exists()
        return
    result = tool.run(args)
    assert result["host_only"] and not result["source_host_pairs_complete"] and not result["folded_child_proven"]
    assert result["consumed_tensor_hashes_verified"]
    assert (args.out / "capture_receipt.json").is_file()
    with pytest.raises(tool.Refusal, match="output must be new"):
        tool.run(args)
