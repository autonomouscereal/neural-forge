"""Tiny explicitly synthetic CPU mechanics; never all-source capability evidence."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
import c17_capture_k3 as tool
import test_k3_dense_pass as fixture_tools


def write_json(path, body):
    path.write_text(json.dumps(body), encoding="utf-8")
    return {"path": str(path), "sha256": tool.host.digest(path)}


def ref(path):
    return {"path": str(path), "sha256": tool.host.digest(path)}


@pytest.fixture(scope="module")
def accepted(tmp_path_factory):
    root = tmp_path_factory.mktemp("c17-native-k3")
    model = root / "synthetic-model"
    # A four-layer tiny model, with L2 MLA, leaves a real end-of-window L3
    # snapshot. The production pass does not snapshot beyond the final layer.
    original_config = fixture_tools.k3_synth.build_config
    def fixture_config(args):
        cfg = original_config(args)
        cfg["text_config"]["linear_attn_config"]["kda_layers"] = [1, 2]
        cfg["text_config"]["linear_attn_config"]["full_attn_layers"] = [3, 4]
        return cfg
    fixture_tools.k3_synth.build_config = fixture_config
    try:
        fixture_tools.build_synth(model, top_k=8, layers=4)
    finally:
        fixture_tools.k3_synth.build_config = original_config
    index = tool.native.K3Index(str(model))
    for layer in (1, 2):
        prefix = index.layer_prefix_for(layer)
        path = model / index.weight_map[prefix + "block_sparse_moe.gate.weight"]
        tensors = load_file(str(path))
        for name in ("gate.weight", "gate.e_score_correction_bias"):
            key = prefix + "block_sparse_moe." + name
            tensors[key] = torch.zeros_like(tensors[key])
        save_file(tensors, str(path))
    ids = fixture_tools.write_ids(root / "k3_input_ids.safetensors", rows=4, seq=16,
            vocab=index.text_config["vocab_size"], seed=11, split=[0, 0, 0, 1], real_lengths=[16, 14, 16, 12], role="evidence")
    review = fixture_tools.write_ids(root / "k3_review.safetensors", rows=2, seq=8,
            vocab=index.text_config["vocab_size"], seed=12, split=[2, 2], real_lengths=[8, 5], role="review")
    argv = fixture_tools.base_argv(str(model), {"evidence": ids, "review": review}, root / "fold",
            root / "receipts", rung=512, body_dir=root / "body", acceptance="gate", snapshot_keep=8)
    receipt = fixture_tools.run_pass(argv)
    evidence = {"schema": "neural-forge-omni-evidence-v1", "rows": [
        {"batch_index": 10 + i * 7, "ids": [i + 2, 4, 5], "split": "held" if i == 3 else "fit"} for i in range(4)]}
    er = write_json(root / "evidence.json", evidence)
    boundary = {"schema": "neural-forge-k3-boundary-map-v2", "sequence_length": 16,
        "evidence_manifest_file_sha256": er["sha256"], "ids_files_sha256": {"k3_input_ids": tool.host.digest(ids)},
        "rows": [{"row_index": i, "file": "k3_input_ids", "source": "evidence",
                  "split": "held" if i == 3 else "fit", "qwen_tokens_real": 3,
                  "k3_tokens_real": length, "pairs": [[0, 1], [2, length - 1]]}
                 for i, length in enumerate([16, 14, 16, 12])]}
    boundary["boundary_map_sha256"] = tool.host.canonical_sha(boundary)
    br = write_json(root / "boundary.json", boundary)
    rr = ref(root / "receipts" / "k3_dense_pass_receipt.json")
    plan = {"schema": tool.PLAN_SCHEMA, "fixture": True, "layer": 2, "accepted_receipt": rr,
        "model_metadata_directory": str(model), "input_ids": ref(ids), "review_input_ids": [ref(review)],
        "evidence": er, "boundary_map": br, "max_capture_bytes": 1024 * 1024,
        "snapshot_manifest": ref(root / "fold" / "snapshots" / "snapshots_manifest.json"),
        "accepted_layers": {}, "body": ref(root / "body" / "L02" / "body.safetensors")}
    for layer in (1, 2):
        directory = root / "fold" / "dense" / f"L{layer:02d}"
        plan["accepted_layers"][str(layer)] = {"ffn": ref(directory / "ffn.safetensors"), "done": ref(directory / "done.json")}
    manifest = json.loads(Path(plan["snapshot_manifest"]["path"]).read_text())
    for name, layer, stream in (("start_folded", 2, "folded"), ("end_teacher", 3, "teacher"), ("end_folded", 3, "folded")):
        key = next(k for k, v in manifest["files"].items() if v["layer"] == layer and v["stream"] == stream)
        plan[name] = {"path": key, "manifest_key": key}
    return root, plan, receipt


def invoke(accepted, plan=None, suffix="capture", **kwargs):
    root, original, _ = accepted
    plan = copy.deepcopy(original) if plan is None else plan
    pin = write_json(root / f"{suffix}-plan.json", plan)
    out = root / suffix
    result = tool.run(Path(pin["path"]), pin["sha256"], out, synthetic_fixture_root=root, **kwargs)
    return result, out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.bool])
def test_chunked_native_hash_exact(dtype):
    value = torch.arange(120).reshape(10, 3, 4).to(dtype)
    assert tool.tensor_sha(value, rows=3) == tool.native.sha256_tensor(value)
    assert tool.tensor_sha(value[:, :, ::2], rows=3) == tool.native.sha256_tensor(value[:, :, ::2])


def test_actual_native_accepted_replay_no_parent_reads(accepted, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("capture must never read parent weights, route experts, or fit")
    for name in ("read_layer", "routed_y_lat"):
        monkeypatch.setattr(tool.native, name, forbidden)
    monkeypatch.setattr(tool.native.K3Index, "read_keys", forbidden)
    monkeypatch.setattr(tool.native.c14, "run_rung", forbidden)
    result, out = invoke(accepted)
    assert result["fixture"] and result["accepted_prefix_replay_exact"]
    assert result["teacher_reference_verified"] and result["teacher_forward_replayed"] is False
    assert result["parent_experts_loaded"] is False and result["new_fit"] is False
    assert result["full_corpus_replayed"] and result["corpus"]["tokens"] == 80
    assert result["source_host_pairs_complete"] is False and result["folded_child_proven"] is False
    assert result["layers"][0]["kind"] == "mla"
    payload = load_file(str(out / result["layers"][0]["path"]))
    assert set(tool.BOUNDARIES) <= set(payload)
    assert payload["ffn_input"].shape == (8, 64)
    assert payload["k3_position"].tolist() == [1, 15, 1, 13, 1, 15, 1, 11]
    assert payload["held"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1]
    assert not torch.equal(payload["post_attention"], payload["ffn_read"])
    assert not torch.equal(payload["ffn_read"], payload["ffn_input"])
    # Actual write, not a residual subtraction reconstructed after rounding.
    root, plan, receipt = accepted
    pins = tool.Pins()
    index = tool.native.K3Index(plan["model_metadata_directory"])
    runner, dense, _ = tool.load_layer(plan, receipt, index, pins, 2, torch.device("cpu"))
    x = payload["ffn_input"]
    latent = tool.native._chunked_linear(x, runner.w.body[tool.SHARED[0]], 7)
    y = tool.native.dense_y_lat(dense, latent, runner.cfg, 7)
    write = tool.native.moe_write(runner.w, x, y, runner.cfg, 7)
    # GEMM row shapes differ from the full replay; this is an independent mathematical check.
    torch.testing.assert_close(payload["ffn_output"], write, rtol=1e-5, atol=1e-6)


def test_native_kda_non_snapshot_capture(accepted):
    root, original, _ = accepted
    plan = copy.deepcopy(original)
    plan["layer"] = 1
    plan["accepted_layers"].pop("2")
    plan["body"] = ref(root / "body" / "L01" / "body.safetensors")
    manifest = json.loads(Path(plan["snapshot_manifest"]["path"]).read_text())
    for name, layer, stream in (("start_folded", 1, "folded"), ("end_teacher", 2, "teacher"), ("end_folded", 2, "folded")):
        key = next(k for k, v in manifest["files"].items() if v["layer"] == layer and v["stream"] == stream)
        plan[name] = {"path": key, "manifest_key": key}
    result, _ = invoke(accepted, plan, suffix="kda")
    assert result["layers"][0]["kind"] == "kda" and result["accepted_prefix_replay_exact"]


@pytest.mark.parametrize("fault", ["software", "runtime", "missing_ancestor", "ffn_sha", "body_sha", "manifest_key",
    "snapshot_payload", "snapshot_meta", "end_prefix", "teacher_prefix", "source_identity", "done_lineage",
    "not_accepted", "boundary_ids", "split", "review_missing", "budget", "fixture_flag"])
def test_fail_closed(accepted, fault):
    root, original, rec = accepted
    plan, receipt = copy.deepcopy(original), copy.deepcopy(rec)
    if fault == "software":
        receipt["resume_context"]["software"]["tool_sha256"] = "0" * 64
    elif fault == "runtime":
        receipt["resume_context"]["runtime"]["backend_flags"]["allow_tf32"] = not torch.backends.cuda.matmul.allow_tf32
    elif fault == "missing_ancestor":
        plan["accepted_layers"].pop("1")
    elif fault == "ffn_sha":
        plan["accepted_layers"]["2"]["ffn"]["sha256"] = "0" * 64
    elif fault == "body_sha":
        plan["body"]["sha256"] = "0" * 64
    elif fault == "manifest_key":
        plan["start_folded"]["manifest_key"] = "/no-such-pruned-snapshot"
    elif fault in ("snapshot_payload", "snapshot_meta"):
        source = Path(plan["start_folded"]["path"])
        payload = load_file(str(source))
        with tool.safe_open(str(source), framework="pt") as sf:
            metadata = sf.metadata()
        if fault == "snapshot_payload":
            payload["prefix_sum"][0, 0] += 1
        else:
            metadata["resume_context"] = "{}"
        target = root / f"{fault}.safetensors"
        save_file(payload, str(target), metadata=metadata)
        plan["start_folded"]["path"] = str(target)
        if fault == "snapshot_meta":
            manifest = json.loads(Path(plan["snapshot_manifest"]["path"]).read_text())
            manifest["files"][plan["start_folded"]["manifest_key"]]["sha256"] = tool.host.digest(target)
            plan["snapshot_manifest"] = write_json(root / f"{fault}-manifest.json", manifest)
    elif fault in ("end_prefix", "teacher_prefix"):
        receipt["layers"]["2"]["prefix_out_sha256"]["folded" if fault == "end_prefix" else "teacher"] = "0" * 64
    elif fault == "source_identity":
        receipt["resume_context"]["model"]["config_sha256"] = "0" * 64
    elif fault in ("done_lineage", "not_accepted"):
        marker = json.loads(Path(plan["accepted_layers"]["2"]["done"]["path"]).read_text())
        if fault == "done_lineage":
            marker["prefix_lineage"]["upstream_ffn_sha256"]["1"] = "0" * 64
        else:
            marker["fold_metrics"]["acceptance"]["verdict"] = "recorded"
        plan["accepted_layers"]["2"]["done"] = write_json(root / f"{fault}-done.json", marker)
    elif fault in ("boundary_ids", "split"):
        boundary = json.loads(Path(plan["boundary_map"]["path"]).read_text())
        if fault == "boundary_ids":
            boundary["ids_files_sha256"]["k3_input_ids"] = "0" * 64
        else:
            boundary["rows"][0]["split"] = "held"
        boundary["boundary_map_sha256"] = tool.host.canonical_sha({k: v for k, v in boundary.items() if k != "boundary_map_sha256"})
        plan["boundary_map"] = write_json(root / f"{fault}-boundary.json", boundary)
    elif fault == "review_missing":
        plan["review_input_ids"] = []
    elif fault == "budget":
        plan["max_capture_bytes"] = 1
    elif fault == "fixture_flag":
        plan["fixture"] = False
    plan["accepted_receipt"] = write_json(root / f"{fault}-receipt.json", receipt)
    with pytest.raises(tool.Refusal):
        invoke(accepted, plan, suffix=f"bad-{fault}")
    assert not (root / f"bad-{fault}" / "capture_receipt.json").exists()


def test_fixture_cannot_be_production(accepted):
    root, plan, _ = accepted
    pin = write_json(root / "production-plan.json", plan)
    with pytest.raises(tool.Refusal, match="synthetic"):
        tool.run(Path(pin["path"]), pin["sha256"], root / "production-capture")


def test_observer_restores_native_functions_on_failure(accepted):
    _, plan, receipt = accepted
    pins = tool.Pins()
    index = tool.native.K3Index(plan["model_metadata_directory"])
    runner, _, _ = tool.load_layer(plan, receipt, index, pins, 2, torch.device("cpu"))
    corpus = tool.native.Corpus(plan["input_ids"]["path"], [x["path"] for x in plan["review_input_ids"]])
    originals = runner.attention, tool.native.apply_attn_res, tool.native.moe_write
    with pytest.raises(RuntimeError, match="injected"):
        with tool.Observe(runner, corpus, torch.tensor([0])):
            raise RuntimeError("injected")
    assert (runner.attention, tool.native.apply_attn_res, tool.native.moe_write) == originals


def test_changed_consumed_file_has_no_final_receipt(accepted, monkeypatch):
    root, plan, _ = accepted
    original = tool.replay
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path = root / "change-during-plan.json"
        path.write_text(path.read_text() + " ", encoding="utf-8")
        return result
    monkeypatch.setattr(tool, "replay", changed)
    with pytest.raises(tool.Refusal, match="changed during"):
        invoke(accepted, plan, suffix="change-during")
    assert not (root / "change-during" / "capture_receipt.json").exists()


@pytest.mark.parametrize("part", ["prefix", "blocks"])
def test_actual_replay_mismatch_never_emits_receipt(accepted, monkeypatch, part):
    root, plan, _ = accepted
    original = tool.replay
    def changed(*args, **kwargs):
        result, payload = original(*args, **kwargs)
        getattr(result, part).flatten()[0] += 1
        return result, payload
    monkeypatch.setattr(tool, "replay", changed)
    with pytest.raises(tool.Refusal, match="replay is NOT exact"):
        invoke(accepted, plan, suffix=f"replay-bad-{part}")
    assert not (root / f"replay-bad-{part}" / "capture_receipt.json").exists()


@pytest.mark.parametrize("layer", [1, 2])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_observation_is_bit_identical_to_unwrapped_native(accepted, layer, dtype):
    root, original, original_receipt = accepted
    plan, receipt = copy.deepcopy(original), copy.deepcopy(original_receipt)
    receipt["resume_context"]["runtime"]["compute_dtype"] = dtype
    plan["body"] = ref(root / "body" / f"L{layer:02d}" / "body.safetensors")
    pins = tool.Pins()
    index = tool.native.K3Index(plan["model_metadata_directory"])
    runner, dense, _ = tool.load_layer(plan, receipt, index, pins, layer, torch.device("cpu"))
    corpus = tool.native.Corpus(plan["input_ids"]["path"], [p["path"] for p in plan["review_input_ids"]])
    saved = load_file(str(root / "fold" / "snapshots" / f"before_L{layer:02d}_folded.safetensors"))
    pair = tool.native.Pair(saved["prefix_sum"].to(tool.native.torch_dtype(dtype)),
                            saved["block_residual"].to(tool.native.torch_dtype(dtype)))
    runtime = receipt["resume_context"]["runtime"]
    with torch.no_grad():
        carried, x, blocks = tool.native.attention_half(runner, pair, corpus, torch.device("cpu"),
                row_chunk=runtime["row_chunk"], token_chunk=runtime["token_chunk"])
        latent = tool.native._chunked_linear(x, runner.w.body[tool.SHARED[0]], runtime["token_chunk"])
        expected, taps = tool.native.ffn_half(runner, carried, x, runner.cfg, device=torch.device("cpu"),
                token_chunk=runtime["token_chunk"], dense=dense, latent=latent)
        write = tool.native.moe_write(runner.w, x, taps["y_lat"], runner.cfg, runtime["token_chunk"])
    positions = torch.tensor([1, 15, 33, 59])
    actual, capture = tool.replay(runner, dense, pair, corpus, {"k3_flat_position": positions}, runtime)
    assert torch.equal(actual.prefix, expected) and torch.equal(actual.blocks, blocks)
    assert torch.equal(capture["ffn_input"], x[positions])
    assert torch.equal(capture["ffn_output"], write[positions])


def test_existing_output_is_never_overwritten(accepted):
    root, plan, _ = accepted
    out = root / "existing"
    out.mkdir()
    note = out / "preserve.txt"
    note.write_text("user data", encoding="utf-8")
    pin = write_json(root / "existing-plan.json", plan)
    with pytest.raises(tool.Refusal, match="fresh"):
        tool.run(Path(pin["path"]), pin["sha256"], out, synthetic_fixture_root=root)
    assert note.read_text() == "user data"


def window_plan(accepted, name):
    root, original, _ = accepted
    plan = copy.deepcopy(original)
    plan.pop("layer")
    plan.pop("body")
    plan["layers"] = [1, 2]
    plan["bodies"] = {str(layer): ref(root / "body" / f"L{layer:02d}" / "body.safetensors") for layer in (1, 2)}
    manifest = json.loads(Path(plan["snapshot_manifest"]["path"]).read_text())
    # Mimic real snapshot_every=12 + window ends: no intermediate L2 entry.
    manifest["files"] = {key: value for key, value in manifest["files"].items() if value["layer"] in (1, 3)}
    plan["snapshot_manifest"] = write_json(root / f"{name}-manifest.json", manifest)
    start = next(k for k, value in manifest["files"].items() if value["layer"] == 1 and value["stream"] == "folded")
    plan["start_folded"] = {"path": start, "manifest_key": start}
    return plan


def test_contiguous_window_needs_no_intermediate_snapshot_or_teacher(accepted, monkeypatch):
    root, _, receipt = accepted
    plan = window_plan(accepted, "window")
    # This unverified intermediate teacher value is deliberately never claimed.
    receipt = copy.deepcopy(receipt)
    receipt["layers"]["1"]["prefix_out_sha256"]["teacher"] = "0" * 64
    plan["accepted_receipt"] = write_json(root / "window-accepted.json", receipt)
    original_file = tool.Pins.file
    def no_intermediate(self, path, expected):
        assert "before_L02" not in str(path), "no intermediate snapshot may be consumed"
        return original_file(self, path, expected)
    monkeypatch.setattr(tool.Pins, "file", no_intermediate)
    def no_experts(*args, **kwargs):
        pytest.fail("a window cannot rerun parent experts")
    monkeypatch.setattr(tool.native, "routed_y_lat", no_experts)
    result, out = invoke(accepted, plan, suffix="window-capture")
    assert [x["layer"] for x in result["layers"]] == [1, 2]
    assert [x["kind"] for x in result["layers"]] == ["kda", "mla"]
    assert [x["teacher_reference_verified"] for x in result["layers"]] == [False, True]
    assert result["teacher_reference_scope"] == "final_window_boundary_only"
    assert result["teacher_reference_layer"] == 2
    assert result["intermediate_teacher_reference_verified"] is False
    assert result["final_window_verification"]["layers"] == [1, 2]
    assert result["final_window_verification"]["all_folded_prefixes_exact"]
    assert result["final_window_verification"]["snapshots_created"] is False
    for entry in result["layers"]:
        assert entry["folded_prefix_sha256"] == receipt["layers"][str(entry["layer"])]["prefix_out_sha256"]["folded"]
        payload = load_file(str(out / entry["path"]))
        assert set(tool.BOUNDARIES) <= set(payload)
        assert payload["ffn_input"].shape == (8, 64)
    assert set(result["loading"]) == {"1", "2"}


def test_window_checks_each_intermediate_folded_prefix(accepted):
    root, _, receipt = accepted
    plan = window_plan(accepted, "intermediate")
    receipt = copy.deepcopy(receipt)
    receipt["layers"]["1"]["prefix_out_sha256"]["folded"] = "0" * 64
    plan["accepted_receipt"] = write_json(root / "intermediate-accepted.json", receipt)
    with pytest.raises(tool.Refusal, match="L01: accepted folded prefix"):
        invoke(accepted, plan, suffix="intermediate-bad")
    assert not (root / "intermediate-bad").exists()


def test_failed_final_window_keeps_partial_rows_without_acceptance(accepted, monkeypatch):
    root, _, _ = accepted
    plan = window_plan(accepted, "final-window")
    original = tool.replay
    def wrong_final(runner, *args, **kwargs):
        pair, payload = original(runner, *args, **kwargs)
        if runner.w.layer_idx == 2:
            pair.blocks.flatten()[0] += 1
        return pair, payload
    monkeypatch.setattr(tool, "replay", wrong_final)
    with pytest.raises(tool.Refusal, match="BlockAttnRes replay is NOT exact"):
        invoke(accepted, plan, suffix="window-final-bad")
    partial = root / "window-final-bad" / "L01" / "k3_boundaries.safetensors"
    assert partial.is_file()
    with tool.safe_open(str(partial), framework="pt") as sf:
        assert "requires final capture_receipt.json" in sf.metadata()["acceptance"]
    assert not (root / "window-final-bad" / "capture_receipt.json").exists()


@pytest.mark.parametrize("fault", ["both", "gap", "duplicate", "reverse", "bool", "zero", "too_many", "missing_body", "extra_body", "singular_body"])
def test_window_selection_is_explicit_and_bounded(accepted, fault):
    _, plan, _ = accepted
    index = tool.native.K3Index(plan["model_metadata_directory"])
    candidate = {"layers": [1, 2], "bodies": {"1": {}, "2": {}}}
    if fault == "both":
        candidate["layer"] = 2
    elif fault == "gap":
        candidate["layers"] = [1, 3]
    elif fault == "duplicate":
        candidate["layers"] = [1, 1]
    elif fault == "reverse":
        candidate["layers"] = [2, 1]
    elif fault == "bool":
        candidate["layers"] = [True, 2]
    elif fault == "zero":
        candidate["layers"] = [0, 1]
    elif fault == "too_many":
        candidate["layers"] = list(range(1, tool.MAX_WINDOW_LAYERS + 2))
    elif fault == "missing_body":
        candidate["bodies"].pop("2")
    elif fault == "extra_body":
        candidate["bodies"]["3"] = {}
    elif fault == "singular_body":
        candidate["body"] = {}
    with pytest.raises(tool.Refusal):
        tool.capture_window(candidate, index)


def test_all_window_bodies_verified_before_any_replay(accepted, monkeypatch):
    plan = window_plan(accepted, "body-preflight")
    plan["bodies"]["2"]["sha256"] = "0" * 64
    def forbidden(*args, **kwargs):
        pytest.fail("must not replay first layer with later body unverified")
    monkeypatch.setattr(tool, "replay", forbidden)
    with pytest.raises(tool.Refusal, match="body differs"):
        invoke(accepted, plan, suffix="body-preflight-bad")
