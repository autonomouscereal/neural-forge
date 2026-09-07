"""Tiny explicitly synthetic fixtures; no test is real QIMG source evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c16_qimg_fold as tool


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def hash_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def dump(path, body):
    path.write_text(json.dumps(body), encoding="utf-8")


def bundle(tmp_path):
    torch.manual_seed(92)
    block, width, inner, head = 0, 8, 32, 2
    frozen = {k: (torch.randn(shape) * .15).to(torch.float8_e4m3fn)
              for k, shape in tool.frozen_shapes(block, width, inner, head).items()}
    frozen_record = tool.write_tensors(tmp_path / "frozen.safetensors", frozen)
    records, entries = [], []
    for index, split in enumerate(("fit", "holdout")):
        count_img, count_txt = 24, 12
        total = count_img + count_txt
        image = hash_text("fixture image " + str(index))
        prompt = hash_text("fixture prompt " + str(index))
        name = "fixture" + str(index)
        state = {"source_ids": torch.zeros(1, total, 3),
                 "rotation_generated": torch.eye(2).reshape(1, 1, 1, 1, 2, 2).repeat(1, 1, total, head // 2, 1, 1),
                 "temb": torch.zeros(1, width, dtype=torch.bfloat16),
                 "text_attention_mask": torch.zeros(1, count_txt, dtype=torch.bfloat16),
                 "prediction": torch.zeros(1, 16, 1, 8, 12, dtype=torch.bfloat16),
                 "model_input": torch.zeros(1, 16, 1, 8, 12, dtype=torch.bfloat16),
                 "sigma": torch.tensor([.3 + index * .5]),
                 "image_patch_indices": torch.zeros(count_img, 3, dtype=torch.int64),
                 "joint_row_kind": torch.tensor([0] * count_txt + [1] * count_img, dtype=torch.int8),
                 "context_token_ids": torch.arange(count_txt)[None],
                 "context_valid": torch.tensor([[True] * 10 + [False] * 2])}
        state["rotation_used"] = state["rotation_generated"].bfloat16()
        for stream, count in (("img", count_img), ("txt", count_txt)):
            x = torch.randn(1, count, width).bfloat16()
            _, y = tool.mlp_forward(x, tool.mlp_weights(frozen, block, stream, "cpu"))
            state[stream + "_mlp_in"], state[stream + "_mlp_out"] = x, y
            for suffix in ("_in", "_out", "_attn_out"):
                state[stream + suffix] = x.clone()
            state[stream + "_modulation"] = torch.zeros(1, 6 * width, dtype=torch.bfloat16)
        record = tool.write_tensors(tmp_path / (name + ".safetensors"), state)
        records.append({**record, "fixture": True, "complete_context": True, "row_order": "text_then_image",
                        "sample_id": name, "split": split, "image_sha256": image, "prompt_sha256": prompt,
                        "image_tokens": count_img, "text_tokens": count_txt, "sigma": float(state["sigma"][0])})
        entries.append({"id": name, "split": split, "image": {"sha256": image}, "prompt_sha256": prompt})
    receipt = {"schema": tool.CAPTURE_SCHEMA, "status": "complete", "fixture": True, "block": block,
               "runtime": {"manual_cast_dtype": "torch.bfloat16", "fp8_optimizations": False,
                           "selected_block_and_time_match_source_bytes": True},
               "source": {"width": width, "head_dim": head, "mlps": {k: {"parent_inner": inner} for k in ("image", "text")}},
               "inputs": {"entries": entries}, "captures": records, "frozen_source": frozen_record}
    receipt_path = tmp_path / "capture_receipt.json"
    dump(receipt_path, receipt)
    policy = {"schema": tool.POLICY_SCHEMA, "preregistered_before_scoring": True,
              "gate_scope": "pooled_and_each_heldout_forward",
              "streams": {s: {"keep": k, "max_source_replay_rel_l2": 0,
                              "max_heldout_rel_l2": 100, "min_heldout_cos": -1} for s, k in (("img", 3), ("txt", 2))}}
    policy_path = tmp_path / "fixture_policy.json"
    dump(policy_path, policy)
    return receipt_path, receipt, policy_path, policy


def test_fixture_conversion_folds_both_biased_mlps_and_preserves_every_other_source_byte(tmp_path):
    path, _, policy_path, policy = bundle(tmp_path)
    data = tool.load_captures([path], fixture=True)
    original = {k: v.clone() for k, v in data["frozen"].items()}
    out = tmp_path / "fold"
    result = tool.convert([path], policy_path, tool.file_sha(policy_path), out, fixture=True)
    assert result["status"] == "measured_candidate" and result["fixture"] is True
    assert result["numerical_policy_passed"] is True
    assert result["law"]["heldout_used_for_selection_or_refit"] is False
    assert result["row_preflight"]["txt"]["fit"] == 10
    stored = tool.load_file(str(out / result["output"]["file"]))
    selection = tool.load_file(str(out / result["selection"]["file"]))
    assert set(stored) == set(original)
    for name, before in original.items():
        if "_mlp." not in name:
            assert torch.equal(stored[name].view(torch.uint8), before.view(torch.uint8))
    for stream in tool.STREAMS:
        prefix = "transformer_blocks.0." + stream + "_mlp."
        chosen = selection[stream + ".selected_channels"]
        for suffix in ("net.0.proj.weight", "net.0.proj.bias"):
            assert stored[prefix + suffix].dtype == torch.float8_e4m3fn
            assert torch.equal(stored[prefix + suffix].view(torch.uint8), original[prefix + suffix].view(torch.uint8)[chosen])
        assert stored[prefix + "net.2.bias"].shape == (8,)
        assert stored[prefix + "net.2.weight"].dtype == stored[prefix + "net.2.bias"].dtype == torch.bfloat16
        assert 4 * result["streams"][stream]["child_parameters"] <= result["streams"][stream]["parent_parameters"]
        assert result["streams"][stream]["stored_update_scale"] <= 1
        assert all(r["scores"]["rel_l2"] == 0 for r in result["source_replay"][stream])
    assert "not installed" in result["capability_claim"]


def test_all_fit_canary_refuses_before_tensor_load(tmp_path, monkeypatch):
    path, receipt, *_ = bundle(tmp_path)
    receipt["captures"][1]["split"] = receipt["inputs"]["entries"][1]["split"] = "fit"
    dump(path, receipt)
    monkeypatch.setattr(tool, "load_file", lambda *a, **k: pytest.fail("all-fit must refuse before tensor payload loading"))
    with pytest.raises(ValueError, match="all-fit canary"):
        tool.load_captures([path], fixture=True)


def test_actual_image_identity_not_sample_name_guards_group_split(tmp_path):
    path, receipt, *_ = bundle(tmp_path)
    image = receipt["captures"][0]["image_sha256"]
    receipt["captures"][1]["image_sha256"] = receipt["inputs"]["entries"][1]["image"]["sha256"] = image
    dump(path, receipt)
    with pytest.raises(ValueError, match="leaks across"):
        tool.load_captures([path], fixture=True)


@pytest.mark.parametrize("problem", ["quarter_bias", "text_rows", "no_held_rows"])
def test_count_and_row_floors_precede_replay_and_solver(tmp_path, monkeypatch, problem):
    path, _, _, policy = bundle(tmp_path)
    data = tool.load_captures([path], fixture=True)
    if problem == "quarter_bias":
        policy["streams"]["img"]["keep"] = data["inner"] // 4
    elif problem == "text_rows":
        policy["streams"]["txt"]["keep"] = 7
    else:
        data["row_counts"]["txt"]["holdout"] = 0
    monkeypatch.setattr(tool, "mlp_forward", lambda *a: pytest.fail("invalid row/count preflight reached replay"))
    with pytest.raises(ValueError):
        tool.fold_loaded(data, policy)


def test_source_replay_failure_cannot_be_hidden_as_fold_error(tmp_path, monkeypatch):
    path, _, _, policy = bundle(tmp_path)
    data = tool.load_captures([path], fixture=True)
    data["forwards"][1]["txt"]["y"] = data["forwards"][1]["txt"]["y"] + 1
    monkeypatch.setattr(tool, "pivoted_columns_gram", lambda *a: pytest.fail("source baseline failure reached pivoting"))
    with pytest.raises(tool.FoldRejected) as exc:
        tool.fold_loaded(data, policy)
    assert exc.value.receipt["status"] == "rejected_source_replay"


def test_heldout_rows_never_influence_selection_or_refit(tmp_path):
    path, _, _, policy = bundle(tmp_path)
    data = tool.load_captures([path], fixture=True)
    first, selected, _ = tool.fold_loaded(data, policy)
    changed = copy.deepcopy(data)
    for stream in tool.STREAMS:
        item = changed["forwards"][1][stream]
        item["x"] = (item["x"].float() * 1.7 + .3).bfloat16()
        _, item["y"] = tool.mlp_forward(item["x"], tool.mlp_weights(changed["frozen"], 0, stream, "cpu"))
    second, selected2, _ = tool.fold_loaded(changed, policy)
    assert all(torch.equal(first[k].view(torch.uint8), second[k].view(torch.uint8)) for k in first)
    assert all(torch.equal(selected[k], selected2[k]) for k in selected)


def test_affine_solver_receives_actual_targets_and_parent_bias_prior(tmp_path, monkeypatch):
    path, _, _, policy = bundle(tmp_path)
    data = tool.load_captures([path], fixture=True)
    original = tool.law_abiding_solve
    calls = []
    def inspect_solve(design, targets, prior):
        assert torch.equal(design[:, -1], torch.ones(len(design), dtype=torch.float64))
        calls.append((targets.clone(), prior[:, -1].clone()))
        return original(design, targets, prior)
    monkeypatch.setattr(tool, "law_abiding_solve", inspect_solve)
    tool.fold_loaded(data, policy)
    for stream, (target, bias) in zip(tool.STREAMS, calls):
        expected = data["forwards"][0][stream]
        assert torch.equal(target, expected["y"].squeeze(0)[expected["valid"]].double())
        assert torch.equal(bias, data["frozen"][f"transformer_blocks.0.{stream}_mlp.net.2.bias"].bfloat16().double())


def test_stored_update_veto_does_not_trust_solver_notes(tmp_path, monkeypatch):
    path, _, _, policy = bundle(tmp_path)
    data = tool.load_captures([path], fixture=True)
    monkeypatch.setattr(tool, "law_abiding_solve", lambda design, target, prior: (prior * 3, {"update_scale": 0}))
    with pytest.raises(tool.FoldRejected) as exc:
        tool.fold_loaded(data, policy)
    assert exc.value.receipt["status"] == "rejected_stored_update"
    assert exc.value.receipt["stored_update_scale"] > 1


def test_preregistered_numeric_rejection_writes_no_candidate(tmp_path):
    path, _, policy_path, policy = bundle(tmp_path)
    policy["streams"]["img"]["max_heldout_rel_l2"] = 0
    dump(policy_path, policy)
    out = tmp_path / "rejected"
    with pytest.raises(tool.FoldRejected) as exc:
        tool.convert([path], policy_path, tool.file_sha(policy_path), out, fixture=True)
    assert exc.value.receipt["status"] == "rejected_numerical_policy"
    assert exc.value.receipt["policy_sha256"] == tool.file_sha(policy_path)
    assert exc.value.receipt["capture_receipts"][0]["sha256"] == tool.file_sha(path)
    assert exc.value.receipt["candidate_written"] is False
    assert not out.exists()


def test_validate_only_checks_rows_and_identities_without_replay_fit_or_output(tmp_path, monkeypatch):
    path, _, policy_path, _ = bundle(tmp_path)
    monkeypatch.setattr(tool, "fold_loaded", lambda *a, **k: pytest.fail("validate-only launched folding"))
    out = tmp_path / "uncreated"
    result = tool.convert([path], policy_path, tool.file_sha(policy_path), out, validate_only=True, fixture=True)
    assert result["validated"] is True
    assert result["source_replay_performed"] is result["fold_launched"] is False
    assert result["row_preflight"]["txt"]["fit"] == 10
    assert result["capture_receipts"][0]["sha256"] == tool.file_sha(path)
    assert not out.exists()


@pytest.mark.parametrize("mutation", ["hash", "nan", "undeclared", "ignored_gate", "ignored_top_gate"])
def test_policy_requires_exact_caller_identity_and_explicit_finite_gates(tmp_path, mutation):
    _, _, path, policy = bundle(tmp_path)
    if mutation == "nan": policy["streams"]["txt"]["max_heldout_rel_l2"] = float("nan")
    elif mutation == "undeclared": policy["preregistered_before_scoring"] = False
    elif mutation == "ignored_gate": policy["streams"]["img"]["unimplemented_gate"] = 0
    elif mutation == "ignored_top_gate": policy["unimplemented_gate"] = 0
    dump(path, policy)
    with pytest.raises(ValueError):
        tool.load_policy(path, "0" * 64 if mutation == "hash" else tool.file_sha(path))


@pytest.mark.parametrize("mutation", ["fixture", "partial_hashes", "escape", "tamper", "wrong_compute", "duplicate"])
def test_capture_identity_and_source_refusals(tmp_path, mutation):
    path, receipt, *_ = bundle(tmp_path)
    if mutation == "partial_hashes": receipt["captures"][0]["tensor_sha256"].pop("img_mlp_in")
    elif mutation == "escape": receipt["captures"][0]["file"] = "../fixture0.safetensors"
    elif mutation == "tamper": receipt["captures"][0]["file_sha256"] = "0" * 64
    elif mutation == "wrong_compute": receipt["runtime"]["manual_cast_dtype"] = "torch.float32"
    elif mutation == "duplicate": receipt["captures"].append(copy.deepcopy(receipt["captures"][0]))
    dump(path, receipt)
    with pytest.raises(ValueError):
        tool.load_captures([path], fixture=mutation != "fixture")


def test_existing_output_and_original_sources_are_preserved(tmp_path):
    path, _, policy_path, _ = bundle(tmp_path)
    out = tmp_path / "historical"
    out.mkdir()
    old = out / "old.txt"
    old.write_text("preserve")
    source_sha = tool.file_sha(tmp_path / "frozen.safetensors")
    with pytest.raises(ValueError, match="new directory"):
        tool.convert([path], policy_path, tool.file_sha(policy_path), out, fixture=True)
    assert old.read_text() == "preserve" and tool.file_sha(tmp_path / "frozen.safetensors") == source_sha


def test_output_failure_preserves_partial_artifacts_with_incomplete_receipt(tmp_path, monkeypatch):
    path, _, policy_path, _ = bundle(tmp_path)
    original_writer = tool.write_tensors
    def failed_selection(target, tensors):
        if target.name == "selection.safetensors":
            raise OSError("fixture disk failure")
        return original_writer(target, tensors)
    monkeypatch.setattr(tool, "write_tensors", failed_selection)
    out = tmp_path / "incomplete"
    with pytest.raises(OSError, match="disk failure"):
        tool.convert([path], policy_path, tool.file_sha(policy_path), out, fixture=True)
    receipt = json.loads((out / "fold_receipt.json").read_text())
    assert receipt["status"] == "incomplete_output"
    assert (out / "folded_source_block_and_time.safetensors").is_file()
    with pytest.raises(ValueError, match="new directory"):
        tool.convert([path], policy_path, tool.file_sha(policy_path), out, fixture=True)
