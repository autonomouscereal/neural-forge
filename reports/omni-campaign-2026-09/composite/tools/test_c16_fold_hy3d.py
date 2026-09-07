"""Synthetic CPU mechanics only. No actual parent fold or 3D capability proof."""
from contextlib import contextmanager, ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import c16_fold_hy3d as mod
from test_c16_replay_hy3d import Block, Time


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def policy():
    body = json.loads((Path(__file__).parent / "hy3d_fold_policy_sep07.json").read_text())
    body.update(policy_id="explicit-synthetic-plumbing-fixture", keep=3)
    for name in mod.SCORED:
        body[name] = {"max_relative_l2": 100, "min_mean_cosine": -1}
    return body


@torch.inference_mode()
def contexts(block, time, embedding, context_width=4, held_shift=0):
    for module in (block, time, embedding):
        module.half().eval().requires_grad_(False)
    rows, objects = [], []
    for index, split in enumerate(("fit", "held", "fit", "held")):
        gen = torch.Generator().manual_seed(40 + index)
        latent = torch.randn(1, 3, 2, generator=gen).half() * .2
        sigma = torch.tensor([.25 + index * .125], dtype=torch.float16)
        token = time(sigma, condition=None)
        initial = torch.cat((token, embedding(latent)), 1)
        context = torch.randn(1, 2, context_width, generator=gen).half()
        if split == "held": context = context + held_shift
        state = {"dit_latent_input": latent, "block_input": initial, "context": context,
                 "sigma": sigma, "timestep": sigma.clone(), "time_token": token}
        with ExitStack() as hooks:
            for name, module in (("self_attention_write", block.attn1), ("cross_attention_write", block.attn2), ("mlp_output", block.mlp)):
                hooks.callback(module.register_forward_hook(lambda m, a, out, name=name: state.__setitem__(name, out.clone())).remove)
            hooks.callback(block.mlp.register_forward_pre_hook(lambda m, a: state.__setitem__("mlp_input", a[0].clone())).remove)
            state["block_output"] = block(initial, token, context)
        objects.append(state)
        rows.append({"split": split, "object_sha256": sha(f"fixture mesh {index}"),
                     "conditioning_sha256": sha(f"fixture image {index}"), "seed": index,
                     "requested_sigma": float(sigma[0]), "geometry_identity": {
                         "schema": mod.capture.GEOMETRY_SCHEMA, "sha256": sha(f"fixture geometry {index}")}})
    return {k: torch.cat([obj[k] for obj in objects]) for k in objects[0]}, rows


@pytest.fixture
def fixture():
    torch.manual_seed(143)
    block, time, embedding = Block(), Time(), torch.nn.Linear(2, 4)
    tensors, rows = contexts(block, time, embedding)
    return tensors, rows, block, time, embedding, policy()


def test_native_bias_inclusive_count_requires_2047_not_2048():
    assert mod.maximum_fourfold_keep(2048, 8192) == 2047
    parent = mod.biased_gelu_mlp_parameters(2048, 8192)
    assert parent == 33_564_672
    assert 4 * mod.biased_gelu_mlp_parameters(2048, 2047) <= parent
    assert 4 * mod.biased_gelu_mlp_parameters(2048, 2048) > parent


def test_literal_byte_equality_preserves_signed_zero_boundary():
    positive, negative = torch.tensor([0.]).half(), torch.tensor([-0.]).half()
    assert torch.equal(positive, negative) and not mod.bit_exact(positive, negative)


def test_external_autocast_refused_before_replay(fixture, monkeypatch):
    monkeypatch.setattr(mod.replay, "replay_one", lambda *a: pytest.fail("external autocast reached native replay"))
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        mod.fold_loaded(*fixture)


def test_complete_context_f16_fold_matches_existing_candidate_loader(fixture, tmp_path):
    tensors, rows, block, time, embedding, frozen = fixture
    original = mod.source_tensor_identity(block, time, embedding)
    state, selected, report = mod.fold_loaded(*fixture)
    assert report["baseline_all_boundaries_exact"] and report["numerical_policy_passed"]
    assert report["counts"]["fit_rows"] == report["counts"]["held_rows"] == 8
    assert report["counts"]["time_token_included"] and report["compression_x"] >= 4
    assert report["stored_update_scale"] <= 1 and not report["law"]["token_shuffle"]
    assert not report["law"]["held_used_for_selection_or_refit"]
    assert all(v.dtype == torch.float16 for v in state.values())
    assert torch.equal(state["fc1.weight"], block.mlp.fc1.weight[selected])
    assert torch.equal(state["fc1.bias"], block.mlp.fc1.bias[selected])
    assert mod.source_tensor_identity(block, time, embedding) == original
    source, body = {"source_identity": {"fixture": True}}, {"block_index": 0, "file_sha256": sha("fixture capture")}
    record = mod.capture.sealed_tensors(tmp_path / "fold.safetensors", state)
    receipt = {"schema": mod.replay.FOLD_SCHEMA, **source, **record, "block_index": 0,
               "capture_sha256": body["file_sha256"], "fit_object_sha256": sorted(r["object_sha256"] for r in rows if r["split"] == "fit")}
    path = tmp_path / "fold.json"
    mod.capture.sealed_json(path, receipt)
    first, second, info, _ = mod.replay.load_candidate(path, source, body, rows, 4, "cpu")
    assert first.out_features == second.in_features == 3
    assert info["parameters"] == report["counts"]["child_parameters"]


@pytest.mark.parametrize("name", ["context", "time_token", "block_input", "mlp_output", "self_attention_write"])
def test_any_source_mismatch_refuses_before_selection(fixture, monkeypatch, name):
    tensors, *rest = fixture
    tensors[name] = tensors[name] + .25
    monkeypatch.setattr(mod.linalg, "pivoted_columns_gram", lambda *a: pytest.fail("baseline mismatch reached fitting"))
    with pytest.raises(mod.FoldRejected) as exc:
        mod.fold_loaded(tensors, *rest)
    assert exc.value.receipt["status"] == "rejected_source_replay"


def test_affine_solver_gets_only_ordered_fit_targets_and_native_bias(fixture, monkeypatch):
    original = mod.linalg.law_abiding_solve
    tensors, rows, block, *_ = fixture
    called = []
    def checked(design, target, prior):
        assert not torch.is_grad_enabled() and design.dtype == target.dtype == prior.dtype == torch.float64
        assert torch.equal(design[:, -1], torch.ones(8, dtype=torch.float64))
        assert torch.equal(target, torch.cat([tensors["mlp_output"][i] for i in (0, 2)]).double())
        assert torch.equal(prior[:, -1], block.mlp.fc2.bias.double())
        called.append(True)
        return original(design, target, prior)
    monkeypatch.setattr(mod.linalg, "law_abiding_solve", checked)
    mod.fold_loaded(*fixture)
    assert called == [True]


def test_changed_held_context_cannot_change_selection_or_fitted_bytes(fixture):
    first, selected, _ = mod.fold_loaded(*fixture)
    _, _, block, time, embedding, frozen = fixture
    second_inputs, rows = contexts(block, time, embedding, held_shift=.5)
    second, selected2, _ = mod.fold_loaded(second_inputs, rows, block, time, embedding, frozen)
    assert torch.equal(selected, selected2)
    assert all(torch.equal(first[k], second[k]) for k in first)


@pytest.mark.parametrize("kind", ["large", "overflow", "nan"])
def test_stored_update_veto_recomputes_f16_update(fixture, monkeypatch, kind):
    def bad(design, target, prior):
        answer = prior * 3 if kind == "large" else torch.full_like(prior, 1e20 if kind == "overflow" else float("nan"))
        return answer, {"update_scale": 0}
    monkeypatch.setattr(mod.linalg, "law_abiding_solve", bad)
    with pytest.raises(ValueError):
        mod.fold_loaded(*fixture)


def test_all_law_rungs_vetoed_keeps_unchanged_prior_not_unbounded_solve(fixture, monkeypatch):
    original = mod.linalg.prior_anchored
    def veto(*args, **kwargs):
        answer, notes = original(*args, **kwargs)
        return answer, {**notes, "update_scale": 2.0}
    monkeypatch.setattr(mod.linalg, "prior_anchored", veto)
    _, _, report = mod.fold_loaded(*fixture)
    assert report["solve"]["fell_back_to_prior"] and report["stored_update_scale"] == 0


def test_qimg_tanh_gelu_cannot_silently_substitute_native_exact_gelu(fixture):
    fixture[2].mlp.gelu = torch.nn.GELU(approximate="tanh")
    with pytest.raises(ValueError, match="exact GELU"):
        mod.fold_loaded(*fixture)


@pytest.mark.parametrize("key", ["object_sha256", "conditioning_sha256", "geometry_identity"])
def test_different_filenames_do_not_permit_cross_object_leakage(fixture, key):
    fixture[1][1][key] = copy.deepcopy(fixture[1][0][key])
    with pytest.raises(ValueError, match="duplicate"):
        mod.fold_loaded(*fixture)


@pytest.mark.parametrize("mutation", ["third_fit", "removed_object", "dtype", "quarter_bias", "skip", "nonfinite"])
def test_preflight_refuses_before_native_replay(fixture, monkeypatch, mutation):
    tensors, rows, block, _, _, frozen = fixture
    if mutation == "third_fit": rows[1]["split"] = "fit"
    elif mutation == "removed_object": rows.pop()
    elif mutation == "dtype": tensors["context"] = tensors["context"].float()
    elif mutation == "quarter_bias": frozen["keep"] = 4
    elif mutation == "skip": tensors["skip_value"] = tensors["block_input"]
    else:
        tensors["context"] = tensors["context"].clone()
        tensors["context"][0, 0, 0] = float("nan")
    monkeypatch.setattr(mod.replay, "replay_one", lambda *a: pytest.fail("bad preflight reached original block"))
    with pytest.raises(ValueError): mod.fold_loaded(*fixture)


def test_each_held_object_gate_and_original_restoration(fixture):
    fixture[-1]["block_output"]["max_relative_l2"] = 0
    original = mod.source_tensor_identity(*fixture[2:5])
    with pytest.raises(mod.FoldRejected) as exc: mod.fold_loaded(*fixture)
    assert exc.value.receipt["status"] == "rejected_numerical_policy"
    assert len(exc.value.receipt["runtime_f16"]["objects"]) == 4
    assert mod.source_tensor_identity(*fixture[2:5]) == original


@pytest.mark.parametrize("mutation", ["hash", "nan", "extra_gate", "extra_top", "baseline", "undeclared", "moe"])
def test_policy_exact_sha_and_implemented_gates_only(tmp_path, mutation):
    body = policy()
    if mutation == "nan": body["mlp_output"]["max_relative_l2"] = float("nan")
    elif mutation == "extra_gate": body["mlp_output"]["ignored"] = 2
    elif mutation == "extra_top": body["ignored"] = 2
    elif mutation == "baseline": body["baseline"] = "close_enough"
    elif mutation == "undeclared": body["preregistered_before_scoring"] = False
    elif mutation == "moe": body["block_index"] = 15
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        mod.load_policy(path, "0" * 64 if mutation == "hash" else mod.capture.digest_file(path))


@pytest.fixture
def conversion(tmp_path, fixture, monkeypatch):
    """Explicitly mocks source/capture identity loaders, never production evidence."""
    tensors, rows, block, time, embedding, frozen = fixture
    parent = tmp_path / "fixture-parent.safetensors"
    state = {prefix + k: v for prefix, module in (("model.blocks.0.", block), ("model.t_embedder.", time), ("model.x_embedder.", embedding))
             for k, v in module.state_dict().items()}
    save_file(state, str(parent))
    source = tmp_path / "native-fixture"
    (source / "configs").mkdir(parents=True)
    (source / "configs/dit_config_2_1.yaml").write_text("fixture: true")
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(frozen))
    body = {"block_index": 0, "file_sha256": sha("fixture combined capture")}
    identity = {"source_identity": {"fixture": True}}
    monkeypatch.setattr(mod.replay, "inspect_source", lambda *a: identity)
    monkeypatch.setattr(mod.replay, "load_capture", lambda *a: (tensors, rows, {}, body))
    monkeypatch.setattr(mod.capture, "verify_native_dependencies", lambda *a: None)
    @contextmanager
    def native(*args, **kwargs): yield {"fixture": True}
    monkeypatch.setattr(mod.capture, "native_modules", native)
    monkeypatch.setattr(mod.replay, "native_components", lambda *a: (block, time, embedding))
    monkeypatch.setattr(mod.replay, "CAPTURE_DIMS", {"hidden": 4})
    return [tmp_path, parent, source, p, mod.capture.digest_file(p), tmp_path / "output"], body, rows, identity


def test_sealed_production_plumbing_with_explicit_identity_fixtures(conversion):
    args, body, rows, source = conversion
    report = mod.convert(*args)
    out = args[-1]
    assert report["schema"] == mod.replay.FOLD_SCHEMA and report["numerical_policy_passed"]
    assert report["source_identity"]["fixture"] and not report["folded_child_proven"]
    _, _, info, _ = mod.replay.load_candidate(out / "fold_receipt.json", source, body, rows, 4, "cpu")
    assert info["parameters"] == report["counts"]["child_parameters"]
    assert len(report["source_tensor_identity"]) > 4
    assert not (out / "incomplete.json").exists()
    with pytest.raises(ValueError, match="fresh output"): mod.convert(*args)


def test_rejection_is_preserved_without_any_candidate(conversion, monkeypatch):
    args, *_ = conversion
    monkeypatch.setattr(mod, "fold_loaded", lambda *a: (_ for _ in ()).throw(mod.FoldRejected("rejected_source_replay")))
    with pytest.raises(mod.FoldRejected): mod.convert(*args)
    assert list(p.name for p in args[-1].iterdir()) == ["rejection.json"]
    assert json.loads((args[-1] / "rejection.json").read_text())["policy_sha256"] == args[-2]


def test_output_failure_preserves_partial_files_without_acceptance(conversion, monkeypatch):
    args, *_ = conversion
    original = mod.capture.sealed_tensors
    def fail(path, tensors):
        if path.name == "folded_mlp.safetensors": raise OSError("fixture disk error")
        return original(path, tensors)
    monkeypatch.setattr(mod.capture, "sealed_tensors", fail)
    with pytest.raises(OSError, match="disk error"): mod.convert(*args)
    assert (args[-1] / "selection.safetensors").exists()
    assert (args[-1] / "incomplete.json").exists()
    assert not (args[-1] / "fold_receipt.json").exists()


def test_policy_mutation_during_fit_refuses_acceptance(conversion, monkeypatch):
    args, *_ = conversion
    original = mod.fold_loaded
    def mutation(*a):
        result = original(*a)
        args[3].write_text("{}")
        return result
    monkeypatch.setattr(mod, "fold_loaded", mutation)
    with pytest.raises(ValueError, match="input changed"): mod.convert(*args)
    assert (args[-1] / "incomplete.json").exists()
    assert not (args[-1] / "fold_receipt.json").exists()


@pytest.mark.skipif(not os.environ.get("HY3D_NATIVE_TEST_SOURCE"), reason="real pinned native source/runtime not mounted")
def test_original_native_class_folding_with_synthetic_f16_weights(monkeypatch):
    """Optional CPU oracle, actual unchanged class but NOT parent checkpoint."""
    monkeypatch.setenv("USE_SAGEATTN", "0")
    monkeypatch.setenv("HY3DGEN_DEBUG", "0")
    with mod.capture.native_modules(os.environ["HY3D_NATIVE_TEST_SOURCE"], include_dit=True) as native:
        config = {"model": {"params": {"depth": 21, "num_moe_layers": 6, "use_pos_emb": False,
            "use_attention_pooling": False, "with_decoupled_ca": False, "hidden_size": 16, "num_heads": 2,
            "context_dim": 6, "qk_norm": True, "qkv_bias": False, "in_channels": 2}}}
        block, time, embedding = mod.replay.native_components(native, config, 0)
        tensors, rows = contexts(block, time, embedding, context_width=6)
        state, selected, result = mod.fold_loaded(tensors, rows, block, time, embedding, policy())
        assert result["baseline_all_boundaries_exact"] and result["numerical_policy_passed"]
        assert state["fc1.weight"].shape == (3, 16) and selected.numel() == 3
