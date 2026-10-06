"""Tiny mechanics only: no original checkpoint, real capture or capability claim."""
import copy
import json
from pathlib import Path
from types import MappingProxyType

import pytest
import torch
import torch.nn.functional as F

import c16_qimg_native_stack as native
from chimera_omni.qimg import QimgNativeSpec, SCHEMA, SOURCE_FILES, PARENT_HEADER_SHA256
from chimera_omni.importing.qimg import component_from_state, tensor_sha256


def tiny_native_fixture(storage="source_fp8", *, seed=461):
    """Actual reused native modules, all60 blocks, width/head32, sourceinner128.

    Returned source_state/global_state are synthetic FP8 coefficient fixtures,
    never source weights. Parent integration can capture this actual native
    forward's taps and fold them through the real corrected fitter/stack driver.
    """
    rng = torch.Generator().manual_seed(seed)
    folded = storage == "folded_fp8_bf16"
    axes = (8, 12, 12)
    specs = [QimgNativeSpec(i, 8 if folded else 128, 5 if folded else 128,
        storage, width=32, head_dim=32, axes=axes, fixture=True) for i in range(60)]
    def value(name, shape, dtype):
        out = torch.randn(shape, generator=rng, dtype=torch.float32) * .025
        if "norm_" in name or name == "txt_norm.weight": out += 1
        return out.to(getattr(torch, dtype.removeprefix("torch.")))
    global_state = {name: value(name, shape, "torch.float8_e4m3fn")
                    for name, shape in native.source_global_shapes(specs[0], 32).items()}
    globals_sha = {k: tensor_sha256(v) for k, v in global_state.items()}
    components = []
    source_state = dict(global_state)
    for spec in specs:
        state = {name: global_state[name].clone() if name.startswith("time_text_embed.") else value(name, shape, dtype)
                 for name, (shape, dtype) in spec.tensors().items()}
        hashes = {k: tensor_sha256(v) for k, v in state.items()}
        manifest = {"schema": SCHEMA, "installed": False, "spec": spec.to_dict(),
            "weights": {"file": "synthetic_block.safetensors", "sha256": "1" * 64, "tensor_sha256": hashes},
            "source": {"parent_header_sha256": PARENT_HEADER_SHA256, "frozen_file_sha256": "2" * 64,
                "code_sha256": dict(SOURCE_FILES), "capture_receipt_sha256": ["3" * 64],
                "fold_receipt_sha256": "4" * 64 if folded else None, "tensor_sha256": dict(hashes)}}
        component = component_from_state(state, manifest, allow_fixture=True)
        component.verification = MappingProxyType({**dict(component.verification),
            "candidate_tensor_sha256": dict(hashes), "synthetic_candidate_pins": True})
        components.append(component)
        source_state.update({k: v for k, v in state.items() if k.startswith("transformer_blocks.")})
    model = native.native_stack_from_components(components, global_state, global_tensor_sha256=globals_sha,
        parent_header_sha256=PARENT_HEADER_SHA256, fixture=True,
        verification={"synthetic_fixture": True, "original_checkpoint_read": False})
    inputs = {"model_input": (torch.randn((1, 16, 1, 4, 4), generator=rng) * .2).bfloat16(),
              "context": (torch.randn((1, 5, 32), generator=rng) * .2).bfloat16(),
              "sigma": torch.tensor([.37], dtype=torch.float32),
              "text_mask": torch.tensor([[0, -.25, 0, -float("inf"), 0]], dtype=torch.bfloat16)}
    return {"model": model, "components": components, "specs": specs, "source_state": source_state,
            "global_state": global_state, "global_tensor_sha256": globals_sha, "inputs": inputs, "axes": axes}


@pytest.fixture(scope="module")
def folded():
    return tiny_native_fixture("folded_fp8_bf16")


def test_full60_real_modules_mlp_calls_time_once_and_counts(folded):
    model = folded["model"]
    calls = []
    handles = []
    for index, block in model.transformer_blocks.items():
        for stream in ("img", "txt"):
            handles.append(getattr(block, stream + "_mlp").register_forward_hook(
                lambda m, a, o, key=(int(index), stream): calls.append(key)))
    handles.append(model.time_text_embed.register_forward_hook(lambda m, a, o: calls.append("time")))
    try:
        taps = model(**folded["inputs"], return_taps=True)
    finally:
        for handle in handles: handle.remove()
    assert calls == ["time"] + [(i, s) for i in range(60) for s in ("img", "txt")]
    assert list(taps["blocks"]) == list(map(str, range(60)))
    assert taps["prediction"].shape == (1, 16, 1, 4, 4)
    assert taps["prediction"].dtype == torch.bfloat16
    assert set(taps["blocks"]["59"]) == {"time_recomputed", "rotation_recomputed",
        "img_modulation", "txt_modulation", "img_attn_out", "txt_attn_out",
        "img_mlp_in", "img_mlp_out", "txt_mlp_in", "txt_mlp_out", "img_out", "txt_out"}
    report = model.runtime_report()
    for key in ("parameters", "storage_bytes", "tensor_count"):
        assert report[key] == report["expected_counts"][key]
    assert report["tensor_count"] == 1933
    assert report["shared_time_instances"] == 1 and report["folded_mlp_count"] == 120
    assert report["block_tensor_count"] + report["shared_time_tensor_count"] + report["remaining_global_tensor_count"] == 1933
    assert report["installed"] is report["accepted"] is report["capability_accepted"] is False
    assert report["fixture"] is True and report["trainable_parameters"] == 0
    assert set(model.state_dict()) == set(folded["source_state"])


def test_actual_reused_component_loop_matches_complete_global_path(folded):
    model, state, args = folded["model"], folded["global_state"], folded["inputs"]
    def linear(x, prefix):
        return F.linear(x, state[prefix + ".weight"].bfloat16(), state[prefix + ".bias"].bfloat16())
    x = args["model_input"]
    # Explicit source patch permutation and final inverse, independently of the
    # wrapper helpers, detect a swapped channel/patch axis or wrong residual feed.
    image = x.view(1,16,1,2,2,2,2).permute(0,2,3,5,1,4,6).reshape(1,4,64)
    image = linear(image, "img_in")
    text = F.rms_norm(args["context"], (32,), state["txt_norm.weight"].bfloat16(), 1e-6)
    text = linear(text, "txt_in")
    ids = native.plain_source_ids((4,4),5,device=torch.device("cpu"))
    for component in folded["components"]:
        text, image = component(image=image, text=text, sigma=args["sigma"], source_ids=ids,
                                image_shape=(4,4), text_mask=args["text_mask"])
    temb = folded["components"][0].time_text_embed(args["sigma"])
    scale, shift = linear(F.silu(temb), "norm_out.linear").chunk(2,1)
    image = torch.addcmul(shift[:,None,:], F.layer_norm(image,(32,),eps=1e-6),(1+scale)[:,None,:])
    image = linear(image, "proj_out")
    prediction = image.view(1,1,2,2,16,2,2).permute(0,4,1,2,5,3,6).reshape_as(x)
    assert torch.equal(prediction, model(**args))


def test_source_fixture_has_all1933_fp8_keys_and_actual_taps():
    data = tiny_native_fixture()
    assert len(data["source_state"]) == 1933
    assert all(v.dtype == torch.float8_e4m3fn for v in data["source_state"].values())
    assert data["specs"][0].image_inner == data["specs"][0].text_inner == 128
    taps = data["model"](**data["inputs"], return_taps=True)
    assert len(taps["blocks"]) == 60 and torch.isfinite(taps["prediction"]).all()
    assert data["model"].runtime_report()["folded_mlp_count"] == 0


def test_complete_source_header_names_and_bias_inclusive_parameter_accounting():
    assert set(native.GLOBAL_NAMES) == set(native.global_shapes())
    specs = [QimgNativeSpec(i,12288,12288) for i in range(60)]
    layout = {k:s for k,s in native.source_global_shapes(specs[0],3584).items()}
    for spec in specs: layout.update({k:s for k,(s,_) in spec.tensors().items() if k.startswith("transformer_blocks.")})
    assert len(layout) == 1933
    assert sum(__import__("math").prod(s) for s in layout.values()) == 20430401088
    assert all(s.counts()["mlp_compression_x"] == {"img":1.0,"txt":1.0} for s in specs)


def test_source_bf16_position_boundary_and_patch_inverse():
    ids = native.plain_source_ids((1024,2),1,device=torch.device("cpu"))
    h = torch.linspace(0,511,512,dtype=torch.bfloat16).float() - 256
    assert torch.equal(ids[0,1:,1],h)
    assert not torch.equal(h,torch.arange(512).float()-256)
    x = torch.arange(16*4*6).reshape(1,16,1,4,6).bfloat16()
    assert torch.equal(native.unpatchify(native.patchify(x),(4,6)),x)


@pytest.mark.parametrize("mutation", ["missing","reordered","duplicate","time","global_nan","global_shape","global_hash"])
def test_structural_identity_failures_remain_functional(folded, mutation):
    components = list(folded["components"])
    state = {k:v.clone() for k,v in folded["global_state"].items()}
    hashes = dict(folded["global_tensor_sha256"])
    if mutation == "missing": components.pop()
    if mutation == "reordered": components[0],components[1] = components[1],components[0]
    if mutation == "duplicate": components[59] = components[58]
    if mutation == "time":
        key = "time_text_embed.timestep_embedder.linear_1.bias"
        state[key].view(torch.uint8).zero_(); hashes[key] = tensor_sha256(state[key])
    if mutation == "global_nan":
        state["proj_out.bias"] = torch.full((64,),float("nan")).to(torch.float8_e4m3fn)
        hashes["proj_out.bias"] = tensor_sha256(state["proj_out.bias"])
    if mutation == "global_shape": state["img_in.weight"] = state["img_in.weight"].flatten()
    if mutation == "global_hash": hashes["txt_norm.weight"] = "0"*64
    with pytest.raises(ValueError):
        native.native_stack_from_components(components,state,global_tensor_sha256=hashes,
                                            parent_header_sha256=PARENT_HEADER_SHA256,fixture=True)


@pytest.mark.parametrize("mutation", ["input_shape","context_shape","sigma","input_nan","context_nan","mask","all_masked"])
def test_complete_native_input_functional_checks(folded, mutation):
    args = {k:v.clone() for k,v in folded["inputs"].items()}
    if mutation == "input_shape": args["model_input"] = args["model_input"][:,:,:,:,:3]
    if mutation == "context_shape": args["context"] = args["context"][:,:,:31]
    if mutation == "sigma": args["sigma"][0] = 1.01
    if mutation == "input_nan": args["model_input"][0,0,0,0,0] = float("nan")
    if mutation == "context_nan": args["context"][0,0,0] = float("nan")
    if mutation == "mask": args["text_mask"][0,0] = .1
    if mutation == "all_masked": args["text_mask"].fill_(-float("inf"))
    with pytest.raises(ValueError): folded["model"](**args)


def test_both_streams_and_late_folded_mlp_influence_actual_prediction(folded):
    model = folded["model"]
    baseline = model(**folded["inputs"], return_taps=True)
    changed = {k:v.clone() for k,v in folded["inputs"].items()}
    changed["context"][0,0] += 1
    alternate = model(**changed, return_taps=True)
    assert not torch.equal(baseline["blocks"]["59"]["txt_out"],alternate["blocks"]["59"]["txt_out"])
    assert not torch.equal(baseline["prediction"],alternate["prediction"])
    bias = model.transformer_blocks["59"].img_mlp.net[2].bias
    saved = bias.detach().clone()
    try:
        bias.add_(2)
        assert not torch.equal(baseline["prediction"],model(**folded["inputs"]))
    finally:
        bias.copy_(saved)


def test_frozen_storage_and_no_caller_alias(folded):
    model = folded["model"]
    for operation in (lambda:model.train(),lambda:model.float(),lambda:model.half(),lambda:model.bfloat16()):
        with pytest.raises(ValueError): operation()
    first = folded["components"][0]
    for k,v in first.state_dict().items():
        assert v.data_ptr() != model.state_dict()[k].data_ptr()


def test_folded_component_mutation_and_absent_candidate_pins_are_refused(folded):
    component = folded["components"][0]
    bias = component.transformer_blocks["0"].img_mlp.net[2].bias
    saved = bias.detach().clone()
    try:
        bias.add_(2)
        with pytest.raises(ValueError,match="bytes changed"):
            native.native_stack_from_components(folded["components"],folded["global_state"],
                global_tensor_sha256=folded["global_tensor_sha256"],parent_header_sha256=PARENT_HEADER_SHA256,fixture=True)
    finally:
        bias.copy_(saved)
    verification = component.verification
    try:
        component.verification = MappingProxyType({k:v for k,v in dict(verification).items() if k!="candidate_tensor_sha256"})
        with pytest.raises(ValueError,match="candidate tensor hash pins"):
            native.native_stack_from_components(folded["components"],folded["global_state"],
                global_tensor_sha256=folded["global_tensor_sha256"],parent_header_sha256=PARENT_HEADER_SHA256,fixture=True)
    finally:
        component.verification = verification


def test_actual_1933_safetensors_native_readback(folded, tmp_path):
    from safetensors.torch import save_file
    path = tmp_path / "standalone_state.safetensors"
    state = {k:v.detach().cpu().clone() for k,v in folded["model"].state_dict().items()}
    save_file(state,str(path))
    record = {"path":str(path.resolve()), "file_sha256":native.file_sha256(path),
        "tensor_sha256_schema":"torch-dtype-name + compact JSON shape + raw contiguous bytes; v1",
        "tensor_sha256":{k:tensor_sha256(v) for k,v in state.items()},
        "shapes":{k:list(v.shape) for k,v in state.items()},"dtypes":{k:str(v.dtype) for k,v in state.items()}}
    before = folded["model"](**folded["inputs"])
    native.readback_consolidated_state(folded["model"],record,root=tmp_path)
    assert torch.equal(before,folded["model"](**folded["inputs"]))
    assert folded["model"].runtime_report()["verification"]["consolidated_tensor_count"] == 1933
    wrong = copy.deepcopy(record); wrong["tensor_sha256"].pop("proj_out.bias")
    with pytest.raises(ValueError): native.readback_consolidated_state(folded["model"],wrong,root=tmp_path)
    # Native storage must remain owned after the physical file is changed.
    key = next(iter(state)); before_hash = tensor_sha256(folded["model"].state_dict()[key])
    with path.open("r+b") as handle:
        header_bytes = int.from_bytes(handle.read(8),"little")
        header = json.loads(handle.read(header_bytes))
        handle.seek(8+header_bytes+header[key]["data_offsets"][0]); byte = handle.read(1)
        handle.seek(-1,1); handle.write(bytes([byte[0]^1]))
    assert tensor_sha256(folded["model"].state_dict()[key]) == before_hash
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        native.readback_consolidated_state(folded["model"],record,root=tmp_path)


@pytest.mark.parametrize("coverage", ["missing","partial","reordered","fixture_unapproved","validation_false","error","no_export"])
def test_stack_receipt_never_inferrs_full60_from_status(tmp_path, coverage):
    body = {"schema":native.FOLD_STACK_SCHEMA,"status":"complete","fixture":True,"accepted":False,
        "capability_accepted":False,"coverage":{"required_blocks":list(range(60)),"requested_blocks":list(range(60)),
        "completed_blocks":list(range(60)),"missing_blocks":[],"full_stack_complete":True},
        "functional_validation_complete":True,"standalone_state":{}}
    if coverage == "missing": body["coverage"]["missing_blocks"] = [59]
    if coverage == "partial": body["coverage"]["requested_blocks"] = [30]
    if coverage == "reordered": body["coverage"]["completed_blocks"].reverse()
    if coverage == "validation_false": body["functional_validation_complete"] = False
    if coverage == "error": body["error"] = {"type":"synthetic_runtime_failure"}
    if coverage == "no_export": body.pop("standalone_state")
    p = tmp_path / "stack.json"; p.write_text(json.dumps(body))
    with pytest.raises(ValueError,match="complete explicit"):
        native.load_qimg_native_stack(p,native.file_sha256(p),allow_fixture=coverage!="fixture_unapproved")
