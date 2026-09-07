"""Small contract fixtures only: these tests demonstrate no 3D capability."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c16_hy3d_contract as hy


def native_header():
    tensors = {}
    def add(name, shape):
        tensors[name] = {"shape": list(shape), "dtype": "F16", "parameters": 1}
    for name, shape in {"x_embedder.weight": (2048,64), "x_embedder.bias": (2048,),
                        "t_embedder.mlp.0.weight": (8192,2048), "t_embedder.mlp.0.bias": (8192,),
                        "t_embedder.mlp.2.weight": (2048,8192), "t_embedder.mlp.2.bias": (2048,)}.items():
        add("model."+name,shape)
    for layer in range(21):
        p=f"model.blocks.{layer}."
        for norm in ("norm1","norm2","norm3"):
            for suffix in ("weight","bias"): add(p+norm+"."+suffix,(2048,))
        for attn, context in (("attn1",2048),("attn2",1024)):
            for proj,width in (("to_q",2048),("to_k",context),("to_v",context)):
                add(p+attn+"."+proj+".weight",(2048,width))
            for norm in ("q_norm","k_norm"): add(p+attn+"."+norm+".weight",(128,))
            add(p+attn+".out_proj.weight",(2048,2048)); add(p+attn+".out_proj.bias",(2048,))
        if layer>10:
            for name,shape in {"skip_linear.weight":(2048,4096),"skip_linear.bias":(2048,),
                               "skip_norm.weight":(2048,),"skip_norm.bias":(2048,)}.items(): add(p+name,shape)
        if layer>=15:
            add(p+"moe.gate.weight",(8,2048))
            prefixes=[p+f"moe.experts.{i}.net." for i in range(8)]+[p+"moe.shared_experts.net."]
            names=("0.proj","2")
        else: prefixes,names=[p+"mlp."],("fc1","fc2")
        for prefix in prefixes:
            for name,shape in zip(names,((8192,2048),(2048,8192))):
                add(prefix+name+".weight",shape); add(prefix+name+".bias",(shape[0],))
    add("conditioner.main_image_encoder.model.embeddings.position_embeddings",(1,1370,1024))
    for name,shape in {"encoder.input_proj.weight":(1024,55),"pre_kl.weight":(128,1024),
                       "post_kl.weight":(1024,64),"encoder.cross_attn.attn.c_q.weight":(1024,1024),
                       "encoder.cross_attn.attn.c_kv.weight":(2048,1024),
                       "geo_decoder.query_proj.weight":(1024,51),"geo_decoder.output_proj.weight":(1,1024)}.items():
        add("vae."+name,shape)
    for prefix,count in (("vae.encoder.self_attn.resblocks.",8),("vae.transformer.resblocks.",16)):
        for i in range(count): add(prefix+str(i)+".fixture",(1,))
    return tensors


def test_real_geometry_requires_self_cross_time_skip_moe_and_encoder():
    result=hy.geometry(native_header())
    assert result["moe_layers"]==list(range(15,21))
    assert result["skip_layers"]==list(range(11,21))
    assert result["surface_shape"]==[81920,7]
    assert "no adaln6" in result["time"]


@pytest.mark.parametrize("prefix",["model.blocks.20.moe.experts.7.","model.blocks.0.attn2.",
    "model.blocks.11.skip_","model.t_embedder.","vae.encoder.","model.blocks.10.",
    "model.blocks.20.moe.shared_experts."])
def test_missing_native_source_components_refused(prefix):
    tensors={k:v for k,v in native_header().items() if not k.startswith(prefix)}
    with pytest.raises(ValueError): hy.geometry(tensors)


@pytest.mark.parametrize("key",["model.pos_embed","model.blocks.0.default_modulation.weight",
                                "model.blocks.0.attn1.to_q.bias"])
def test_wrong_fixture_or_positional_variant_refused(key):
    tensors=native_header(); tensors[key]={"shape":[2048],"dtype":"F16"}
    with pytest.raises(ValueError): hy.geometry(tensors)


def test_header_only_counts_and_never_claims_body_digest(tmp_path):
    path=tmp_path/"x.safetensors"
    save_file({"x":torch.arange(12,dtype=torch.float16).view(3,4)},str(path))
    result=hy.tensor_header(path)
    assert result["tensors"]["x"]["parameters"]==12
    assert result["full_weights_hashed"] is False
    assert "file_sha256" not in result


@pytest.mark.parametrize("offsets",[[0,7],[1,9],[0,100],[-1,7]])
def test_invalid_header_span_refused(tmp_path,offsets):
    raw=json.dumps({"x":{"dtype":"F32","shape":[2],"data_offsets":offsets}}).encode()
    path=tmp_path/"bad.safetensors"; path.write_bytes(struct.pack("<Q",len(raw))+raw+b"0"*8)
    with pytest.raises(ValueError): hy.tensor_header(path)


def test_zip_pickle_vae_is_not_silently_loaded(tmp_path):
    path=tmp_path/"vae.ckpt"; path.write_bytes(b"PK\x03\x04"+b"0"*24)
    with pytest.raises(ValueError,match="ZIP/pickle"): hy.tensor_header(path)


def test_changed_parent_header_and_changed_source_both_fail_closed(tmp_path,monkeypatch):
    header={"bytes":hy.PARENT_BYTES,"header_sha256":hy.PARENT_HEADER_SHA,"tensors":native_header()}
    monkeypatch.setattr(hy,"tensor_header",lambda _:header)
    code=tmp_path/"native.py"; code.write_text("source fixture")
    monkeypatch.setattr(hy,"SOURCE_SHA",{"native.py":hashlib.sha256(code.read_bytes()).hexdigest()})
    report=hy.inspect_source("fixture only",tmp_path)
    assert report["full_weights_hashed"] is report["folded_child_proven"] is False
    header["header_sha256"]="0"*64
    with pytest.raises(ValueError,match="audited parent"): hy.inspect_source("fixture only",tmp_path)
    header["header_sha256"]=hy.PARENT_HEADER_SHA
    code.write_text("changed native attention")
    with pytest.raises(ValueError,match="source revision changed"): hy.inspect_source("fixture only",tmp_path)


@pytest.fixture
def capture(tmp_path,monkeypatch):
    # Only test the validator at tiny dimensions; there is deliberately no CLI
    # fixture switch, and production inspect_source pins the real parent header.
    monkeypatch.setattr(hy,"CAPTURE_DIMS",{"tokens":4,"hidden":8,"channels":3,"context_tokens":5,
                                          "context_width":6,"surface_points":12})
    identity={"test_only":"not real model evidence"}
    body={"schema":hy.CAPTURE_SCHEMA,"source_identity":identity,"file":"capture.safetensors",
          "samples":[{"object_sha256":"a"*64,"conditioning_sha256":"b"*64,"seed":1,"split":"fit"},
                     {"object_sha256":"c"*64,"conditioning_sha256":"d"*64,"seed":2,"split":"held"}],
          "producer":{"native_forward":True,"strict_weights":True,"eval":True,"modified_attention":False}}
    def write(kind,change=None):
        record=copy.deepcopy(body)
        if kind=="dit_block":
            record.update(kind=kind,block_index=20,attention="self_then_image_cross",
                          time_path="sigma_prepended_token",position_path="sampled_anchors_no_dit_rope")
            tensors={k:torch.randn(2,5,8) for k in ("block_input","block_output","mlp_input","mlp_output","skip_value","shared_output")}
            tensors.update(context=torch.randn(2,5,6),sigma=torch.tensor([0.25,0.75]),
                           timestep=torch.tensor([0.25,0.75]),time_token=torch.randn(2,1,8),
                           surface=torch.randn(2,12,7),query_source_index=torch.tensor([[0,2,4,6],[1,3,5,7]]),
                           router_logits=torch.randn(2,5,8))
            tensors["latent_xyz"]=tensors["surface"][...,:3].gather(1,tensors["query_source_index"][...,None].expand(-1,-1,3))
            values,ids=tensors["router_logits"].softmax(-1).topk(2,dim=-1,sorted=False)
            tensors.update(router_topk_ids=ids,router_topk_weights=values)
        else:
            record.update(kind=kind,posterior="mode",anchor_source="native_encoder_query_pc",
                          feature_kind="anchor_local_sdf",feature_samples=3)
            surface=torch.randn(2,12,7); ids=torch.tensor([[0,2,4,6],[1,3,5,7]])
            anchors=surface[...,:3].gather(1,ids[...,None].expand(-1,-1,3)); offsets=torch.randn(3,3)
            tensors={"surface":surface,"query_source_index":ids,"anchor_xyz":anchors,
                     "posterior_mode":torch.randn(2,4,3),"feature_offsets":offsets,
                     "feature_queries":anchors[:,:,None]+offsets,"feature_sdf":torch.randn(2,4,3),
                     "parent_roundtrip_sdf":torch.randn(2,4,3)}
        if change: change(record,tensors)
        save_file({k:v.contiguous() for k,v in tensors.items()},str(tmp_path/record["file"]))
        record["file_sha256"]=hashlib.sha256((tmp_path/record["file"]).read_bytes()).hexdigest()
        (tmp_path/"capture.json").write_text(json.dumps(record))
        return hy.validate_capture(tmp_path,{"source_identity":identity})
    write.directory=tmp_path
    return write


@pytest.mark.parametrize("kind",["dit_block","shape_vae_pairs"])
def test_native_capture_contract_preserves_no_capability_claim(capture,kind):
    result=capture(kind)
    assert result["contract_valid"] is True
    assert result["native_forward_independently_replayed"] is result["folded_child_proven"] is False


@pytest.mark.parametrize("tensor",["context","skip_value","time_token","shared_output","latent_xyz"])
def test_native_joint_dependencies_cannot_be_dropped(capture,tensor):
    with pytest.raises(ValueError,match="missing/extra"):
        capture("dit_block",lambda _b,t:t.pop(tensor))


def test_non_native_sigma_scaling_refused(capture):
    with pytest.raises(ValueError,match=r"sigma\*1000"):
        capture("dit_block",lambda _b,t:t["timestep"].mul_(1000))


def test_normalizing_source_topk_masses_is_refused(capture):
    def change(_b,t): t["router_topk_weights"]/=t["router_topk_weights"].sum(-1,keepdim=True)
    with pytest.raises(ValueError,match="unnormalized"): capture("dit_block",change)


def test_object_leakage_across_sigmas_cannot_pass_as_holdout(capture):
    def change(b,_t): b["samples"][1]["object_sha256"]=b["samples"][0]["object_sha256"]
    with pytest.raises(ValueError,match="leaked"): capture("dit_block",change)


@pytest.mark.parametrize("tensor",["anchor_xyz","feature_queries"])
def test_3d_pairing_requires_actual_surface_anchor_ownership(capture,tensor):
    with pytest.raises(ValueError,match="anchors|ownership"):
        capture("shape_vae_pairs",lambda _b,t:t[tensor].add_(0.1))


def test_nonfinite_source_capture_is_not_waived(capture):
    def change(_b,t): t["posterior_mode"][0,0,0]=float("nan")
    with pytest.raises(ValueError,match="nonfinite"): capture("shape_vae_pairs",change)


def test_dit_positions_cannot_be_replaced_with_a_guessed_grid(capture):
    with pytest.raises(ValueError,match="anchors"):
        capture("dit_block",lambda _b,t:t["latent_xyz"].zero_())


@pytest.mark.parametrize("field,value",[("source_identity",{}),("file_sha256","0"*64),
    ("file","../escape.safetensors"),("attention","joint_flux"),("producer",{})])
def test_substitution_stale_seal_traversal_or_producer_waiver_refused(capture,field,value):
    capture("dit_block")
    path=capture.directory/"capture.json"
    record=json.loads(path.read_text())
    identity=record["source_identity"]
    record[field]=value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError): hy.validate_capture(capture.directory,{"source_identity":identity})


def test_duplicate_header_names_are_refused():
    with pytest.raises(ValueError,match="duplicate"):
        hy.unique_json('{"same":1,"same":2}')
