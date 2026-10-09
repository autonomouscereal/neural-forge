"""Actual tiny Torch hook/I/O tests, never real model/encoding evidence.

Driver fixtures mock parent/header/source loaders; only simultaneous hook API,
source-pass sharing, tensor bytes, journals and resume behavior are established.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from safetensors.torch import save_file
import c16_qimg_stack_capture as stack
from test_c16_qimg_capture import FixtureBlock, FixturePositions


class TinyAPI(nn.Module):
    def __init__(self, depth=3, context_width=6):
        super().__init__()
        self.transformer_blocks=nn.ModuleList([FixtureBlock() for _ in range(depth)])
        self.pe_embedder=FixturePositions()
        self.img_in=nn.Linear(64,8);self.txt_in=nn.Linear(context_width,8);self.final=nn.Linear(8,64)
        self.global_a=nn.Parameter(torch.ones(1));self.global_b=nn.Parameter(torch.ones(1));self.global_c=nn.Parameter(torch.ones(1))
        self.time_text_embed=nn.Module();self.time_text_embed.timestep_embedder=nn.Module()
        self.time_text_embed.timestep_embedder.linear_1=nn.Linear(1,8)
        self.time_text_embed.timestep_embedder.linear_2=nn.Linear(8,8)
        self.calls=0;self.fail_after=None

    def forward(self,x,sigma,context,attention_mask=None,transformer_options={}):
        self.calls+=1
        if self.fail_after is not None and self.calls>self.fail_after:raise RuntimeError("injected source-forward failure")
        h,w=x.shape[-2]//2,x.shape[-1]//2
        rows=x[:,:,0].reshape(1,16,h,2,w,2).permute(0,2,4,1,3,5).reshape(1,h*w,64)
        img,txt=self.img_in(rows),self.txt_in(context)
        rr,cc=torch.meshgrid(torch.arange(h),torch.arange(w),indexing="ij")
        image_ids=torch.stack((torch.zeros(h*w),rr.flatten()-h//2,cc.flatten()-w//2),1)
        start=max(h//2,w//2);text_ids=torch.arange(start,start+txt.shape[1]).float()[:,None].expand(-1,3)
        ids=torch.cat((text_ids,image_ids))[None]
        pe=self.pe_embedder(ids).to(x.dtype)
        time=self.time_text_embed.timestep_embedder
        temb=time.linear_2(time.linear_1(sigma[:,None].to(x.dtype)))
        mask=(attention_mask-1).to(x.dtype)*1e30 if attention_mask is not None else None
        for block in self.transformer_blocks:
            txt,img=block(hidden_states=img,encoder_hidden_states=txt,encoder_hidden_states_mask=mask,temb=temb,
                          image_rotary_emb=pe,timestep_zero_index=None,transformer_options=transformer_options)
        predicted=self.final(img).reshape(1,h,w,16,2,2).permute(0,3,1,4,2,5).reshape_as(x)
        return predicted


def fixture(dtype=torch.float32):
    torch.manual_seed(812)
    model=TinyAPI().to(dtype).eval()
    return model,torch.randn(1,16,1,4,4).to(dtype),torch.tensor([.6]),torch.randn(1,3,6).to(dtype),torch.tensor([[1,1,0]]),torch.tensor([[7,8,9]]),torch.tensor([[True,True,False]])


def no_hooks(model):
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
def test_shared_pass_matches_each_single_block_actual_hook_api(dtype):
    model,x,sigma,context,mask,ids,valid=fixture(dtype)
    emitted={}
    stack.capture_stack_forward(model,[0,1,2],x,sigma,context,mask,ids,valid,
                                fixture=True,emit=lambda b,t,m:emitted.setdefault(b,(dict(t),m)))
    assert model.calls==1
    no_hooks(model)
    for b in range(3):
        expected,meta=stack.single.capture_forward(model,b,x,sigma,context,mask,fixture=True)
        got=emitted[b][0]
        assert all(stack.single.tensor_sha(got[k])==stack.single.tensor_sha(v) for k,v in expected.items())
        assert emitted[b][1]["block"]==b and emitted[b][1]["shared_source_forward"]
        assert got["joint_row_kind"].tolist()==[0,0,0,1,1,1,1]
        assert got["context_valid"].tolist()==valid.tolist()
        assert got["img_in"].shape==(1,4,8) and got["txt_in"].shape==(1,3,8)
    assert model.calls==4
    no_hooks(model)


def test_emitter_failure_removes_every_hook_and_does_not_repeat_source_forward():
    model,*args=fixture();seen=[]
    def emit(b,t,m):
        seen.append(b)
        if b==1:raise OSError("injected stream-write failure")
        return b
    with pytest.raises(OSError):stack.capture_stack_forward(model,[0,1,2],*args,fixture=True,emit=emit)
    assert seen==[0,1] and model.calls==1
    no_hooks(model)


def test_source_error_before_complete_prediction_emits_nothing_and_cleans_hooks():
    model,*args=fixture();model.transformer_blocks[1].fail=True;seen=[]
    with pytest.raises(RuntimeError):stack.capture_stack_forward(model,[0,1,2],*args,fixture=True,emit=lambda *x:seen.append(x))
    assert not seen and model.calls==1
    no_hooks(model)


@pytest.mark.parametrize("blocks",[[1,0],[0,0],[-1],[3],[],[True]])
def test_invalid_requested_block_contract_refuses_before_any_forward(blocks):
    model,*args=fixture()
    with pytest.raises(ValueError):stack.capture_stack_forward(model,blocks,*args,fixture=True,emit=lambda *x:None)
    assert model.calls==0;no_hooks(model)


@pytest.mark.parametrize("field",["sigma","input","context"])
def test_actual_nonfinite_inputs_refuse_without_forward(field):
    model,x,sigma,context,mask,ids,valid=fixture()
    {"sigma":sigma,"input":x,"context":context}[field].flatten()[0]=float("nan")
    with pytest.raises(ValueError):stack.capture_stack_forward(model,[0,1],x,sigma,context,mask,ids,valid,fixture=True,emit=lambda *x:None)
    assert model.calls==0;no_hooks(model)


def test_synthetic_model_is_not_a_production_source_class():
    model,*args=fixture()
    with pytest.raises(ValueError,match="actual Qwen-Image"):stack.capture_stack_forward(model,[0,1],*args,emit=lambda *x:None)
    assert model.calls==0;no_hooks(model)


def test_nonfinite_actual_prediction_refuses_and_cleans_hooks():
    model,*args=fixture();model.final.weight.data.fill_(float("nan"))
    with pytest.raises(ValueError):stack.capture_stack_forward(model,[0,1],*args,fixture=True,emit=lambda *x:None)
    assert model.calls==1;no_hooks(model)


def test_negative_infinity_native_mask_is_allowed_but_nan_is_not():
    stack._finite("text_attention_mask",torch.tensor([[0.,float("-inf")]]))
    with pytest.raises(ValueError):stack._finite("text_attention_mask",torch.tensor([[float("nan")]]))


def test_tensor_artifacts_are_exclusive_finite_and_fully_reverified(tmp_path):
    values={"huge_but_finite":torch.full((2,3),1e10,dtype=torch.bfloat16),"index":torch.arange(3)}
    record=stack._tensors_exclusive(tmp_path/"real.safetensors",values)
    got=stack._verify_tensors(record,[tmp_path])
    assert all(stack.single.tensor_sha(got[k])==stack.single.tensor_sha(v) for k,v in values.items())
    with pytest.raises(ValueError):stack._tensors_exclusive(tmp_path/"real.safetensors",values)
    record["shapes"]["index"]=[4]
    with pytest.raises(ValueError):stack._verify_tensors(record,[tmp_path])
    with pytest.raises(ValueError):stack._tensors_exclusive(tmp_path/"nonfinite.safetensors",{"bad":torch.tensor([float("inf")])})


def test_json_publication_is_exclusive_complete_and_pinned(tmp_path):
    ref=stack._json_exclusive(tmp_path/"journal.json",{"value":7,"complete":True})
    assert stack._json(ref["path"],ref["sha256"])["value"]==7
    with pytest.raises(FileExistsError):stack._json_exclusive(tmp_path/"journal.json",{"value":9})
    with pytest.raises(ValueError):stack._json(ref["path"],"0"*64)
    with pytest.raises(ValueError):stack._json_exclusive(tmp_path/"bad.json",{"bad":float("nan")})


def test_declared_subset_is_never_full60_coverage():
    request={"blocks":[0,30,59],"depth":60,"expected_forwards":["a","b"]}
    assert not stack._coverage(request,{"a":1,"b":2},usable=True)["full_parent_capture_complete"]
    request["blocks"]=list(range(60))
    assert stack._coverage(request,{"a":1,"b":2},usable=True)["full_parent_capture_complete"]
    assert not stack._coverage(request,{"a":1,"b":2})["full_parent_capture_complete"]
    assert stack._coverage(request,{"a":1})["missing_forwards"]==["b"]


def driver_fixture(tmp_path,monkeypatch):
    """Mock source/geometry identity only; exercise real encoded-input parser/I/O."""
    torch.manual_seed(127)
    # A private fixture subtype simulates the source class guard without
    # changing shared type metadata or relying on pytest to restore __name__.
    driver_type=type("QwenImageTransformer2DModel",(TinyAPI,),{"__module__":"comfy.ldm.qwen_image.model"})
    model=driver_type(depth=60,context_width=3584).eval()
    source=tmp_path/"fixture_sources";source.mkdir()
    weights=source/"tiny_weights.safetensors";save_file(model.state_dict(),str(weights))
    header,identity=stack.single.read_header(weights)
    monkeypatch.setattr(stack.single,"PARENT_HEADER_SHA256",identity["header_sha256"])
    def inspect(path,b):
        return {"block":b,"depth":60,"source_identity":identity,"width":8,"head_dim":8,
                "block_parameters":sum(v.numel() for k,v in model.state_dict().items() if k.startswith(f"transformer_blocks.{b}.")),
                "retained_global_time_parameters":sum(v.numel() for k,v in model.state_dict().items() if k.startswith("time_text_embed."))}
    monkeypatch.setattr(stack.single,"inspect_weights",inspect)
    monkeypatch.setattr(stack.single,"verify_source_code",lambda root:{"scope":"synthetic source-loader test stub"})
    loads=[]
    def loader(*args):
        loads.append(1)
        return object(),model,torch.float32,{"device":"cpu","torch":torch.__version__,"fixture_source":"tiny synthetic API, not actual native runtime"}
    monkeypatch.setattr(stack.single,"load_parent",loader)
    monkeypatch.setattr(stack.shutil,"disk_usage",lambda p:SimpleNamespace(free=16<<30))
    vae=source/"tiny_vae.safetensors";save_file({"tiny":torch.ones(2,2)},str(vae))
    text=source/"tiny_text.safetensors";save_file({"tiny":torch.ones(2,2)},str(text))
    entries=[];links={}
    for j,(name,split) in enumerate((("fit","fit"),("held","holdout"))):
        image=source/(name+".image");image.write_bytes(("synthetic fixture image "+name).encode())
        tensors=source/(name+".safetensors")
        state={"model_inputs":torch.randn(1,16,1,4,4),"sigma":torch.tensor([.2+j*.5]),"context":torch.randn(1,3,3584),
               "context_mask":torch.tensor([[True,True,False]]),"context_token_ids":torch.tensor([[7,8,9]])}
        save_file(state,str(tensors));prompt="synthetic test prompt "+name;pd=hashlib.sha256(prompt.encode()).hexdigest()
        ent={"id":name,"split":split,"image":{"path":str(image),"sha256":stack.single.file_sha(image)},"prompt":prompt,"prompt_sha256":pd,
             "tensors":{"path":str(tensors),"sha256":stack.single.file_sha(tensors)}}
        entries.append(ent);links[name]={"image_sha256":ent["image"]["sha256"],"prompt_sha256":pd,"tensor_file_sha256":ent["tensors"]["sha256"]}
    encoding=source/"encoding.json"
    enc={"status":"complete","kind":"real_image_vae_qwen25_context","latent_space":"wan21_model_input","entries":links}
    for role,path in (("vae",vae),("text_encoder",text)):enc[role]={"path":str(path),"header_sha256":stack.single.read_header(path)[1]["header_sha256"]}
    encoding.write_text(json.dumps(enc),encoding="utf-8")
    manifest=source/"capture_inputs.json"
    manifest.write_text(json.dumps({"schema":stack.single.INPUT_SCHEMA,"kind":enc["kind"],"latent_space":enc["latent_space"],"context_kind":"qwen25_vl_7b_hidden_states",
        "encoding_receipt":{"path":str(encoding),"sha256":stack.single.file_sha(encoding)},"entries":entries}),encoding="utf-8")
    return {"weights":weights,"inputs":[manifest],"comfy_root":tmp_path/"mock_comfy","device":"cpu","blocks":[0,30,59]},model,loads


def test_driver_loads_parent_once_shares_each_forward_and_emits_real_partition_receipts(tmp_path,monkeypatch):
    kwargs,model,loads=driver_fixture(tmp_path,monkeypatch)
    ref=stack.capture(**kwargs,out=tmp_path/"out")
    body=stack._json(ref["path"],ref["sha256"])
    assert loads==[1] and model.calls==2
    assert body["status"]=="complete" and body["functional_validation_complete"]
    assert body["coverage"]["requested_coverage_complete"] and not body["coverage"]["full_parent_capture_complete"]
    assert set(body["block_receipts"])=={"0","30","59"}
    for br in body["block_receipts"].values():
        part=stack._json(br["path"],br["sha256"])
        assert part["status"]=="complete" and len(part["completed_forward_journals"])==2
    completed,roots,_=stack._load_resume(ref["path"],ref["sha256"],body["request"])
    assert len(completed)==2 and roots==[(tmp_path/"out").resolve()]
    assert not body["capability_accepted"] and not body["folded"] and not body["installed"]
    no_hooks(model)


def test_source_failure_has_explicit_missing_coverage_and_verified_new_output_resume(tmp_path,monkeypatch):
    kwargs,model,loads=driver_fixture(tmp_path,monkeypatch);model.fail_after=1
    with pytest.raises(RuntimeError):stack.capture(**kwargs,out=tmp_path/"first")
    path=tmp_path/"first/stack_capture_manifest.json";sha=stack.single.file_sha(path)
    body=stack._json(path,sha)
    assert body["status"]=="partial" and len(body["completed_forwards"])==1
    assert len(body["coverage"]["missing_forwards"])==1 and not body["functional_validation_complete"]
    first_forward=copy.deepcopy(body["completed_forwards"][0]);first_hash=sha
    model.fail_after=None
    ref=stack.capture(**kwargs,out=tmp_path/"second",resume_manifest=path,resume_sha256=sha)
    final=stack._json(ref["path"],ref["sha256"])
    assert final["status"]=="complete" and final["completed_forwards"][0]==first_forward
    assert loads==[1,1] and model.calls==3
    assert stack.single.file_sha(path)==first_hash
    stack._load_resume(ref["path"],ref["sha256"],final["request"])
    no_hooks(model)


def test_partial_block_write_is_preserved_but_never_counted_as_completed_pass(tmp_path,monkeypatch):
    kwargs,model,_=driver_fixture(tmp_path,monkeypatch)
    original=stack._tensors_exclusive
    def fail(path,tensors):
        if "/block_30/batch" in Path(path).as_posix():raise OSError("injected block30 disk fault")
        return original(path,tensors)
    monkeypatch.setattr(stack,"_tensors_exclusive",fail)
    with pytest.raises(OSError):stack.capture(**kwargs,out=tmp_path/"partial")
    path=tmp_path/"partial/stack_capture_manifest.json";body=stack._json(path,stack.single.file_sha(path))
    assert body["status"]=="partial" and not body["completed_forwards"] and len(body["coverage"]["missing_forwards"])==2
    assert list((tmp_path/"partial/block_00/batch00").glob("*.safetensors"))
    assert not list((tmp_path/"partial/forwards").glob("*.json"))
    no_hooks(model)


def test_resume_rejects_changed_request_and_payload_hash_without_loading_parent(tmp_path,monkeypatch):
    kwargs,model,loads=driver_fixture(tmp_path,monkeypatch)
    ref=stack.capture(**kwargs,out=tmp_path/"first");body=stack._json(ref["path"],ref["sha256"])
    changed=copy.deepcopy(body["request"]);changed["blocks"]=[0,1]
    with pytest.raises(ValueError):stack._load_resume(ref["path"],ref["sha256"],changed)
    forward=stack._json(body["completed_forwards"][0]["path"],body["completed_forwards"][0]["sha256"])
    p=Path(forward["blocks"]["0"]["tensors"]["path"])
    raw=p.read_bytes();p.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    with pytest.raises(ValueError):stack.capture(**kwargs,out=tmp_path/"second",resume_manifest=ref["path"],resume_sha256=ref["sha256"])
    assert loads==[1]


def test_complete_resume_is_verified_without_reloading_or_replaying_parent(tmp_path,monkeypatch):
    kwargs,model,loads=driver_fixture(tmp_path,monkeypatch)
    ref=stack.capture(**kwargs,out=tmp_path/"first")
    second=stack.capture(**kwargs,out=tmp_path/"second",resume_manifest=ref["path"],resume_sha256=ref["sha256"])
    body=stack._json(second["path"],second["sha256"])
    assert body["status"]=="complete" and body["parent_loaded_this_attempt"] is False
    assert loads==[1] and model.calls==2


def test_final_identity_failure_never_claims_complete_even_if_all_forwards_exist(tmp_path,monkeypatch):
    kwargs,model,_=driver_fixture(tmp_path,monkeypatch)
    stat=stack._stat;calls=[]
    def changed(path):
        calls.append(1);value=stat(path)
        return value if len(calls)==1 else {**value,"mtime_ns":value["mtime_ns"]+1}
    monkeypatch.setattr(stack,"_stat",changed)
    with pytest.raises(ValueError):stack.capture(**kwargs,out=tmp_path/"first")
    path=tmp_path/"first/stack_capture_manifest.json";body=stack._json(path,stack.single.file_sha(path))
    assert len(body["completed_forwards"])==2 and body["coverage"]["requested_coverage_complete"]
    assert body["status"]=="partial" and not body["functional_validation_complete"]
    no_hooks(model)


def test_validate_only_never_loads_parent_and_existing_output_refuses(tmp_path,monkeypatch):
    kwargs,model,loads=driver_fixture(tmp_path,monkeypatch)
    result=stack.capture(**kwargs,out=tmp_path/"out",validate_only=True)
    assert result["validated"] and not result["capture_launched"] and not loads and model.calls==0
    (tmp_path/"out").mkdir()
    with pytest.raises(ValueError):stack.capture(**kwargs,out=tmp_path/"out")
    assert not loads


@pytest.mark.parametrize("value",["59-0","0,0","0,60","x","0,1-2,2",
                                 "0-1000000000","1000000000","0,"*300,"0-"+"9"*513,""])
def test_cli_block_parser_rejects_ambiguous_or_out_of_source_indices(value):
    with pytest.raises(ValueError):stack.parse_blocks(value)


def test_full60_parser_is_explicit_exact_coverage():
    assert stack.parse_blocks("0-59")==list(range(60))


def test_tiny_all60_driver_runs_two_shared_passes_and_records_all_actual_blocks(tmp_path,monkeypatch):
    kwargs,model,loads=driver_fixture(tmp_path,monkeypatch);kwargs["blocks"]=list(range(60))
    ref=stack.capture(**kwargs,out=tmp_path/"all60")
    body=stack._json(ref["path"],ref["sha256"])
    assert loads==[1] and model.calls==2
    assert body["coverage"]["full_parent_capture_complete"] and len(body["block_receipts"])==60
    assert not body["capability_accepted"] and not body["folded"]
    for forward in body["completed_forwards"]:
        journal=stack._json(forward["path"],forward["sha256"])
        assert set(journal["blocks"])==set(map(str,range(60)))
        assert len({c["tensors"]["tensor_sha256"]["prediction"] for c in journal["blocks"].values()})==1
    no_hooks(model)


def test_insufficient_actual_disk_budget_refuses_before_parent_load(tmp_path,monkeypatch):
    kwargs,model,loads=driver_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(stack.shutil,"disk_usage",lambda p:SimpleNamespace(free=1<<30))
    with pytest.raises(ValueError,match="less than4GiB"):stack.capture(**kwargs,out=tmp_path/"out")
    assert not loads and model.calls==0 and not (tmp_path/"out").exists()


def test_loaded_depth_mismatch_is_functional_failure_not_claimed_capture(tmp_path,monkeypatch):
    kwargs,model,loads=driver_fixture(tmp_path,monkeypatch)
    model.transformer_blocks=model.transformer_blocks[:3]
    with pytest.raises(ValueError,match="depth"):stack.capture(**kwargs,out=tmp_path/"out")
    path=tmp_path/"out/stack_capture_manifest.json";body=stack._json(path,stack.single.file_sha(path))
    assert body["status"]=="partial" and not body["completed_forwards"] and not body["functional_validation_complete"]
    no_hooks(model)


@pytest.mark.parametrize("field",["runtime","coefficients","pass_id","block","split","image","prompt","prediction"])
def test_repinning_foreign_shared_forward_metadata_does_not_make_it_eligible_for_resume(tmp_path,monkeypatch,field):
    kwargs,model,loads=driver_fixture(tmp_path,monkeypatch)
    root=tmp_path/"first";ref=stack.capture(**kwargs,out=root);body=stack._json(ref["path"],ref["sha256"])
    journal=stack._json(body["completed_forwards"][0]["path"],body["completed_forwards"][0]["sha256"])
    if field=="runtime":journal["actual_runtime_sha256"]="0"*64
    elif field=="coefficients":journal["source_coefficients_sha256"]="0"*64
    elif field=="pass_id":journal["source_pass_id"]="not-a-source-pass"
    elif field=="prediction":journal["prediction_sha256"]="0"*64
    else:
        key={"block":"block","split":"split","image":"image_sha256","prompt":"prompt_sha256"}[field]
        journal["blocks"]["0"]["meta"][key]=1 if field=="block" else "foreign"
    foreign=stack._json_exclusive(root/"foreign_forward.json",journal)
    foreign["forward_id"]=body["completed_forwards"][0]["forward_id"];body["completed_forwards"][0]=foreign
    # Remove redundant views so this test specifically reaches journal checks.
    body.pop("block_receipts")
    altered=stack._json_exclusive(root/"foreign_manifest.json",body)
    with pytest.raises(ValueError):stack._load_resume(altered["path"],altered["sha256"],body["request"])
    assert loads==[1] and model.calls==2


def test_repeated_source_pass_id_and_unpinned_block_receipt_refuse_resume(tmp_path,monkeypatch):
    kwargs,_,_=driver_fixture(tmp_path,monkeypatch);root=tmp_path/"first"
    ref=stack.capture(**kwargs,out=root);body=stack._json(ref["path"],ref["sha256"])
    first=stack._json(body["completed_forwards"][0]["path"],body["completed_forwards"][0]["sha256"])
    second=stack._json(body["completed_forwards"][1]["path"],body["completed_forwards"][1]["sha256"])
    second["source_pass_id"]=first["source_pass_id"]
    new=stack._json_exclusive(root/"duplicate_pass.json",second);new["forward_id"]=body["completed_forwards"][1]["forward_id"]
    duplicate=copy.deepcopy(body);duplicate["completed_forwards"][1]=new;duplicate.pop("block_receipts")
    altered=stack._json_exclusive(root/"duplicate_manifest.json",duplicate)
    with pytest.raises(ValueError,match="unique"):stack._load_resume(altered["path"],altered["sha256"],body["request"])
    body["block_receipts"]["30"]["sha256"]="0"*64
    altered=stack._json_exclusive(root/"bad_receipt_manifest.json",body)
    with pytest.raises(ValueError,match="identity"):stack._load_resume(altered["path"],altered["sha256"],body["request"])


def test_cross_batch_image_split_leak_refuses_before_source_load(tmp_path,monkeypatch):
    kwargs,model,loads=driver_fixture(tmp_path,monkeypatch)
    body=json.loads(kwargs["inputs"][0].read_text());body["entries"]=body["entries"][:1];body["entries"][0]["split"]="holdout"
    second=kwargs["inputs"][0].with_name("second_batch.json");second.write_text(json.dumps(body))
    kwargs["inputs"].append(second)
    with pytest.raises(ValueError,match="input-batch"):stack.capture(**kwargs,out=tmp_path/"out")
    assert not loads and model.calls==0 and not (tmp_path/"out").exists()


@pytest.mark.parametrize("changed",["tensor","image","helper","comfy"])
def test_final_input_and_implementation_mutation_never_claims_complete(tmp_path,monkeypatch,changed):
    kwargs,model,loads=driver_fixture(tmp_path,monkeypatch)
    if changed in {"helper","comfy"}:
        target=stack if changed=="helper" else stack.single
        name="_implementation" if changed=="helper" else "verify_source_code"
        original=getattr(target,name);calls=[]
        def after(*args):
            value=original(*args);calls.append(1)
            return value if len(calls)==1 else {**value,"changed_after_actual_capture":"different"}
        monkeypatch.setattr(target,name,after)
    else:
        original=stack._tensors_exclusive
        def after(path,tensors):
            record=original(path,tensors)
            if Path(path).name=="held_sigma00.safetensors":
                name="held.safetensors" if changed=="tensor" else "held.image"
                source=tmp_path/"fixture_sources"/name;raw=source.read_bytes();source.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            return record
        monkeypatch.setattr(stack,"_tensors_exclusive",after)
    with pytest.raises(ValueError):stack.capture(**kwargs,out=tmp_path/"out")
    path=tmp_path/"out/stack_capture_manifest.json";body=stack._json(path,stack.single.file_sha(path))
    assert body["status"]=="partial" and not body["functional_validation_complete"]
    assert len(body["completed_forwards"])==2 and loads==[1] and model.calls==2
    no_hooks(model)
