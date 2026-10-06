"""Actual tiny source/fitter/storage/native integration; no donor weights."""
import copy
import hashlib
import json
from pathlib import Path
import uuid

import pytest
import torch

import c16_qimg_stack_capture as capture
import c16_qimg_stack_fold as fold
import c16_qimg_native_stack as native
import c16_qimg_fold as original
import c16_qimg_joint_replay as joint
from test_c16_qimg_native_stack import tiny_native_fixture


def fixture_capture(root, *, blocks=None):
    """Write actual tiny native60 source taps in explicit captureV4 fixture.

    Exactly two original native forwards supply all requested block captures;
    no saved outcomes invented, no teacher/runtime class identity fabricated.
    """
    root=Path(root);root.mkdir()
    fixture=tiny_native_fixture();model=fixture["model"]
    blocks=list(range(60))if blocks is None else blocks
    frozen={}
    global_record=capture._tensors_exclusive(root/"source/global_source.safetensors",fixture["global_state"])
    for block in blocks:
        values={k:v for k,v in fixture["source_state"].items()if k.startswith(f"transformer_blocks.{block}.")or k.startswith("time_text_embed.")}
        frozen[str(block)]=capture._tensors_exclusive(root/f"source/block_{block:02d}/source.safetensors",values)
    entries=[{"id":name,"split":side,"image":{"sha256":hashlib.sha256(name.encode()).hexdigest()},
        "prompt_sha256":hashlib.sha256((name+" prompt").encode()).hexdigest()}for name,side in (("fit_image","fit"),("held_image","holdout"))]
    expected=[f"batch00/{e['id']}/sigma00"for e in entries]
    request={"blocks":blocks,"depth":60,"source":{"header_sha256":native.PARENT_HEADER_SHA256},
        "source_code_sha256":{},"inputs":[{"entries":entries}],"expected_forwards":expected,"max_tokens":2048,
        "fixture_origin":"actual native tiny source forward, synthetic FP8 coefficients; not original Comfy checkpoint"}
    runtime={"fixture":True,"parent":"actual tiny native60 source_fp8 module","teacher_calls":2}
    coefficients=capture._digest({"blocks":{b:r["tensor_sha256"]for b,r in frozen.items()},"globals":global_record["tensor_sha256"]})
    refs=[]
    for i,entry in enumerate(entries):
        args={k:v.clone()for k,v in fixture["inputs"].items()}
        if i:
            args["model_input"]=(args["model_input"].float()*1.1+.03).bfloat16()
            args["context"]=(args["context"].float()*.9-.02).bfloat16()
            args["sigma"]=torch.tensor([.61],dtype=torch.float32)
        taps=model(**args,return_taps=True)
        globals_=taps["globals"];nodes={}
        previous_img,previous_txt=globals_["img_in"],globals_["txt_in"]
        for b in range(60):
            tap=taps["blocks"][str(b)]
            if b in blocks:
                rows,cols=torch.meshgrid(torch.arange(2),torch.arange(2),indexing="ij")
                values={k:tap[k].detach().clone()for k in tap if k not in {"time_recomputed","rotation_recomputed"}}
                values.update(img_in=previous_img,txt_in=previous_txt,source_ids=globals_["source_ids"],
                    rotation_generated=globals_["rotation_recomputed"],rotation_used=globals_["rotation_used"],
                    temb=globals_["time_recomputed"],text_attention_mask=args["text_mask"],
                    prediction=taps["prediction"],sigma=args["sigma"],model_input=args["model_input"],
                    image_patch_indices=torch.stack((torch.zeros(4,dtype=torch.int64),rows.flatten(),cols.flatten()),1),
                    joint_row_kind=torch.tensor([0]*5+[1]*4,dtype=torch.int8),context_token_ids=torch.arange(5)[None],
                    context_valid=torch.isfinite(args["text_mask"]))
                assert set(values)==original.CAPTURE_KEYS
                record=capture._tensors_exclusive(root/f"block_{b:02d}/{entry['id']}.safetensors",values)
                meta={"fixture":True,"block":b,"complete_context":True,"row_order":"text_then_image",
                    "sample_id":entry["id"],"split":entry["split"],"image_sha256":entry["image"]["sha256"],
                    "prompt_sha256":entry["prompt_sha256"],"sigma":float(args["sigma"][0]),"image_tokens":4,"text_tokens":5}
                nodes[str(b)]={"tensors":record,"meta":meta}
            previous_img,previous_txt=tap["img_out"],tap["txt_out"]
        journal={"schema":capture.FORWARD_SCHEMA,"status":"complete","fixture":True,"forward_id":expected[i],
            "source_pass_id":uuid.uuid4().hex,"request_sha256":capture._digest(request),
            "actual_runtime_sha256":capture._digest(runtime),"source_coefficients_sha256":coefficients,
            "prediction_sha256":nodes[str(blocks[0])]["tensors"]["tensor_sha256"]["prediction"],"blocks":nodes}
        ref=capture._json_exclusive(root/f"forwards/forward_{i:06d}.json",journal);ref["forward_id"]=expected[i];refs.append(ref)
    body={"schema":capture.SCHEMA,"fixture":True,"request":request,"request_sha256":capture._digest(request),
        "runtime":runtime,"actual_runtime_sha256":capture._digest(runtime),"source_coefficients_sha256":coefficients,
        "frozen_sources":frozen,"global_source":global_record,"completed_forwards":refs,
        "functional_validation_complete":True,"status":"complete",
        "coverage":capture._coverage(request,{r["forward_id"]:r for r in refs},usable=True)}
    return capture._json_exclusive(root/"stack_capture_manifest.json",body),fixture


def policy_file(path):
    # Zero error/unit cosine diagnostic references deliberately remain unmet
    # for lossy actual stored tiny folds. Keep3 also lacks the old2keep rows.
    body={"schema":original.POLICY_SCHEMA,"preregistered_before_scoring":True,
        "gate_scope":"pooled_and_each_heldout_forward","policy_id":"synthetic mechanics only",
        "streams":{s:{"keep":3,"max_source_replay_rel_l2":0.,"max_heldout_rel_l2":0.,"min_heldout_cos":1.}
                   for s in original.STREAMS}}
    return capture._json_exclusive(path,body)


def test_actual_full60_capture_fold_consolidate_native_load_and_forward(tmp_path):
    cap,fixture=fixture_capture(tmp_path/"capture")
    policy=policy_file(tmp_path/"policy/policy.json")
    ref=fold.fold_stack(capture_manifest=cap["path"],capture_sha256=cap["sha256"],policy_path=policy["path"],
        policy_sha256=policy["sha256"],out=tmp_path/"folded",_fixture=True)
    body=capture._json(ref["path"],ref["sha256"])
    assert body["status"]=="complete"and body["coverage"]["completed_blocks"]==list(range(60))
    assert body["accepted"]is body["capability_accepted"]is False
    physical=body["standalone_state"]
    assert physical["tensor_count"]==1933 and physical["shared_timestep_instances_in_standalone_state"]==1
    assert len(physical["tensor_sha256"])==1933 and len([k for k in physical["tensor_sha256"]if k.startswith("time_text_embed.")])==4
    assert sum("net.2."in k and physical["dtypes"][k]=="torch.bfloat16"for k in physical["dtypes"])==240
    numeric=[]
    for row in body["blocks"]:
        receipt=capture._json(row["fold_receipt"]["path"],row["fold_receipt"]["sha256"])
        assert receipt["structural_artifact_valid"]is True and receipt["candidate_written"]is True
        assert receipt["persisted_tensors_rescored_without_refit"]is True
        numeric.append(receipt["numerical_policy_passed"])
    assert not any(numeric),"finite out-of-reference diagnostics must not veto any of60 actual candidates"
    loaded=native.load_qimg_native_stack(ref["path"],ref["sha256"],allow_fixture=True,fixture_axes=fixture["axes"])
    taps=loaded(**fixture["inputs"],return_taps=True)
    assert len(taps["blocks"])==60 and torch.isfinite(taps["prediction"]).all()
    report=loaded.runtime_report()
    assert report["tensor_count"]==1933 and report["shared_time_instances"]==1 and report["folded_mlp_count"]==120
    assert report["parameters"]==physical["parameters"]and report["storage_bytes"]==physical["storage_bytes"]
    assert loaded.verification["consolidated_state_loaded_into_actual_native_modules"]is True
    assert set(loaded.state_dict())==set(fixture["source_state"])
    # Exact stored native bytes persist under readback; no blanket FP8 recast.
    assert {k:native.tensor_sha256(v)for k,v in loaded.state_dict().items()}==physical["tensor_sha256"]


def test_partial_resume_no_duplicate_fit_and_missing_coverage(tmp_path,monkeypatch):
    cap,fixture=fixture_capture(tmp_path/"capture",blocks=[0,1])
    policy=policy_file(tmp_path/"policy/policy.json")
    real=fold._persist_block;calls=[]
    def fail_second(data,*args,**kwargs):
        calls.append(data["block"])
        if data["block"]==1:raise RuntimeError("synthetic interrupted second component")
        return real(data,*args,**kwargs)
    monkeypatch.setattr(fold,"_persist_block",fail_second)
    with pytest.raises(RuntimeError,match="interrupted"):
        fold.fold_stack(capture_manifest=cap["path"],capture_sha256=cap["sha256"],policy_path=policy["path"],
            policy_sha256=policy["sha256"],out=tmp_path/"interrupted",_fixture=True)
    old=tmp_path/"interrupted/stack_fold_manifest.json";old_sha=native.file_sha256(old)
    oldbody=capture._json(old,old_sha)
    assert oldbody["coverage"]["completed_blocks"]==[0]and oldbody["status"]=="partial"
    monkeypatch.setattr(fold,"_persist_block",real)
    bad_receipt=capture._json(oldbody["blocks"][0]["fold_receipt"]["path"],oldbody["blocks"][0]["fold_receipt"]["sha256"])
    bad_receipt["policy"]["streams"]["img"]["keep"]=2
    bad_ref=capture._json_exclusive(tmp_path/"mismatched_component/fold_receipt.json",bad_receipt)
    bad_body=copy.deepcopy(oldbody);bad_body["blocks"][0]["fold_receipt"]=bad_ref
    bad_resume=capture._json_exclusive(tmp_path/"interrupted/mismatched_resume.json",bad_body)
    with pytest.raises(ValueError,match="policy/recipe"):
        fold.fold_stack(capture_manifest=cap["path"],capture_sha256=cap["sha256"],policy_path=policy["path"],
            policy_sha256=policy["sha256"],out=tmp_path/"bad_resume_output",resume_manifest=bad_resume["path"],
            resume_sha256=bad_resume["sha256"],_fixture=True)
    assert not(tmp_path/"bad_resume_output").exists()
    result=fold.fold_stack(capture_manifest=cap["path"],capture_sha256=cap["sha256"],policy_path=policy["path"],
        policy_sha256=policy["sha256"],out=tmp_path/"resumed",resume_manifest=old,resume_sha256=old_sha,_fixture=True)
    body=capture._json(result["path"],result["sha256"])
    assert body["blocks"][0]==oldbody["blocks"][0]
    assert body["coverage"]["completed_blocks"]==[0,1]and len(body["coverage"]["missing_blocks"])==58
    assert body["status"]=="partial"and body["standalone_state"]is None
    assert not (tmp_path/"resumed/block_00").exists() and native.file_sha256(old)==old_sha
    with pytest.raises(ValueError,match="complete"):
        native.load_qimg_native_stack(result["path"],result["sha256"],allow_fixture=True)


def test_metadata_only_native_mode_and_payload_tamper(tmp_path,monkeypatch):
    cap,_=fixture_capture(tmp_path/"capture",blocks=[0])
    index=fold.CaptureIndex(cap["path"],cap["sha256"],fixture=True)
    original_verify=capture._verify_tensors;verified=[]
    def spy(record,roots):
        verified.append(record["path"])
        return original_verify(record,roots)
    monkeypatch.setattr(capture,"_verify_tensors",spy)
    data=fold.load_block_inputs(cap["path"],cap["sha256"],0,fixture=True,index=index,load_forwards=False)
    assert len(verified)==2 and data["forwards"]==[] and data["capture_forward_payloads_verified"]is False
    assert data["capture_receipts"]==[cap]
    path=Path(index.journals[index.request["expected_forwards"][0]]["body"]["blocks"]["0"]["tensors"]["path"])
    raw=path.read_bytes();path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    with pytest.raises(ValueError,match="identity"):
        fold.load_block_inputs(cap["path"],cap["sha256"],0,fixture=True,index=index)


def test_complete_metadata_cannot_hide_missing_journal_and_production_fixture_refusal(tmp_path):
    cap,_=fixture_capture(tmp_path/"capture",blocks=[0])
    with pytest.raises(ValueError,match="pinned|ancestry|pins"):
        fold.CaptureIndex(cap["path"],cap["sha256"],fixture=False)
    body=capture._json(cap["path"],cap["sha256"])
    body["completed_forwards"].pop()
    ref=capture._json_exclusive(tmp_path/"capture/dishonest_complete.json",body)
    with pytest.raises(ValueError,match="coverage|complete"):
        fold.CaptureIndex(ref["path"],ref["sha256"],fixture=True)


def test_payload_mutation_after_fit_refuses_canonical_publication(tmp_path,monkeypatch):
    cap,_=fixture_capture(tmp_path/"capture",blocks=[0])
    policy=policy_file(tmp_path/"policy/policy.json")
    actual=fold.development._fit_loaded_candidate
    def mutate_after_fit(data,*args,**kwargs):
        result=actual(data,*args,**kwargs)
        path=Path(data["forwards"][0]["file"]);raw=path.read_bytes()
        path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        return result
    monkeypatch.setattr(fold.development,"_fit_loaded_candidate",mutate_after_fit)
    with pytest.raises(ValueError,match="payload changed"):
        fold.fold_stack(capture_manifest=cap["path"],capture_sha256=cap["sha256"],policy_path=policy["path"],
            policy_sha256=policy["sha256"],out=tmp_path/"refused",_fixture=True)
    assert not(tmp_path/"refused/block_00/fold_receipt.json").exists()
    body=json.loads((tmp_path/"refused/stack_fold_manifest.json").read_text())
    assert body["status"]=="partial"and body["coverage"]["completed_blocks"]==[]
    assert body["error"]["type"]=="ValueError"and body["incomplete_artifacts_preserved"]is True


@pytest.mark.parametrize("mutation",["sigma","source_ids","stream_continuity"])
def test_one_pass_id_cannot_hide_foreign_block_forward(tmp_path,mutation):
    cap,_=fixture_capture(tmp_path/"capture",blocks=[0,1])
    body=capture._json(cap["path"],cap["sha256"]);ref=body["completed_forwards"][0]
    journal=capture._json(ref["path"],ref["sha256"])
    if mutation=="sigma":journal["blocks"]["1"]["meta"]["sigma"]+=.1
    elif mutation=="source_ids":journal["blocks"]["1"]["tensors"]["tensor_sha256"]["source_ids"]="a"*64
    else:journal["blocks"]["1"]["tensors"]["tensor_sha256"]["img_in"]="a"*64
    altered=capture._json_exclusive(tmp_path/"capture/forwards/altered.json",journal)
    altered["forward_id"]=ref["forward_id"];body["completed_forwards"][0]=altered
    new=capture._json_exclusive(tmp_path/"capture/altered_manifest.json",body)
    with pytest.raises(ValueError,match="invariant|continuity"):
        fold.CaptureIndex(new["path"],new["sha256"],fixture=True)
