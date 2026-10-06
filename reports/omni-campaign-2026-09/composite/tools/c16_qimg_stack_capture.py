#!/usr/bin/env python3
"""Authentic shared-forward multi-block QIMG capture, never fold acceptance.

The production CLI verifies actual encoded inputs and the existing pinned Comfy
parent, loads it once, and streams actual block taps to new immutable artifacts.
Completed shared-forward journals can resume into a NEW output directory. A
partial forward remains evidence but is never silently counted as completed.
No fidelity, cosine, update or compression target controls capture availability.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid
import torch
from safetensors import safe_open
from safetensors.torch import load_file
import c16_qimg_capture as single

SCHEMA = "neural_forge.qimg_stack_capture.v1"
FORWARD_SCHEMA = "neural_forge.qimg_stack_source_forward.v1"
BLOCK_SCHEMA = "neural_forge.qimg_stack_block_capture.v1"
MAX_BATCHES = 4
MAX_RESUMES = 8
MAX_JSON_BYTES = 16 << 20


def _now(): return datetime.now(timezone.utc).isoformat()


def _digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.absolute() != path.resolve():
        raise ValueError("regular nonsymlink canonical artifact path required")
    return path


def _json(path, sha):
    path = _regular(path)
    if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("explicit lowercase SHA256 required")
    if path.stat().st_size > MAX_JSON_BYTES: raise ValueError("JSON exceeds metadata bound")
    raw = path.read_bytes()
    if len(raw) > MAX_JSON_BYTES or hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("JSON size/identity mismatch")
    return json.loads(raw.decode("utf-8"), parse_constant=lambda s: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _json_exclusive(path, body):
    path = Path(path)
    raw = (json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > MAX_JSON_BYTES: raise ValueError("journal exceeds metadata bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".pending")
    with stage.open("xb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    os.link(stage, path)  # only complete canonical bytes; never overwrite
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def _finite(name, tensor):
    if not tensor.is_floating_point(): return
    for chunk in tensor.reshape(-1).split(1 << 20):
        value = chunk.float()
        if name == "text_attention_mask":
            if bool((torch.isnan(value) | (value > 0)).any()): raise ValueError("invalid native additive mask")
        elif not bool(torch.isfinite(value).all()): raise ValueError("nonfinite actual tensor: " + name)


def _tensors_exclusive(path, tensors):
    path = Path(path)
    if path.exists() or path.is_symlink(): raise ValueError("tensor output exists; no overwrite")
    for name, tensor in tensors.items(): _finite(name, tensor)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".pending")
    record = single.write_tensors(stage, tensors)
    with stage.open("rb") as handle: os.fsync(handle.fileno())
    os.link(stage, path)
    record.update(file=path.name, path=str(path.resolve()), file_bytes=path.stat().st_size)
    _verify_tensors(record, [path.parent.resolve()])
    return record


def _verify_tensors(record, roots):
    path = _regular(record["path"])
    if not any(path.is_relative_to(root) for root in roots): raise ValueError("artifact escapes verified capture roots")
    if not 0 < record["file_bytes"] <= 2 << 30: raise ValueError("tensor artifact exceeds bounded component file size")
    if path.stat().st_size != record["file_bytes"] or single.file_sha(path) != record["file_sha256"]:
        raise ValueError("tensor file identity/size changed")
    values = load_file(str(path), device="cpu")
    if set(values) != set(record["tensor_sha256"]) or set(values) != set(record["shapes"]) or set(values) != set(record["dtypes"]):
        raise ValueError("incomplete tensor identity metadata")
    for name, tensor in values.items():
        if (single.tensor_sha(tensor) != record["tensor_sha256"][name] or list(tensor.shape) != record["shapes"][name]
                or str(tensor.dtype) != record["dtypes"][name]): raise ValueError("tensor identity changed: " + name)
        _finite(name, tensor)
    return values


def parse_blocks(value):
    if not isinstance(value,str) or not 1<=len(value)<=512:raise ValueError("bounded explicit block list required")
    numbers = []
    for word in value.split(","):
        if re.fullmatch(r"\d+-\d+", word):
            start, stop = map(int, word.split("-"))
            if not 0<=start<=stop<60: raise ValueError("block range endpoints outside actual0..59 source")
            numbers.extend(range(start, stop + 1))
        elif re.fullmatch(r"\d+", word):
            number=int(word)
            if number>=60:raise ValueError("block index outside actual0..59 source")
            numbers.append(number)
        else: raise ValueError("blocks must be comma-separated indices/ranges")
        if len(numbers)>60:raise ValueError("block list exceeds bounded actual depth")
    return _blocks(numbers, 60)


def _blocks(blocks, depth):
    if (not isinstance(blocks, (list, tuple)) or not blocks or len(blocks) > 60 or
            any(type(b) is not int or not 0 <= b < depth for b in blocks) or list(blocks) != sorted(set(blocks))):
        raise ValueError("requested blocks must be nonempty sorted unique actual indices")
    return list(blocks)


def _common(dit, block, tap, x, sigma, prediction, ids, valid, fixture):
    records, meta = tap.finish()
    nt, ni = meta["text_tokens"], meta["image_tokens"]
    if ids.shape != (1, nt) or ids.dtype != torch.int64 or bool((ids < 0).any()): raise ValueError("actual encoder token IDs required")
    if valid.shape != (1, nt) or valid.dtype != torch.bool or not bool(valid.any()): raise ValueError("actual context validity required")
    h, w = x.shape[-2] // 2, x.shape[-1] // 2
    if ni != h*w or x.shape[-3] != 1: raise ValueError("complete plain latent patch grid required")
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    patch = torch.stack((torch.zeros(h*w, dtype=torch.int64), rows.flatten(), cols.flatten()), 1)
    if not fixture:
        image_ids = patch.float(); image_ids[:, 1] -= h//2; image_ids[:, 2] -= w//2
        start = max(h//2, w//2)
        text_ids = torch.arange(start, start+nt).float()[:, None].expand(-1, 3)
        if not torch.equal(records["source_ids"], torch.cat((text_ids, image_ids))[None]): raise ValueError("actual source grid identity differs")
    records.update(prediction=prediction, sigma=sigma.detach().clone().float().cpu(), model_input=x.detach().clone().cpu(),
                   image_patch_indices=patch, joint_row_kind=torch.tensor([0]*nt+[1]*ni, dtype=torch.int8),
                   context_token_ids=ids.detach().clone().cpu(), context_valid=valid.detach().clone().cpu())
    if "text_attention_mask" not in records:
        records["text_attention_mask"] = torch.zeros((1, nt), dtype=records["txt_in"].dtype)
    return records, {**meta, "block": block, "shared_source_forward": True}


@torch.no_grad()
def capture_stack_forward(dit, blocks, x, sigma, context, mask, token_ids, valid, *, emit, fixture=False):
    """Run ONE unchanged parent forward; stream complete actual block records.

    ``emit`` owns serialization. Synthetic tests explicitly use fixture=True;
    the production CLI has no fixture/model substitution switch.
    """
    blocks = _blocks(blocks, len(dit.transformer_blocks))
    if dit.training: raise ValueError("source capture requires eval mode")
    if sigma.shape != (1,) or sigma.dtype != torch.float32 or not bool(torch.isfinite(sigma).all()) or not 0 <= float(sigma[0]) <= 1:
        raise ValueError("one actual finite FP32 sigma in [0,1] required")
    if x.ndim != 5 or x.shape[:3] != (1,16,1) or min(x.shape[-2:]) < 2 or any(n%2 for n in x.shape[-2:]):
        raise ValueError("complete one-forward plain model input required")
    _finite("model_input", x); _finite("context", context)
    reports = {}
    with torch.inference_mode(), ExitStack() as stack:
        taps = {b: stack.enter_context(single.QimgBlockTap(dit, b, fixture=fixture)) for b in blocks}
        prediction = dit(x, sigma, context, attention_mask=mask, transformer_options={})
        if not isinstance(prediction, torch.Tensor): raise ValueError("parent prediction must be an actual tensor")
        prediction = prediction.detach().clone().cpu().contiguous()
        _finite("prediction", prediction)
        if not fixture and prediction.shape != x.shape: raise ValueError("complete parent prediction shape required")
        for b in blocks:
            records, meta = _common(dit,b,taps[b],x,sigma,prediction,token_ids,valid,fixture)
            reports[b] = emit(b, records, meta)
            taps[b].records.clear()
            del records
    return reports


def _implementation():
    return {Path(module.__file__).name: single.file_sha(Path(module.__file__)) for module in
            (sys.modules[__name__], single, sys.modules["c16_qimg_preflight"])}


def _stat(path):
    value = path.stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


def _coverage(request, completed, *, usable=False):
    expected = request["expected_forwards"]
    missing = [f for f in expected if f not in completed]
    requested = request["blocks"]
    return {"expected_forward_count":len(expected),"completed_shared_forward_count":len(completed),
            "missing_forwards":missing,"requested_blocks":requested,"missing_parent_blocks":[b for b in range(request["depth"]) if b not in requested],
            "requested_coverage_complete":not missing,"all_parent_blocks_requested":requested==list(range(request["depth"])),
            "full_parent_capture_complete":usable and not missing and requested==list(range(request["depth"]))}


def _load_resume(path, sha, request, *, fixture=False, depth=0, verified=None):
    if depth >= MAX_RESUMES: raise ValueError("resume ancestry exceeds bounded depth")
    path = _regular(path); body = _json(path,sha)
    if (body.get("schema") != SCHEMA or body.get("fixture") is not fixture or body.get("request") != request
            or body.get("request_sha256") != _digest(request) or body.get("status") not in {"partial","complete"}):
        raise ValueError("resume source/input/code/device request identity mismatch")
    roots = [path.parent.resolve()]
    if verified is None: verified=set()
    earlier = body.get("resume_from")
    prior = {}
    if earlier:
        prior, inherited, _ = _load_resume(earlier["path"],earlier["sha256"],request,fixture=fixture,depth=depth+1,verified=verified)
        roots.extend(inherited)
    def verify(record):
        key=_digest(record)
        if key not in verified:
            _verify_tensors(record,roots)
            verified.add(key)
    frozen = body.get("frozen_sources", {})
    if set(frozen) != set(map(str,request["blocks"])): raise ValueError("missing declared frozen source blocks")
    for record in frozen.values(): verify(record)
    verify(body["global_source"])
    coefficients=_digest({"blocks":{b:r["tensor_sha256"] for b,r in frozen.items()},"globals":body["global_source"]["tensor_sha256"]})
    if body.get("source_coefficients_sha256")!=coefficients or body.get("actual_runtime_sha256")!=_digest(body["runtime"]):
        raise ValueError("source coefficient/runtime identity mismatch")
    completed = {}
    pass_ids=set()
    for ref in body.get("completed_forwards",[]):
        key=ref["forward_id"]
        if key in completed or key not in request["expected_forwards"]: raise ValueError("duplicate/unknown completed forward")
        jp=_regular(ref["path"])
        if not any(jp.is_relative_to(r) for r in roots): raise ValueError("forward journal outside verified ancestry")
        journal=_json(jp,ref["sha256"])
        if (journal.get("schema")!=FORWARD_SCHEMA or journal.get("status")!="complete" or journal.get("fixture") is not fixture
                or journal.get("request_sha256")!=_digest(request) or journal.get("forward_id")!=key
                or journal.get("actual_runtime_sha256")!=body["actual_runtime_sha256"]
                or journal.get("source_coefficients_sha256")!=coefficients
                or set(journal.get("blocks",{}))!=set(map(str,request["blocks"]))): raise ValueError("incomplete/foreign shared source pass")
        pass_id=journal.get("source_pass_id","")
        if not isinstance(pass_id,str) or not re.fullmatch(r"[a-f0-9]{32}",pass_id) or pass_id in pass_ids:
            raise ValueError("source pass identity must be explicit and unique")
        pass_ids.add(pass_id)
        batch,sample,_=key.split("/")
        entry=next(e for e in request["inputs"][int(batch[5:])]["entries"] if e["id"]==sample)
        for b,capture in journal["blocks"].items():
            verify(capture["tensors"])
            meta=capture["meta"]
            if (meta.get("block")!=int(b) or meta.get("fixture") is not fixture or meta.get("complete_context") is not True
                    or meta.get("row_order")!="text_then_image" or meta.get("sample_id")!=sample
                    or meta.get("split")!=entry["split"] or meta.get("image_sha256")!=entry["image"]["sha256"]
                    or meta.get("prompt_sha256")!=entry["prompt_sha256"]
                    or capture["tensors"]["tensor_sha256"].get("prediction")!=journal["prediction_sha256"]):
                raise ValueError("block metadata/shared prediction identity mismatch")
        completed[key]=ref
    if any(key not in completed or completed[key]!=ref for key,ref in prior.items()): raise ValueError("resume dropped/replaced committed source forwards")
    if body.get("coverage")!=_coverage(request,completed,usable=body.get("functional_validation_complete") is True): raise ValueError("manifest coverage does not match actual journals")
    if body["status"]=="complete" and (body.get("functional_validation_complete") is not True or body.get("error")):
        raise ValueError("complete manifest lacks final functional validation")
    # Final block receipts are separately pinned views of the same source
    # journals. Intermediate snapshots intentionally do not have them yet.
    if "block_receipts" in body:
        if set(body["block_receipts"])!=set(frozen):raise ValueError("block receipt coverage differs")
        for b,ref in body["block_receipts"].items():
            receipt_path=_regular(ref["path"])
            if not any(receipt_path.is_relative_to(r) for r in roots):raise ValueError("block receipt outside verified ancestry")
            receipt=_json(receipt_path,ref["sha256"])
            if (receipt.get("schema")!=BLOCK_SCHEMA or receipt.get("fixture") is not fixture or receipt.get("block")!=int(b)
                    or receipt.get("request_sha256")!=_digest(request) or receipt.get("runtime")!=body["runtime"]
                    or receipt.get("actual_runtime_sha256")!=body["actual_runtime_sha256"]
                    or receipt.get("source_coefficients_sha256")!=coefficients or receipt.get("frozen_source")!=frozen[b]
                    or receipt.get("completed_forward_journals")!=body["completed_forwards"]
                    or receipt.get("missing_forwards")!=body["coverage"]["missing_forwards"]
                    or receipt.get("status")!=body["status"]):raise ValueError("block receipt differs from verified manifest")
    return completed,roots,body


def _snapshot(out, body, completed, name):
    value={**body,"completed_forwards":[completed[k] for k in body["request"]["expected_forwards"] if k in completed],
           "coverage":_coverage(body["request"],completed,usable=body.get("functional_validation_complete") is True),"recorded_at_utc":_now()}
    value["status"]="complete" if value["coverage"]["requested_coverage_complete"] and body.get("functional_validation_complete") is True and not body.get("error") else "partial"
    return _json_exclusive(out/name,value),value


def _freeze_sources(weights, dit, blocks, header, out):
    frozen={}; globals_=[n for n in header if not n.startswith("transformer_blocks.")]
    time_names=[n for n in globals_ if n.startswith("time_text_embed.")]
    if len(time_names)!=4 or len(globals_)!=13: raise ValueError("actual four time and nine remaining global source tensors required")
    runtime=dit.state_dict()
    def read(handle,names):
        values={n:handle.get_tensor(n) for n in names}
        for n,v in values.items():
            _finite(n,v)
            actual=runtime[n].detach().cpu().contiguous()
            if actual.dtype!=v.dtype or actual.shape!=v.shape or not torch.equal(actual.reshape(-1).view(torch.uint8),v.contiguous().reshape(-1).view(torch.uint8)):
                raise ValueError("loaded parent differs from stored source coefficient: "+n)
        return values
    with safe_open(str(weights),framework="pt",device="cpu") as handle:
        global_record=_tensors_exclusive(out/"source/global_source.safetensors",read(handle,globals_))
        for b in blocks:
            names=[n for n in header if n.startswith(f"transformer_blocks.{b}.")]+time_names
            frozen[str(b)]=_tensors_exclusive(out/f"source/block_{b:02d}/source_block_and_time.safetensors",read(handle,names))
    return frozen,global_record


def capture(*, weights, inputs, out, blocks, comfy_root, device="cuda:0", max_tokens=2048,max_forwards=32,
            resume_manifest=None,resume_sha256=None,validate_only=False):
    """Production entry: no fixture or alternate parent factory accepted."""
    weights, out, comfy_root=Path(weights),Path(out),Path(comfy_root)
    inputs=[Path(p) for p in inputs];blocks=_blocks(blocks,60)
    if not 1<=len(inputs)<=MAX_BATCHES or len(set(p.resolve() for p in inputs))!=len(inputs): raise ValueError("one to four unique encoded input manifests required")
    if out.exists() or out.is_symlink(): raise ValueError("new output directory required, including resume")
    if device not in {"cpu","cuda","cuda:0"}: raise ValueError("unsupported source device")
    destination=out.resolve()
    protected=[weights.resolve(),comfy_root.resolve(),Path(__file__).resolve().parent,*[p.resolve().parent for p in inputs]]
    if any(destination==p or destination.is_relative_to(p) or p.is_relative_to(destination) for p in protected): raise ValueError("capture output overlaps source or actual input directory")
    source_stat=_stat(weights)
    header,identity=single.read_header(weights)
    if identity["header_sha256"]!=single.PARENT_HEADER_SHA256: raise ValueError("wrong actual QIMG FP8 source header")
    preflight={str(b):single.inspect_weights(weights,b) for b in blocks}
    if any(p["depth"]!=60 or p["source_identity"]!=identity for p in preflight.values()): raise ValueError("actual contiguous60 source geometry required")
    batches=[];expected=[];estimate=0;image_splits={}
    for bi,path in enumerate(inputs):
        items,evidence=single.load_inputs(path,max_tokens=max_tokens,max_forwards=max_forwards)
        for item in items:
            image=item["entry"]["image"]["sha256"];split=item["entry"]["split"]
            if image in image_splits and image_splits[image]!=split:raise ValueError("one image leaks across input-batch fit/holdout roles")
            image_splits[image]=split
        batches.append((items,evidence));estimate+=sum(single.output_size_estimate(p,items) for p in preflight.values())
        for item in items:
            expected.extend(f"batch{bi:02d}/{item['entry']['id']}/sigma{j:02d}" for j in range(item["forwards"]))
    source_code=single.verify_source_code(comfy_root)
    request={"blocks":blocks,"depth":60,"source":identity,"source_stat":source_stat,"source_code_sha256":source_code,
             "implementation_sha256":_implementation(),"inputs":[e for _,e in batches],"expected_forwards":expected,
             "device":device,"max_tokens":max_tokens,"max_forwards_per_batch":max_forwards}
    completed={};old=None;resume=None
    if bool(resume_manifest)!=bool(resume_sha256): raise ValueError("resume manifest and exact SHA must both be supplied")
    if resume_manifest:
        resume={"path":str(_regular(resume_manifest)),"sha256":resume_sha256}
        completed,_,old=_load_resume(resume["path"],resume["sha256"],request)
        if destination==Path(resume["path"]).parent or destination.is_relative_to(Path(resume["path"]).parent): raise ValueError("resume must use a separate new output root")
    missing=[f for f in expected if f not in completed]
    if validate_only:
        return {"validated":True,"capture_launched":False,"request":request,"request_sha256":_digest(request),"coverage":_coverage(request,completed),"estimated_output_upper_bound_bytes":estimate}
    probe=out.parent.resolve()
    while not probe.exists():probe=probe.parent
    free=shutil.disk_usage(probe).free
    if missing and free<estimate+(4<<30): raise ValueError("bounded capture estimate leaves less than4GiB; no cleanup")
    out.mkdir(parents=True,exist_ok=False)
    body={"schema":SCHEMA,"fixture":False,"request":request,"request_sha256":_digest(request),"resume_from":resume,
          "status":"partial","runtime":None,"frozen_sources":{},"global_source":None,"capability_accepted":False,"functional_validation_complete":False,
          "folded":False,"installed":False,"numerical_target_policy":"diagnostic only; no fidelity target controls capture",
          "estimated_output_upper_bound_bytes":estimate,"free_bytes_before":free,"started_at_utc":_now(),
          "artifact_identity_scope":"full stored tensor/file hashes for emitted captures/source subsets; original parent full-file digest not newly established"}
    start=time.monotonic();patcher=None;sequence=0
    try:
        if missing:
            saved=sys.argv
            try:
                sys.argv=[sys.argv[0]]
                patcher,dit,dtype,runtime=single.load_parent(weights,comfy_root,device)
            finally:sys.argv=saved
            if len(dit.transformer_blocks)!=request["depth"]:raise ValueError("loaded native parent depth differs from actual source header")
            body["runtime"]=runtime
            body["frozen_sources"],body["global_source"]=_freeze_sources(weights,dit,blocks,header,out)
            runtime["selected_block_and_time_match_source_bytes"]=True
            if old and runtime!=old["runtime"]:raise ValueError("actual runtime changed across resume")
            body["actual_runtime_sha256"]=_digest(runtime)
            body["source_coefficients_sha256"]=_digest({"blocks":{b:r["tensor_sha256"] for b,r in body["frozen_sources"].items()},"globals":body["global_source"]["tensor_sha256"]})
            # Compare raw output tensor identities, not new serialized-file headers.
            if old and (any(body["frozen_sources"][b]["tensor_sha256"]!=old["frozen_sources"][b]["tensor_sha256"] for b in body["frozen_sources"])
                        or body["global_source"]["tensor_sha256"]!=old["global_source"]["tensor_sha256"]):raise ValueError("actual parent coefficients changed across resume")
            _snapshot(out,body,completed,"snapshot_000000.json")
            for bi,(items,evidence) in enumerate(batches):
                for item in items:
                    if single.file_sha(item["tensor_path"])!=item["entry"]["tensors"]["sha256"]:raise ValueError("actual input changed after validation")
                    state=load_file(str(item["tensor_path"]),device="cpu")
                    single.validate_inputs(state,max_tokens=max_tokens)
                    for j in range(item["forwards"]):
                        key=f"batch{bi:02d}/{item['entry']['id']}/sigma{j:02d}"
                        if key in completed:continue
                        x=state["model_inputs"][j:j+1].to(device,dtype);sigma=state["sigma"][j:j+1].to(device)
                        context=state["context"].to(device,dtype);mask=state["context_mask"].to(device,torch.int64)
                        pass_id=uuid.uuid4().hex
                        def emit(b,tensors,meta):
                            if tensors["img_in"].shape[-1]!=preflight[str(b)]["width"] or tensors["rotation_used"].shape[-3]!=preflight[str(b)]["head_dim"]//2:
                                raise ValueError("actual captured width/head geometry differs from source header")
                            path=out/f"block_{b:02d}/batch{bi:02d}/{item['entry']['id']}_sigma{j:02d}.safetensors"
                            record=_tensors_exclusive(path,tensors)
                            return {"tensors":record,"meta":{**meta,"sample_id":item["entry"]["id"],"split":item["entry"]["split"],
                                    "image_sha256":item["entry"]["image"]["sha256"],"prompt_sha256":item["entry"]["prompt_sha256"],"sigma":float(sigma[0]),
                                    "model_input_cast_max_abs":float((x.float().cpu()-state["model_inputs"][j:j+1].float()).abs().max()),
                                    "context_cast_max_abs":float((context.float().cpu()-state["context"].float()).abs().max())}}
                        captured=capture_stack_forward(dit,blocks,x,sigma,context,mask,state["context_token_ids"],state["context_mask"],emit=emit)
                        prediction_sha=captured[blocks[0]]["tensors"]["tensor_sha256"]["prediction"]
                        if any(c["tensors"]["tensor_sha256"]["prediction"]!=prediction_sha for c in captured.values()):raise ValueError("shared prediction changed between block writes")
                        journal={"schema":FORWARD_SCHEMA,"status":"complete","fixture":False,"forward_id":key,"source_pass_id":pass_id,
                                 "request_sha256":_digest(request),"prediction_sha256":prediction_sha,"blocks":{str(b):c for b,c in captured.items()},
                                 "actual_runtime_sha256":body["actual_runtime_sha256"],"source_coefficients_sha256":body["source_coefficients_sha256"],
                                 "completed_at_utc":_now(),"scope":"one authentic complete parent forward shared by every declared requested block"}
                        ref=_json_exclusive(out/f"forwards/forward_{expected.index(key):06d}.json",journal);ref["forward_id"]=key
                        completed[key]=ref;sequence+=1
                        _snapshot(out,body,completed,f"snapshot_{sequence:06d}.json")
                        print(json.dumps({"captured_shared_forward":key,"blocks":blocks,"completed":len(completed),"expected":len(expected)}),flush=True)
            if _stat(weights)!=source_stat or single.read_header(weights)[1]!=identity:raise ValueError("source parent changed during capture")
            # The bounded input verifier covers upstream image/encoder headers
            # as well as actual encoded tensor bytes, not only JSON manifests.
            if any(single.load_inputs(Path(e["manifest"]),max_tokens=max_tokens,max_forwards=max_forwards)[1]!=e for _,e in batches):
                raise ValueError("actual encoded input/source identity changed during capture")
            if single.verify_source_code(comfy_root)!=source_code or _implementation()!=request["implementation_sha256"]:
                raise ValueError("actual source/helper implementation changed during capture")
            body["source_header_stat_stable_after_capture"]=True
            body["input_source_code_implementation_stable_after_capture"]=True
            body["functional_validation_complete"]=True
        else:
            body.update(runtime=old["runtime"],frozen_sources=old["frozen_sources"],global_source=old["global_source"],
                        actual_runtime_sha256=old["actual_runtime_sha256"],source_coefficients_sha256=old["source_coefficients_sha256"],
                         source_header_stat_stable_after_capture=True,input_source_code_implementation_stable_after_capture=True,
                         parent_loaded_this_attempt=False,functional_validation_complete=True)
        if missing:body["parent_loaded_this_attempt"]=True
    except BaseException as exc:
        body["error"]={"type":type(exc).__name__,"message":str(exc)}
        body["functional_validation_complete"]=False
        body["incomplete_pass_artifacts_preserved"]=True
        raise
    finally:
        body["seconds"]=time.monotonic()-start
        try:
            import resource
            body["peak_cpu_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        except ImportError:body["peak_cpu_rss_bytes"]=None
        if device.startswith("cuda") and torch.cuda.is_initialized():
            body["peak_vram_allocated_bytes"]=torch.cuda.max_memory_allocated(device)
            body["peak_vram_reserved_bytes"]=torch.cuda.max_memory_reserved(device)
        if body["frozen_sources"] and body["global_source"]:
            block_refs={}
            for b in blocks:
                partition={"schema":BLOCK_SCHEMA,"fixture":False,"block":b,"source":preflight[str(b)],"frozen_source":body["frozen_sources"][str(b)],
                            "runtime":body["runtime"],"actual_runtime_sha256":body.get("actual_runtime_sha256"),
                            "source_coefficients_sha256":body.get("source_coefficients_sha256"),
                            "request_sha256":_digest(request),"completed_forward_journals":[completed[k] for k in expected if k in completed],
                           "missing_forwards":[k for k in expected if k not in completed],"status":"complete" if len(completed)==len(expected) and body["functional_validation_complete"] and "error" not in body else "partial",
                           "scope":"new stack block receipt; not a legacy single-block capture schema or fold/capability proof"}
                block_refs[str(b)]=_json_exclusive(out/f"block_{b:02d}/stack_block_capture_receipt.json",partition)
            body["block_receipts"]=block_refs
        body["finished_at_utc"]=_now()
        ref,final=_snapshot(out,body,completed,"stack_capture_manifest.json")
        print(json.dumps({"manifest":ref,"status":final["status"],"coverage":final["coverage"],"error":final.get("error")}),flush=True)
        del patcher
    return ref


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights",type=Path,required=True);parser.add_argument("--inputs",nargs="+",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True);parser.add_argument("--blocks",required=True)
    parser.add_argument("--comfy-root",type=Path,default=Path("/opt/ComfyUI"));parser.add_argument("--device",choices=("cpu","cuda","cuda:0"),default="cuda:0")
    parser.add_argument("--max-tokens",type=int,default=2048);parser.add_argument("--max-forwards",type=int,default=32)
    parser.add_argument("--resume-manifest",type=Path);parser.add_argument("--resume-sha256");parser.add_argument("--validate-only",action="store_true")
    args=vars(parser.parse_args());args["blocks"]=parse_blocks(args["blocks"])
    result=capture(**args)
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":main()
