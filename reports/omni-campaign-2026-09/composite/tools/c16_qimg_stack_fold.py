#!/usr/bin/env python3
"""Fold authentic shared-forward capture V4, retaining finite diagnostics.

No teacher forward is executed here. Each block is fitted exactly once through
the corrected, source-pinned development fitter. New exclusive per-block files
are reloaded by the actual joint reader before an immutable progress marker is
published. A complete stack is 60*32 block tensors + shared4 + sourceglobal9.
This engineering artifact grants no numerical or behavioral acceptance.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import uuid

import torch
import c16_qimg_stack_capture as capture
import c16_qimg_capture as single
import c16_qimg_fold as original
import c16_qimg_development_refold as development
import c16_qimg_joint_replay as joint

SCHEMA = "neural_forge.qimg_folded_stack.v1"
DEPTH = 60
MAX_METADATA_BYTES = 256 << 20
TIME_PREFIX = "time_text_embed."
GLOBAL_NAMES = {"img_in.bias", "img_in.weight", "norm_out.linear.bias", "norm_out.linear.weight",
                "proj_out.bias", "proj_out.weight", "txt_in.bias", "txt_in.weight", "txt_norm.weight"}
TENSOR_HASH_SCHEMA = "torch-dtype-name + compact JSON shape + raw contiguous bytes; v1"


def _reference(value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("exact path/SHA256 artifact reference required")
    return capture._regular(value["path"]), original.digest(value["sha256"])


class CaptureIndex:
    """Bounded metadata-only capture session; no all-block activation rescan.

    All completed shared-forward journals and immutable ancestry are bound once.
    A fitter then fully verifies only its block's coefficients/activation files.
    A native reader deliberately uses coefficients + bound metadata, not the
    entire activation expansion again. This distinction is explicit in data.
    """
    def __init__(self, manifest_path, expected_sha256, *, fixture=False):
        if type(fixture) is not bool: raise ValueError("explicit fixture boolean required")
        self.path = capture._regular(manifest_path)
        self.sha256 = original.digest(expected_sha256)
        self.fixture = fixture
        self._metadata = {}; self._metadata_bytes = 0
        self.body = self._read(self.path, self.sha256)
        self.request = self.body.get("request", {})
        blocks = self.request.get("blocks")
        capture._blocks(blocks, DEPTH)
        expected = self.request.get("expected_forwards")
        if (self.request.get("depth") != DEPTH or not isinstance(expected, list)
                or not 1 <= len(expected) <= 128 or len(set(expected)) != len(expected)
                or not isinstance(self.request.get("inputs"), list) or not 1 <= len(self.request["inputs"]) <= 4):
            raise ValueError("bounded actual source depth/input/forward identities required")
        if not fixture:
            runtime = self.body.get("runtime", {})
            if self.request.get("implementation_sha256")!=capture._implementation():
                raise ValueError("production capture requires authoritative sealed V4 generator/helper pins")
            if (self.request.get("source", {}).get("header_sha256") != single.PARENT_HEADER_SHA256
                    or self.request.get("source_code_sha256") != single.SOURCE_FILES
                    or runtime.get("manual_cast_dtype") != "torch.bfloat16"
                    or runtime.get("fp8_optimizations") is not False
                    or runtime.get("selected_block_and_time_match_source_bytes") is not True
                    or runtime.get("full_residency_verified") is not True
                    or runtime.get("dynamic_patcher") is not False
                    or runtime.get("weight_dtype_requested") != "torch.float8_e4m3fn"
                    or runtime.get("parent_class") != "comfy.model_base.QwenImage"
                    or self.body.get("source_header_stat_stable_after_capture") is not True
                    or self.body.get("input_source_code_implementation_stable_after_capture") is not True):
                raise ValueError("actual pinned BF16-manual-cast FP8 source/runtime required")
        self.roots = []
        self.journals = self._ancestor(self.path, self.sha256, 0)
        if (self.body.get("status") != "complete" or self.body.get("functional_validation_complete") is not True
                or self.body.get("error") or set(self.journals) != set(expected)):
            raise ValueError("only functionally complete requested capture coverage can fold")
        self._globals = None

    def _read(self, path, sha):
        path = capture._regular(path); key = (str(path), sha)
        if key in self._metadata: return self._metadata[key]
        size = path.stat().st_size
        self._metadata_bytes += size
        if self._metadata_bytes > MAX_METADATA_BYTES: raise ValueError("aggregate metadata bound exceeded")
        body = capture._json(path, sha)
        self._metadata[key] = body
        return body

    def _ancestor(self, path, sha, depth):
        if depth >= capture.MAX_RESUMES: raise ValueError("capture resume ancestry exceeds bound")
        body = self._read(path, sha); root = Path(path).resolve().parent
        if root not in self.roots: self.roots.append(root)
        if (body.get("schema") != capture.SCHEMA or body.get("fixture") is not self.fixture
                or body.get("request") != self.request or body.get("request_sha256") != capture._digest(self.request)
                or body.get("status") not in {"partial", "complete"}):
            raise ValueError("capture ancestry source/input/code/request identity differs")
        prior = {}
        if body.get("resume_from"):
            earlier, earlier_sha = _reference(body["resume_from"])
            prior = self._ancestor(earlier, earlier_sha, depth + 1)
        frozen = body.get("frozen_sources", {})
        if set(frozen) != set(map(str, self.request["blocks"])): raise ValueError("source block coverage differs")
        coefficients = capture._digest({"blocks": {b: r["tensor_sha256"] for b, r in frozen.items()},
                                        "globals": body["global_source"]["tensor_sha256"]})
        if (body.get("source_coefficients_sha256") != coefficients
                or body.get("actual_runtime_sha256") != capture._digest(body["runtime"])
                or coefficients != self.body.get("source_coefficients_sha256")
                or body["runtime"] != self.body.get("runtime")):
            raise ValueError("source coefficient/runtime digest differs")
        completed = {}; pass_ids = set()
        for ref in body.get("completed_forwards", []):
            if not isinstance(ref, dict) or set(ref) != {"path", "sha256", "forward_id"}:
                raise ValueError("explicit shared-forward reference required")
            key = ref["forward_id"]
            if key in completed or key not in self.request["expected_forwards"]: raise ValueError("duplicate/unknown forward")
            jp = capture._regular(ref["path"])
            if not any(jp.is_relative_to(r) for r in self.roots): raise ValueError("journal outside capture ancestry")
            journal = self._read(jp, ref["sha256"])
            if (journal.get("schema") != capture.FORWARD_SCHEMA or journal.get("status") != "complete"
                    or journal.get("fixture") is not self.fixture or journal.get("forward_id") != key
                    or journal.get("request_sha256") != capture._digest(self.request)
                    or journal.get("actual_runtime_sha256") != body["actual_runtime_sha256"]
                    or journal.get("source_coefficients_sha256") != coefficients
                    or set(journal.get("blocks", {})) != set(frozen)):
                raise ValueError("incomplete/foreign shared forward")
            pass_id = journal.get("source_pass_id", "")
            if not isinstance(pass_id, str) or not re.fullmatch("[a-f0-9]{32}", pass_id) or pass_id in pass_ids:
                raise ValueError("distinct actual source pass identity required")
            pass_ids.add(pass_id)
            parts = key.split("/")
            if len(parts) != 3 or not re.fullmatch(r"batch\d{2}", parts[0]) or not re.fullmatch(r"sigma\d{2}", parts[2]):
                raise ValueError("source forward identity format differs")
            batch = int(parts[0][5:]); sample = parts[1]
            if batch >= len(self.request["inputs"]): raise ValueError("unknown source batch")
            entries = self.request["inputs"][batch]["entries"]
            hits = [e for e in entries if e["id"] == sample]
            if len(hits) != 1: raise ValueError("input entry identity must be unique")
            entry = hits[0]
            shared=None;previous=None
            shared_keys=("prediction","model_input","sigma","context_valid","context_token_ids","source_ids",
                         "rotation_generated","rotation_used","temb","image_patch_indices","joint_row_kind","text_attention_mask")
            for block in self.request["blocks"]:
                b=str(block);node=journal["blocks"][b]
                meta = node["meta"]; record = node["tensors"]
                if (meta.get("block") != int(b) or meta.get("fixture") is not self.fixture
                        or meta.get("complete_context") is not True or meta.get("row_order") != "text_then_image"
                        or meta.get("sample_id") != sample or meta.get("split") != entry["split"]
                        or meta.get("image_sha256") != entry["image"]["sha256"]
                        or meta.get("prompt_sha256") != entry["prompt_sha256"]
                        or record["tensor_sha256"].get("prediction") != journal.get("prediction_sha256")):
                    raise ValueError("block input/split/shared prediction metadata differs")
                invariant=({k:original.digest(record["tensor_sha256"][k])for k in shared_keys},
                           {k:meta[k]for k in ("sigma","image_tokens","text_tokens")})
                if shared is None:shared=invariant
                elif invariant!=shared:raise ValueError("shared source-forward invariant tensors/geometry differ across blocks")
                if previous is not None and block==previous[0]+1:
                    for stream in original.STREAMS:
                        if previous[1][stream+"_out"]!=record["tensor_sha256"][stream+"_in"]:
                            raise ValueError("adjacent actual source block stream continuity differs")
                previous=(block,record["tensor_sha256"])
            completed[key] = {"ref": ref, "body": journal}
        if any(k not in completed or completed[k]["ref"] != v["ref"] for k, v in prior.items()):
            raise ValueError("capture resume dropped/replaced completed forwards")
        refs = {k: v["ref"] for k, v in completed.items()}
        if body.get("coverage") != capture._coverage(self.request, refs, usable=body.get("functional_validation_complete") is True):
            raise ValueError("capture coverage differs from actual shared journals")
        if body.get("status") == "complete" and (body.get("functional_validation_complete") is not True or body.get("error")):
            raise ValueError("complete capture lacks final functional validation")
        if "block_receipts" in body:
            if set(body["block_receipts"]) != set(frozen): raise ValueError("capture block receipt coverage differs")
            for b, ref in body["block_receipts"].items():
                rp, rs = _reference(ref)
                if not any(rp.is_relative_to(r) for r in self.roots): raise ValueError("block receipt outside ancestry")
                receipt = self._read(rp, rs)
                checks = {"schema": capture.BLOCK_SCHEMA, "fixture": self.fixture, "block": int(b),
                    "request_sha256": capture._digest(self.request), "runtime": body["runtime"],
                    "actual_runtime_sha256": body["actual_runtime_sha256"], "source_coefficients_sha256": coefficients,
                    "frozen_source": frozen[b], "completed_forward_journals": body["completed_forwards"],
                    "missing_forwards": body["coverage"]["missing_forwards"], "status": body["status"]}
                if any(receipt.get(k) != v for k, v in checks.items()): raise ValueError("block receipt is not the declared journal view")
        return completed

    def global_state(self):
        if self._globals is None:
            state = capture._verify_tensors(self.body["global_source"], self.roots)
            time = {k for k in state if k.startswith(TIME_PREFIX)}
            if len(time) != 4 or set(state) != GLOBAL_NAMES | time or any(v.dtype != torch.float8_e4m3fn for v in state.values()):
                raise ValueError("exact 9 source globals + shared4 FP8 timestep tensors required")
            width = state["img_in.bias"].numel(); context = state["txt_norm.weight"].numel()
            shapes = {"img_in.weight": (width, 64), "img_in.bias": (width,), "txt_in.weight": (width, context),
                "txt_in.bias": (width,), "txt_norm.weight": (context,), "norm_out.linear.weight": (2*width, width),
                "norm_out.linear.bias": (2*width,), "proj_out.weight": (64,width), "proj_out.bias": (64,),
                "time_text_embed.timestep_embedder.linear_1.weight": (width,256),
                "time_text_embed.timestep_embedder.linear_1.bias": (width,),
                "time_text_embed.timestep_embedder.linear_2.weight": (width,width),
                "time_text_embed.timestep_embedder.linear_2.bias": (width,)}
            if ({k: tuple(v.shape) for k,v in state.items()} != shapes
                    or (not self.fixture and (width,context) != (3072,3584))):
                raise ValueError("complete original source global geometry required")
            self._globals = state
        return self._globals

    def revalidate(self):
        if not self.fixture and self.request.get("implementation_sha256")!=capture._implementation():
            raise ValueError("authoritative capture generator/helper source changed during operation")
        for (path, sha), body in self._metadata.items():
            if capture._json(path, sha) != body: raise ValueError("capture metadata changed during operation")
        return True

    def revalidate_block_payloads(self,block):
        """Rehash consumed mappings before publication; metadata alone is insufficient."""
        records=[self.body["frozen_sources"][str(block)],self.body["global_source"]]
        records.extend(self.journals[key]["body"]["blocks"][str(block)]["tensors"]for key in self.request["expected_forwards"])
        for record in records:
            path=capture._regular(record["path"])
            if (not any(path.is_relative_to(r)for r in self.roots)or path.stat().st_size!=record["file_bytes"]
                    or single.file_sha(path)!=record["file_sha256"]):
                raise ValueError("consumed source/capture payload changed before component publication")
        return True


def load_block_inputs(manifest_path, expected_sha256, block, *, fixture=False, index=None, load_forwards=True):
    if type(block) is not int or not 0 <= block < DEPTH or type(load_forwards) is not bool:
        raise ValueError("actual block index/activation loading boolean required")
    index = index or CaptureIndex(manifest_path, expected_sha256, fixture=fixture)
    if (index.path != Path(manifest_path).resolve() or index.sha256 != expected_sha256 or index.fixture is not fixture
            or block not in index.request["blocks"]): raise ValueError("capture session/block identity differs")
    record = index.body["frozen_sources"][str(block)]
    frozen = capture._verify_tensors(record, index.roots)
    prefix = f"transformer_blocks.{block}."
    width = frozen[prefix+"img_mlp.net.0.proj.weight"].shape[1]
    inner = frozen[prefix+"img_mlp.net.0.proj.weight"].shape[0]
    head = frozen[prefix+"attn.norm_q.weight"].numel()
    if ({k: tuple(v.shape) for k,v in frozen.items()} != original.frozen_shapes(block,width,inner,head)
            or any(v.dtype != torch.float8_e4m3fn for v in frozen.values())
            or (not fixture and (width,head,inner) != (3072,128,12288))):
        raise ValueError("complete source-pinned dual-stream FP8 block/time geometry required")
    globals_ = index.global_state()
    if any(not joint.same_bytes(v,globals_[k]) for k,v in frozen.items() if k.startswith(TIME_PREFIX)):
        raise ValueError("per-block shared timestep bytes differ from actual source globals")
    forwards = []; counts = {s:{"fit":0,"holdout":0} for s in original.STREAMS}; splits = {}; seen = set()
    for key in index.request["expected_forwards"]:
        node = index.journals[key]["body"]["blocks"][str(block)]; meta = node["meta"]; capture_record = node["tensors"]
        image = original.digest(meta["image_sha256"]); split = meta["split"]
        if split not in {"fit","holdout"} or (image in splits and splits[image] != split): raise ValueError("image split leakage")
        splits[image] = split
        sha = original.digest(capture_record["file_sha256"])
        if sha in seen: raise ValueError("duplicate captured payload cannot count twice")
        seen.add(sha)
        ni,nt = meta["image_tokens"],meta["text_tokens"]
        if any(type(n) is not int or n < 1 for n in (ni,nt)) or ni+nt > index.request.get("max_tokens",2048):
            raise ValueError("bounded complete joint context required")
        if not load_forwards: continue
        state = capture._verify_tensors(capture_record,index.roots)
        if set(state) != original.CAPTURE_KEYS: raise ValueError("complete original joint-context capture keys required")
        sigma = state["sigma"]; valid = state["context_valid"]
        if (sigma.shape != (1,) or sigma.dtype != torch.float32 or not 0 <= float(sigma[0]) <= 1
                or float(sigma[0]) != meta["sigma"] or valid.dtype != torch.bool or valid.shape != (1,nt) or not bool(valid.any())):
            raise ValueError("actual sigma/text-validity identity differs")
        if (state["source_ids"].shape != (1,ni+nt,3) or state["rotation_used"].shape != (1,1,ni+nt,head//2,2,2)
                or not torch.equal(state["rotation_generated"].to(state["rotation_used"].dtype),state["rotation_used"])):
            raise ValueError("complete source coordinate/rotation boundaries differ")
        x = state["model_input"]
        if (x.ndim != 5 or tuple(x.shape[:3]) != (1,16,1) or any(n%2 for n in x.shape[-2:])
                or x.shape[-2]//2*(x.shape[-1]//2) != ni or state["prediction"].shape != x.shape
                or state["image_patch_indices"].shape != (ni,3)
                or state["context_token_ids"].shape != (1,nt) or state["context_token_ids"].dtype != torch.int64):
            raise ValueError("complete latent/context/prediction geometry differs")
        rows,cols=torch.meshgrid(torch.arange(x.shape[-2]//2),torch.arange(x.shape[-1]//2),indexing="ij")
        patch=torch.stack((torch.zeros(ni,dtype=torch.int64),rows.flatten(),cols.flatten()),1)
        if (not torch.equal(patch,state["image_patch_indices"])
                or not torch.equal(state["joint_row_kind"],torch.tensor([0]*nt+[1]*ni,dtype=torch.int8))):
            raise ValueError("actual text-then-image row/grid ordering differs")
        item = {"file":capture_record["path"],"file_sha256":sha,"image_sha256":image,
                "sample_id":meta["sample_id"],"split":split,"sigma":meta["sigma"]}
        for stream,tokens in (("img",ni),("txt",nt)):
            a,b=state[stream+"_mlp_in"],state[stream+"_mlp_out"]
            if a.shape != (1,tokens,width) or b.shape != a.shape or a.dtype != torch.bfloat16 or b.dtype != a.dtype:
                raise ValueError("complete BF16 actual source MLP rows required")
            mask=valid.flatten() if stream=="txt" else torch.ones(tokens,dtype=torch.bool)
            item[stream]={"x":a,"y":b,"valid":mask};counts[stream][split]+=int(mask.sum())
            if sum(counts[stream].values()) > original.MAX_ROWS_PER_STREAM: raise ValueError("bounded fitting row memory exceeded")
        forwards.append(item)
    if set(splits.values()) != {"fit","holdout"}: raise ValueError("image-disjoint fit AND holdout source captures required")
    return {"frozen":frozen,"frozen_file_sha256":record["file_sha256"],"block":block,"width":width,"inner":inner,
        "forwards":forwards,"row_counts":counts,"image_splits":splits,"fixture":fixture,
        "capture_receipts":[{"path":str(index.path),"sha256":index.sha256}],
        "stack_capture_adapter":"explicit V4 metadata and selected-block payload reader; no legacy schema synthesized",
        "capture_forward_payloads_verified":load_forwards}


def _rescore(data,candidate,policy,device):
    measured={}
    for stream in original.STREAMS:
        weights=original.mlp_weights(candidate,data["block"],stream,device)
        per=[];pools={s:[[],[]] for s in ("fit","holdout")}
        for item in data["forwards"]:
            rows=item[stream];_,pred=development.mlp_forward(rows["x"],weights)
            pred=pred.squeeze(0).cpu()[rows["valid"]];target=rows["y"].squeeze(0)[rows["valid"]]
            scores=development.score(pred,target)
            per.append({"file":item["file"],"image_sha256":item["image_sha256"],"split":item["split"],"sigma":item["sigma"],"scores":scores})
            pools[item["split"]][0].append(pred);pools[item["split"]][1].append(target)
        fit,held=(development.score(torch.cat(pools[s][0]),torch.cat(pools[s][1])) for s in ("fit","holdout"))
        passed=original.numerical_pass(held,policy["streams"][stream]) and all(original.numerical_pass(r["scores"],policy["streams"][stream]) for r in per if r["split"]=="holdout")
        measured[stream]={"fit_scores":fit,"heldout_scores":held,"per_forward":per,"numerical_policy_passed":bool(passed)}
    return original.json_diagnostics(measured)


def _persist_block(data,policy,out,index,policy_ref,device):
    out.mkdir(parents=True,exist_ok=False)
    index.revalidate()
    candidate,selection,result=development._fit_loaded_candidate(data,policy,device=device)
    result.update(schema=original.OWNER_FOLD_SCHEMA,status="measured_candidate",recipe_id=development.RECIPE_ID,
        numerical_policy_passed=result.pop("development_numeric_gates_passed"),candidate_written=True,selection_tensors_written=True,
        stack_capture={"path":str(index.path),"sha256":index.sha256},policy_reference=policy_ref,
        artifact_admitted=False,accepted=False,acceptance_enabled=False,capability_accepted=False,
        acceptance_scope="structurally verified unevaluated stack development component; no fresh/legacy capture admission implied")
    for values in result["streams"].values(): values["numerical_policy_passed"]=values.pop("development_numeric_gates_passed")
    result["output"]=capture._tensors_exclusive(out/"folded_source_block_and_time.safetensors",candidate)
    result["selection"]=capture._tensors_exclusive(out/"selection.safetensors",selection)
    provisional=capture._json_exclusive(out/"provisional_fold_receipt.json",original.json_diagnostics(result))
    loaded,identity=joint.load_candidate(Path(provisional["path"]),provisional["sha256"],data)
    selected,_=original.verified_tensors(out,result["selection"],1<<20)
    if (set(loaded)!=set(candidate) or any(not joint.same_bytes(loaded[k],v) for k,v in candidate.items())
            or set(selected)!=set(selection) or any(not joint.same_bytes(selected[k],v) for k,v in selection.items())):
        raise ValueError("actual fitted candidate/selection readback bytes differ")
    scores=_rescore(data,loaded,policy,device)
    for stream,values in scores.items(): result["streams"][stream].update(values)
    result.update(numerical_policy_passed=all(v["numerical_policy_passed"] for v in scores.values()),
        persisted_numeric_gates_passed=all(v["numerical_policy_passed"] for v in scores.values()),
        persisted_tensors_rescored_without_refit=True,fit_or_selector_retry=False,
        serialization_validation="exact source/fold/selection stored bytes reloaded by actual joint reader",
        structural_artifact_valid=True,structural_identity=identity,
        full_joint_block_replay_performed=False,child_accepted=False,media_installed=False,
        publication_protocol="canonical exclusive fold_receipt.json only; pending/provisional is not a publication marker")
    result["diagnostic_measurements_complete"]=original.diagnostics_complete(result)
    result["all_numeric_diagnostic_targets_met"]=original.all_numeric_targets_met(result,"numerical_policy_passed")
    index.revalidate()
    for record in (result["output"],result["selection"]):
        capture._verify_tensors(record,[out.resolve()])
    index.revalidate_block_payloads(data["block"])
    return capture._json_exclusive(out/"fold_receipt.json",original.json_diagnostics(result))


def _source_header(path):
    path=capture._regular(path)
    with path.open("rb") as handle:
        size=struct.unpack("<Q",handle.read(8))[0]
        if not 0<size<=8<<20: raise ValueError("bounded actual safetensors header required")
        raw=handle.read(size)
    header=json.loads(raw)
    return {k:v for k,v in header.items() if k!="__metadata__"},size+8


def _validate_component(ref,data,policy,policy_ref):
    """Publication + exact requested recipe, distinct from numeric diagnostics."""
    path,sha=_reference(ref)
    if path.name!="fold_receipt.json":raise ValueError("only canonical completed fold receipt can progress")
    body=capture._json(path,sha)
    required_true=("structural_artifact_valid","candidate_written","selection_tensors_written","persisted_tensors_rescored_without_refit")
    required_false=("accepted","acceptance_enabled","capability_accepted","fit_or_selector_retry")
    if (body.get("schema")!=original.OWNER_FOLD_SCHEMA or body.get("status")!="measured_candidate"
            or body.get("owner_policy")!=original.current_owner_policy()or body.get("fixture")is not data["fixture"]
            or body.get("artifact_state")!="unevaluated_development"
            or any(body.get(k)is not True for k in required_true)or any(body.get(k)is not False for k in required_false)
            or type(body.get("numerical_policy_passed"))is not bool):
        raise ValueError("current canonical structurally verified development publication required")
    if (body.get("policy")!=policy or body.get("policy_reference")!=policy_ref
            or body.get("recipe_id")!=development.RECIPE_ID or body.get("law",{}).get("selection")!=development.RECIPE_ID
            or body.get("capture_receipts")!=data["capture_receipts"]
            or any(body.get("streams",{}).get(s,{}).get("keep")!=policy["streams"][s]["keep"]for s in original.STREAMS)):
        raise ValueError("completed component policy/recipe/capture identity differs from requested stack")
    values,identity=joint.load_candidate(path,sha,data)
    return body,values,identity


def consolidate(out,index,blocks):
    """Stream physical1933-key state; no full60 state dict accumulated in RAM."""
    records=[index.body["global_source"]]
    for row in blocks:
        path,sha=_reference(row["fold_receipt"]);body=capture._json(path,sha)
        records.append({**body["output"],"path":str(path.parent/body["output"]["file"])})
    entries={};hashes={};shapes={};dtypes={};sources={};offset=0
    for ri,record in enumerate(records):
        path=capture._regular(record["path"])
        if single.file_sha(path)!=record["file_sha256"]:raise ValueError("component/global file changed before physical assembly")
        header,start=_source_header(path)
        if set(header)!=set(record["tensor_sha256"]):raise ValueError("incomplete physical source file header")
        for name,entry in header.items():
            if ri and name.startswith(TIME_PREFIX):
                if record["tensor_sha256"][name]!=records[0]["tensor_sha256"][name]:raise ValueError("duplicate shared timestep state differs")
                continue
            if name in entries:raise ValueError("duplicate physical stack tensor")
            if entry["shape"]!=record["shapes"][name]:raise ValueError("physical stack shape metadata differs")
            entries[name]=dict(entry);sources[name]=(path,start,entry["data_offsets"])
            hashes[name]=record["tensor_sha256"][name];shapes[name]=record["shapes"][name];dtypes[name]=record["dtypes"][name]
    if len(entries)!=1933 or len([k for k in entries if k.startswith("transformer_blocks.")])!=1920:
        raise ValueError("full60 physical accounting requires exactly1920+4+9 tensors")
    for name in sorted(entries):
        lo,hi=sources[name][2]
        if type(lo)is not int or type(hi)is not int or lo<0 or hi<=lo:raise ValueError("invalid physical tensor offsets")
        entries[name]["data_offsets"]=[offset,offset+hi-lo];offset+=hi-lo
    raw=json.dumps({k:entries[k] for k in sorted(entries)},separators=(",",":"),allow_nan=False).encode()
    raw+=b" "*((-len(raw))%8)
    out=Path(out);stage=out.with_name("."+out.name+"."+uuid.uuid4().hex+".pending")
    with stage.open("xb") as target:
        target.write(struct.pack("<Q",len(raw)));target.write(raw)
        for name in sorted(entries):
            path,start,(lo,hi)=sources[name];digest=hashlib.sha256()
            digest.update(dtypes[name].encode());digest.update(json.dumps(shapes[name],separators=(",",":")).encode())
            with path.open("rb") as source:
                source.seek(start+lo);remaining=hi-lo
                while remaining:
                    chunk=source.read(min(4<<20,remaining))
                    if not chunk:raise ValueError("truncated actual source component")
                    digest.update(chunk);target.write(chunk);remaining-=len(chunk)
            if digest.hexdigest()!=hashes[name]:raise ValueError("streamed physical tensor identity changed")
        target.flush();os.fsync(target.fileno())
    for record in records:
        if single.file_sha(Path(record["path"]))!=record["file_sha256"]:raise ValueError("source file changed during physical assembly")
    if stage.stat().st_size!=8+len(raw)+offset:raise ValueError("physical assembly byte count differs")
    # Independently verify actual output byte ranges before canonical publication.
    header,start=_source_header(stage)
    with stage.open("rb") as handle:
        for name,entry in header.items():
            lo,hi=entry["data_offsets"];handle.seek(start+lo);remaining=hi-lo;digest=hashlib.sha256()
            digest.update(dtypes[name].encode());digest.update(json.dumps(shapes[name],separators=(",",":")).encode())
            while remaining:
                chunk=handle.read(min(4<<20,remaining))
                if not chunk:raise ValueError("truncated assembled state")
                digest.update(chunk);remaining-=len(chunk)
            if digest.hexdigest()!=hashes[name]:raise ValueError("assembled physical tensor readback differs")
    os.link(stage,out)
    return {"file":out.name,"path":str(out.resolve()),"file_bytes":out.stat().st_size,"file_sha256":single.file_sha(out),
        "tensor_sha256":hashes,"shapes":shapes,"dtypes":dtypes,"tensor_sha256_schema":TENSOR_HASH_SCHEMA,
        "tensor_count":1933,"parameters":sum(math.prod(s) for s in shapes.values()),"storage_bytes":offset,
        "shared_timestep_copies_in_component_files":60,"shared_timestep_instances_in_standalone_state":1,
        "readback":"full streamed dtype/shape/raw tensor identity hashes verified before exclusive publication"}


def _coverage(requested,completed):
    missing=[b for b in range(DEPTH) if b not in completed]
    return {"required_blocks":list(range(DEPTH)),"requested_blocks":requested,"completed_blocks":sorted(completed),
            "missing_blocks":missing,"full_stack_complete":not missing and requested==list(range(DEPTH))}


def fold_stack(*,capture_manifest,capture_sha256,policy_path,policy_sha256,out,blocks=None,device="cpu",
               resume_manifest=None,resume_sha256=None,_fixture=False):
    """Executable independent block folds; resume into NEW directory only."""
    if device not in {"cpu","cuda:0"}:raise ValueError("actual fitter CPU/cuda:0 only")
    out=Path(out)
    if out.exists()or out.is_symlink():raise ValueError("new separate output root required; no overwrite")
    index=CaptureIndex(capture_manifest,capture_sha256,fixture=_fixture)
    requested=capture._blocks(index.request["blocks"] if blocks is None else blocks,DEPTH)
    if any(b not in index.request["blocks"] for b in requested):raise ValueError("requested fold block lacks authentic capture")
    policy_path=capture._regular(policy_path);policy=original.load_policy(policy_path,policy_sha256)
    policy_ref={"path":str(policy_path),"sha256":policy_sha256}
    implementation={**development.implementation_sha256(),Path(__file__).name:single.file_sha(Path(__file__)),
                    "c16_qimg_stack_capture.py":single.file_sha(Path(capture.__file__)),
                    "c16_qimg_joint_replay.py":single.file_sha(Path(joint.__file__))}
    protected=[*index.roots,policy_path.parent,Path(__file__).resolve().parent]
    if any(out.resolve()==p or out.resolve().is_relative_to(p)or p.is_relative_to(out.resolve())for p in protected):
        raise ValueError("fold output overlaps source/capture/policy/code roots")
    base={"schema":SCHEMA,"fixture":_fixture,"status":"partial","stack_capture":{"path":str(index.path),"sha256":index.sha256},
        "parent_header_sha256":index.request["source"]["header_sha256"],"global_source":index.body["global_source"],
        "policy":policy_ref,"recipe_id":development.RECIPE_ID,"owner_policy":original.current_owner_policy(),
        "implementation_sha256":implementation,"accepted":False,"capability_accepted":False,
        "scope":"plain QIMG60 backbone engineering only; ImageEdit contract and behavioral acceptance unproven"}
    completed={};resume=None
    if resume_manifest or resume_sha256:
        if not resume_manifest or not resume_sha256:raise ValueError("both exact resume reference fields required")
        rp=capture._regular(resume_manifest);old=capture._json(rp,resume_sha256)
        if out.resolve()==rp.parent or out.resolve().is_relative_to(rp.parent)or rp.parent.is_relative_to(out.resolve()):
            raise ValueError("resume output must be a distinct nonoverlapping new root")
        if any(old.get(k)!=v for k,v in base.items()if k!="status"):
            raise ValueError("resume source/policy/recipe/executable identity differs")
        rows=old.get("blocks",[])
        if (any(set(r)!={"block","fold_receipt"}or type(r["block"])is not int or r["block"]not in requested for r in rows)
                or [r["block"]for r in rows]!=sorted(set(r["block"]for r in rows))):raise ValueError("resume block coverage invalid")
        for row in rows:
            path,sha=_reference(row["fold_receipt"])
            data=load_block_inputs(index.path,index.sha256,row["block"],fixture=_fixture,index=index,load_forwards=False)
            _validate_component(row["fold_receipt"],data,policy,policy_ref)
            completed[row["block"]]=row["fold_receipt"]
        if old.get("coverage")!=_coverage(requested,completed):raise ValueError("resume coverage does not match verified components")
        resume={"path":str(rp),"sha256":resume_sha256}
    out.mkdir(parents=True,exist_ok=False)
    base["resume_from"]=resume
    def snapshot(name,extra=None):
        coverage=_coverage(requested,completed)
        body={**base,"blocks":[{"block":b,"fold_receipt":completed[b]}for b in sorted(completed)],"coverage":coverage}
        if extra:body.update(extra)
        body["status"]="complete"if coverage["full_stack_complete"]and body.get("standalone_state")and not body.get("error")else"partial"
        return capture._json_exclusive(out/name,body)
    snapshot("snapshot_000000.json")
    try:
        for block in requested:
            if block in completed:continue
            data=load_block_inputs(index.path,index.sha256,block,fixture=_fixture,index=index)
            completed[block]=_persist_block(data,policy,out/f"block_{block:02d}",index,policy_ref,device)
            snapshot(f"snapshot_{len(completed):06d}.json")
            print(json.dumps({"folded_block":block,"completed_blocks":len(completed),"requested_blocks":len(requested),"numeric_targets":"diagnostic"}),flush=True)
            del data;gc.collect()
        index.revalidate()
        if original.load_policy(policy_path,policy_sha256)!=policy:raise ValueError("policy changed during fold")
        current_implementation={**development.implementation_sha256(),Path(__file__).name:single.file_sha(Path(__file__)),
                    "c16_qimg_stack_capture.py":single.file_sha(Path(capture.__file__)),
                    "c16_qimg_joint_replay.py":single.file_sha(Path(joint.__file__))}
        if current_implementation!=implementation:
            raise ValueError("fitter/helper source changed during fold")
        standalone=None
        if _coverage(requested,completed)["full_stack_complete"]:
            for block in sorted(completed):
                data=load_block_inputs(index.path,index.sha256,block,fixture=_fixture,index=index,load_forwards=False)
                _validate_component(completed[block],data,policy,policy_ref)
                del data;gc.collect()
            standalone=consolidate(out/"plain_qimg_full60.safetensors",index,[{"block":b,"fold_receipt":completed[b]}for b in sorted(completed)])
        return snapshot("stack_fold_manifest.json",{"standalone_state":standalone,"functional_validation_complete":True})
    except BaseException as exc:
        snapshot("stack_fold_manifest.json",{"functional_validation_complete":False,"error":{"type":type(exc).__name__,"message":str(exc)},
            "incomplete_artifacts_preserved":True})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-manifest",required=True,type=Path);p.add_argument("--capture-sha256",required=True)
    p.add_argument("--policy-path",required=True,type=Path);p.add_argument("--policy-sha256",required=True)
    p.add_argument("--out",required=True,type=Path);p.add_argument("--blocks");p.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    p.add_argument("--resume-manifest",type=Path);p.add_argument("--resume-sha256")
    args=vars(p.parse_args())
    if args["blocks"]is not None:args["blocks"]=capture.parse_blocks(args["blocks"])
    print(json.dumps(fold_stack(**args),sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":main()
