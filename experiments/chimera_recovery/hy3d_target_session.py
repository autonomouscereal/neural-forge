"""One bounded warm CPU session prepares actual source-sealed ShapeVAE targets.

Original parent surface sampler is called once per source and captured. The
original posterior.sample uses one prescribed seed; no best-seed selection.
This is clean target preparation, not a guided cache or training admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

try:
    from . import hy3d_surface_probe as H
    from . import geometry_conditioning as C
except ImportError:
    import hy3d_surface_probe as H
    import geometry_conditioning as C
A,G=H.A,H.G
PLAN="recovery_hy3d_target_plan.v1"
SESSION="recovery_hy3d_target_session.v1"
SEED=20261013


def tool_pins():
    base=Path(__file__).resolve().parent
    return {name:A.file_sha(base/name) for name in ("hy3d_target_session.py","hy3d_surface_probe.py",
                                                  "geometry_conditioning.py","geometry_prepare.py","audio_admission.py")}


def make_plan(manifest,manifest_sha256,output):
    if output.exists():raise ValueError("choose a new target preparation plan")
    m,streams=C.preparation(manifest,manifest_sha256)
    entries=[];seen=set()
    for item_id,row in streams["source_ready"].items():
        admission=streams["admission"][item_id];family=row["provenance"]["template_family"]
        representative=family not in seen;seen.add(family)
        if admission["source_row_sha256"]!=A.digest(row):raise ValueError("source admission join changed")
        entries.append({"id":item_id,"source_row_sha256":A.digest(row),"source_admission_sha256":A.digest(admission),
             "original_id":row["provenance"]["original_id"],"family":family,"source_group_id":row["provenance"]["group_id"],
             "split":row["split"],"source_train_eligible":admission["train_admitted"],"sample_seed":SEED,
             "repeat_sample":representative,"dense_resolution":32 if representative and family!="mesh_box" else 16})
    plan={"schema":PLAN,"sealed":True,"source_year":2026,"preparation":{"path":str(manifest.resolve()),"sha256":manifest_sha256},
          "original_global_source_pins":m["sources"],"global_source_graph":m["graph_manifest"],
          "parent_source_pins":H.PINS,"original_whole_parent_sha256":H.WEIGHT_SHA,"tool_pins":tool_pins(),"sources":entries,
          "policy":{"sampler":"unchanged original SharpEdgeSurfaceLoader81920uniform/0sharp captured once per source",
             "posterior":"unchanged original posterior.sample(); fixed seed20261013", "best_seed_selection":False,
             "sample_surface_seedability_claimed":False,"batch_size":1,"device":"cpu",
             "geometry_oracle":"actual normalized convex mesh half-spaces, prescribed125queries",
             "geometry_quality_admission":"requires separate reviewed thresholds/amendment; no automatic admission",
             "clean_encoder_query_xyz_generation_input_allowed":False,"native_train_ready":False},
          "rows":len(entries),"training_started":False}
    plan["plan_sha256"]=A.digest(plan);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(A.canonical(plan)+b"\n")
    return plan


def validate_plan(path,expected_sha):
    if not A.SHA.fullmatch(str(expected_sha)) or A.file_sha(path)!=expected_sha:
        raise ValueError("target preparation plan file pin differs")
    plan=json.loads(path.read_text(encoding="utf-8"))
    if (plan.get("schema")!=PLAN or plan.get("sealed") is not True or
        A.digest({k:v for k,v in plan.items() if k!="plan_sha256"})!=plan.get("plan_sha256") or
        plan.get("parent_source_pins")!=H.PINS or plan.get("original_whole_parent_sha256")!=H.WEIGHT_SHA or
        plan.get("tool_pins")!=tool_pins() or plan.get("policy",{}).get("best_seed_selection") is not False or
        plan.get("policy",{}).get("clean_encoder_query_xyz_generation_input_allowed") is not False):
        raise ValueError("reviewed target preparation plan/source/objective differs")
    records=plan["sources"]
    if len(records)!=plan["rows"] or len(records)!=24 or len({r["id"] for r in records})!=len(records):
        raise ValueError("full unique24source plan required")
    for r in records:
        if (r["sample_seed"]!=SEED or r["dense_resolution"] not in (16,32) or type(r["repeat_sample"]) is not bool or
            not re.fullmatch(r"media-seed-[a-f0-9]{24}",r["original_id"]) or
            r["id"]!="geometry-corrected-"+r["original_id"] or
            (r["source_train_eligible"] and r["split"]!="train")):
            raise ValueError("fixed source/seed/family/held plan policy differs")
    return plan


def admission_join(plan,manifest,input_root):
    body=json.loads(manifest.read_text(encoding="utf-8"));ref=body["outputs"]["admission"]
    path=input_root/Path(ref["path"].replace("\\","/")).name
    if A.file_sha(path)!=ref["sha256"]:raise ValueError("complete actual source admission bytes changed")
    records=list(A.rows(path))
    if len(records)!=24 or len({r["id"] for r in records})!=24:raise ValueError("full unique actual24source admission required")
    index={r["id"]:r for r in records}
    if set(index)!={r["id"] for r in plan["sources"]}:raise ValueError("plan/admission source sets differ")
    for entry in plan["sources"]:
        r=index[entry["id"]]
        if (A.digest(r)!=entry["source_admission_sha256"] or r["source_row_sha256"]!=entry["source_row_sha256"] or
            r.get("schema")!=G.ADMISSION or r.get("item_id")!=entry["id"] or
            r.get("split")!=entry["split"] or r.get("train_admitted") is not entry["source_train_eligible"] or
            (r["train_admitted"] and (r.get("held_out") or r.get("blockers") or r["split"]!="train"))):
            raise ValueError("source admission/split/frozen eligibility changed")
    return index


def tensor_hash(value):
    value=value.detach().cpu().contiguous()
    h=hashlib.sha256(A.canonical({"dtype":str(value.dtype),"shape":list(value.shape)}))
    h.update(value.numpy().tobytes());return h.hexdigest()


def model_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(A.canonical([name,tensor_hash(value)]))
    return h.hexdigest()


def run(*,plan_path,plan_sha256,input_root,source_root,weights,output,threads=8):
    if output.exists():raise ValueError("choose a fresh warm target session output")
    output.mkdir(parents=True)
    plan=validate_plan(plan_path,plan_sha256)
    source_pins=H.source_pins(source_root)
    manifest=input_root/"manifest.json"
    if A.file_sha(manifest)!=plan["preparation"]["sha256"]:raise ValueError("actual preparation manifest changed")
    admission_join(plan,manifest,input_root)
    if weights.stat().st_size!=H.WEIGHT_BYTES or A.file_sha(weights)!=H.WEIGHT_SHA:
        raise ValueError("actual whole original parent checkpoint changed")
    import numpy as np
    import torch
    import torch_cluster
    import yaml
    from safetensors import safe_open
    from safetensors.torch import save_file
    if torch.cuda.is_initialized() or not 1<=threads<=8:raise ValueError("bounded CPU-only target session required")
    torch.set_num_threads(threads)
    params=H.validate_config(yaml.safe_load((source_root/"configs/dit_config_2_1.yaml").read_text()))
    fps=H.fps_dependency(torch,torch_cluster)
    vae=H.native_vae(source_root)(**params).to(device="cpu",dtype=torch.float32).eval().requires_grad_(False)
    with safe_open(str(weights),framework="pt",device="cpu") as handle:
        expected=vae.state_dict();names={k[4:] for k in handle.keys() if k.startswith("vae.")}
        if names!=set(expected) or len(names)!=410:raise ValueError("actual original410VAE tensor keys differ")
        state={}
        for name,reference in expected.items():
            value=handle.get_tensor("vae."+name)
            if value.shape!=reference.shape:raise ValueError("native VAE tensor shape differs:"+name)
            state[name]=value.float()
        vae.load_state_dict(state,strict=True);del state,expected
    initial=model_hash(vae);measured={};started=time.monotonic();receipts=[]
    def observe(module,args,result):
        if list(args[0].shape)!=[1,81920,3] or list(args[1].shape)!=[1,81920,4]:
            raise ValueError("warm native encoder actual source dimensions differ")
        measured["query_xyz"]=result[1][0].detach().cpu().clone()
    hook=vae.encoder.register_forward_hook(observe)
    for entry in plan["sources"]:
        directory=output/entry["id"];directory.mkdir();asset=input_root/"assets"/entry["original_id"]
        row,operator_surface,_=H.source_item(manifest,entry["id"],asset/"target_surface.npy",asset/"physical_object_rgba.png")
        H.actual_mesh_surface(row,operator_surface,asset/"target_geometry.obj")
        if A.digest(row)!=entry["source_row_sha256"]:raise ValueError("warm source row identity differs")
        surface,v,f,control=H.parent_surface(source_root,asset/"target_geometry.obj",directory,torch,SEED)
        x=torch.from_numpy(surface.copy());tick=time.monotonic()
        with torch.inference_mode():
            H.reset_cpu_rng(torch,SEED);raw=vae.encode(x,sample_posterior=True);query=measured["query_xyz"].clone()
            if list(raw.shape)!=[1,4096,64] or not torch.isfinite(raw).all():raise ValueError("actual clean raw VAE target invalid")
            repeated=None
            if entry["repeat_sample"]:
                H.reset_cpu_rng(torch,SEED);again=vae.encode(x,sample_posterior=True)
                repeated=bool(torch.equal(raw,again) and torch.equal(query,measured["query_xyz"]))
                if not repeated:raise ValueError("warm prescribed native sample/query repeat failed")
            modelspace=raw*float(vae.scale_factor)
            if not torch.allclose(modelspace/float(vae.scale_factor),raw,atol=2e-7,rtol=2e-7):
                raise ValueError("native raw/model-space inverse failed")
            decoded=vae.decode(raw)
            if list(decoded.shape)!=[1,4096,1024] or not torch.isfinite(decoded).all():raise ValueError("actual clean target decoder invalid")
            queries,inside=H.convex_oracle(v,f);qt=torch.from_numpy(queries).unsqueeze(0)
            logits=vae.geo_decoder(queries=qt,latents=decoded)
            if not torch.isfinite(logits).all():raise ValueError("native geometry head output nonfinite")
            dense=H.dense_reconstruction(vae,decoded,v,f,directory,"sample",entry["dense_resolution"],SEED)
        target=directory/"clean_target.safetensors"
        save_file({"clean_raw_latent":raw.contiguous(),"clean_model_space_latent":modelspace.contiguous(),
                   "target_only_encoder_query_xyz":query.contiguous(),"physical_oracle_queries":qt.contiguous(),
                   "native_geometry_logits":logits.contiguous(),"physical_oracle_inside":torch.from_numpy(inside).contiguous()},str(target))
        report={"schema":"recovery_hy3d_clean_target.v1", "id":entry["id"],"source_row_sha256":entry["source_row_sha256"],
            "source_admission_sha256":entry["source_admission_sha256"],"source_group_id":entry["source_group_id"],
            "split":entry["split"],"family":entry["family"],"source_train_eligible":entry["source_train_eligible"],
            "parent_surface_control":control,"sample_seed":SEED,"encoder":"unchanged original posterior.sample()",
            "repeated_sample_and_query_bitwise_equal":repeated,"new_seed_selection":False,
            "raw_latent_shape":[1,4096,64],"model_space_policy":"raw * original VAE.scale_factor", "scale_factor":float(vae.scale_factor),
            "raw_model_inverse_verified":True,"decoded_shape":[1,4096,1024],"decoded_feature_sha256":tensor_hash(decoded),
            "actual_encode_decode_seconds":time.monotonic()-tick,"raw_target_sha256":tensor_hash(raw),
            "clean_target":{"path":str(target),"sha256":A.file_sha(target)},"dense_reconstruction":dense,
            "oracle_confusion":H.oracle_metrics(logits[0,:,0].numpy(),inside),"conditioning_image_sha256":row["media"][0]["sha256"],
            "source_mesh_sha256":row["geometry_target"]["mesh"]["sha256"],"target_query_xyz_generation_input_allowed":False,
            "native_train_ready":False,"geometry_quality_admission_granted":False,"training_started":False}
        report["receipt_sha256"]=A.digest(report);(directory/"receipt.json").write_bytes(A.canonical(report)+b"\n")
        receipts.append({"id":entry["id"],"path":str(directory/"receipt.json"),"sha256":A.file_sha(directory/"receipt.json"),
                         "split":entry["split"],"source_train_eligible":entry["source_train_eligible"]})
        (output/"status.json").write_bytes(A.canonical({"schema":SESSION,"complete":False,"completed":len(receipts),"expected":24,
            "elapsed_seconds":time.monotonic()-started,"native_train_ready":False,"receipts":receipts})+b"\n")
        print("Actual native target:",len(receipts),entry["id"],flush=True)
    hook.remove();final=model_hash(vae)
    if final!=initial or H.source_pins(source_root)!=source_pins or A.file_sha(weights)!=H.WEIGHT_SHA:
        raise ValueError("original warm parent model/source/whole-weight integrity changed")
    result={"schema":SESSION,"sealed":True,"complete_source_traversal":True,"source_year":2026,"rows":24,
        "plan":{"path":str(plan_path),"sha256":plan_sha256},"tool_pins":tool_pins(),"source_pins":source_pins,
        "weights":{"sha256":H.WEIGHT_SHA,"bytes":H.WEIGHT_BYTES},"runtime":{"torch":str(torch.__version__),"numpy":str(np.__version__)},
        "device":"cpu","threads":threads,"fps_dependency":fps,"initial_vae_full_state_sha256":initial,"final_vae_full_state_sha256":final,
        "original_weights_unchanged":True,"whole_checkpoint_reverified_after":True,"elapsed_seconds":time.monotonic()-started,
        "targets":receipts,"source_train_eligible":sum(r["source_train_eligible"] for r in receipts),
        "held_sources_retained":sum(r["split"]!="train" for r in receipts),"native_train_ready":False,"training_started":False,
        "blockers":["Separate reviewed per-source geometry quality admission is required.",
                    "Source-bound raw C11 input and explicit native query-chart/objective/preregistration amendment joins remain required."]}
    result["manifest_sha256"]=A.digest(result);(output/"manifest.json").write_bytes(A.canonical(result)+b"\n")
    (output/"status.json").write_bytes(A.canonical({"schema":SESSION,"complete":True,"completed":24,"expected":24,
        "elapsed_seconds":result["elapsed_seconds"],"manifest_sha256":result["manifest_sha256"],"native_train_ready":False})+b"\n")
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    prep=sub.add_parser("plan");prep.add_argument("--prepared-manifest",type=Path,required=True)
    prep.add_argument("--prepared-manifest-sha256",required=True);prep.add_argument("--output",type=Path,required=True)
    execute=sub.add_parser("run")
    for name in ("plan","input-root","source-root","weights","output"):execute.add_argument("--"+name,type=Path,required=True)
    execute.add_argument("--plan-sha256",required=True);execute.add_argument("--threads",type=int,default=8)
    a=p.parse_args(argv)
    if a.action=="plan":result=make_plan(a.prepared_manifest,a.prepared_manifest_sha256,a.output)
    else:result=run(plan_path=a.plan,plan_sha256=a.plan_sha256,input_root=a.input_root,source_root=a.source_root,
                    weights=a.weights,output=a.output,threads=a.threads)
    print(json.dumps({"schema":result["schema"],"rows":result["rows"],"native_train_ready":False}))


if __name__=="__main__":main()
