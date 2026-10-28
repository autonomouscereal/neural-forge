#!/usr/bin/env python3
"""Bind four completed native parent meshes to block0 capture, without inference.

Preparation and post-capture lineage checks only. Parent generation, posterior
flow probing, block replay/folding and final child quality remain distinct.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re

import c16_capture_hy3d as capture
import c16_hy3d_parent_meshes as generator
from c16_hy3d_contract import CAPTURE_SCHEMA, MODEL_PATH, SOURCE_ROOT, require, tensor_header, validate_capture

SCHEMA = "neural-forge.hy3d-parent-to-block0-plan.v1"
CHECK_SCHEMA = "neural-forge.hy3d-parent-to-block0-lineage.v1"
PROBE = "object_posterior_flow_interpolation_not_generated_scheduler_trajectory"
FIELD = "native_binary_field_logits_not_metric_sdf"
IMAGE_SIZE = 512


class Pins:
    def __init__(self):
        self.files = {}

    def file(self, path, expected=None, limit=256 << 20):
        path = Path(path).resolve(strict=True)
        digest = capture.digest_file(path, limit)
        require(expected is None or digest == expected, f"source artifact hash mismatch: {path.name}")
        record = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
        old = self.files.setdefault(str(path), record)
        require(old == record, "source artifact changed during preparation")
        return record

    def json(self, path, expected=None):
        ref = self.file(path, expected, 2 << 20)
        result = capture.unique_json(capture.bounded_bytes(Path(ref["path"])))
        require(isinstance(result, dict), "receipt must be a JSON object")
        return result, ref

    def stable(self):
        for ref in list(self.files.values()):
            self.file(ref["path"], ref["sha256"], max(ref["bytes"], 1))


def local_file(directory, recorded, expected_name):
    """Explicit mount remap by exact known basename, then content verification.

    The historical container paths stay in original immutable receipts. We do
    not rewrite them or infer correspondence from names without digest joins.
    """
    require(isinstance(recorded, str) and Path(recorded).name == expected_name
            and re.fullmatch(r"[A-Za-z0-9_.-]+", expected_name), "unexpected source filename/mount mapping")
    root = Path(directory).resolve(strict=True)
    path = (root / expected_name).resolve(strict=True)
    require(path.parent == root, "source path escapes declared mount")
    return path


def trajectory(path, record, settings, pins):
    ref = pins.file(path, record["file_sha256"])
    require(ref["bytes"] == record["bytes"], "trajectory size mismatch")
    header = tensor_header(path)
    t = header["tensors"]
    steps, dims = settings["steps"], capture.CAPTURE_DIMS
    n, ch, h, ct, cw = (dims[k] for k in ("tokens", "channels", "hidden", "context_tokens", "context_width"))
    shapes = {"context": [2,ct,cw], "trajectory_latent_before": [steps,1,n,ch],
              "trajectory_model_timestep": [steps,2], "trajectory_model_output": [steps,2,n,ch],
              "trajectory_time_token": [steps,2,1,h], "trajectory_cfg_output": [steps,1,n,ch],
              "trajectory_scheduler_timestep": [steps], "trajectory_sigma_pair": [steps,2],
              "trajectory_latent_after": [steps,1,n,ch], "step_index_before": [steps],
              "step_index_after": [steps], "final_latent": [1,n,ch],
              "field_logits": [1]+[settings["octree_resolution"]+1]*3,
              "conditioning_image": [1,3,IMAGE_SIZE,IMAGE_SIZE], "conditioning_mask": [1,1,IMAGE_SIZE,IMAGE_SIZE]}
    require(set(t) == set(shapes), "incomplete native trajectory inventory")
    for name, shape in shapes.items():
        require(t[name]["shape"] == shape, f"native trajectory shape mismatch: {name}")
        expected_dtype = ("I64" if name.startswith("step_index_") else "F32" if name in {
            "trajectory_sigma_pair", "trajectory_scheduler_timestep", "field_logits", "conditioning_image", "conditioning_mask"} else "F16")
        require(t[name]["dtype"] == expected_dtype, f"native trajectory dtype mismatch: {name}")
    # Scheduler state is evidence only; never substitute it for capture's mesh
    # posterior interpolation, choose a different sigma, or regenerate context.
    from safetensors import safe_open
    import torch
    with safe_open(str(path), framework="pt", device="cpu") as sf:
        before, after = sf.get_tensor("step_index_before"), sf.get_tensor("step_index_after")
        require(torch.equal(before, torch.tensor([-1]+list(range(1,steps))))
                and torch.equal(after, torch.arange(1,steps+1)), "native scheduler step identity mismatch")
        sigmas = sf.get_tensor("trajectory_sigma_pair")
        require(bool(torch.isfinite(sigmas).all()) and bool(((sigmas >= 0) & (sigmas <= 1)).all())
                and bool((sigmas[:,0] >= sigmas[:,1]).all()) and bool(sigmas[-1,1] == 0)
                and torch.equal(sigmas[:-1,1], sigmas[1:,0]), "native sigma sequence mismatch")
        # Actual source has a final duplicate zero; do not invent strict descent.
        require(torch.equal(sf.get_tensor("final_latent"), sf.get_tensor("trajectory_latent_after")[-1]),
                "native final latent/trajectory mismatch")
    return {**ref, "header_sha256": header["header_sha256"], "steps": steps,
            "semantics": "actual_CFG_parent_generation_not_capture_probe"}


def prepare(parent_plan, meshes, qimg, output, *, checkpoint=MODEL_PATH, source_root=SOURCE_ROOT):
    """Verify immutable completed source outputs and emit a fresh local plan."""
    output = Path(output).resolve()
    require(not output.exists(), "fresh preparation directory required")
    pins = Pins()
    plan, plan_ref = pins.json(parent_plan)
    pre, pre_ref = pins.json(Path(meshes)/"preregistered-plan.json")
    run, run_ref = pins.json(Path(meshes)/"mesh-run.json")
    objects, objects_ref = pins.json(Path(meshes)/"objects.json")
    require(plan.get("schema") == generator.PLAN_SCHEMA and pre.get("plan") == plan
            and pre.get("plan_sha256") == run.get("plan_sha256") == plan_ref["sha256"], "preregistered parent plan mismatch")
    require(run.get("schema") == generator.SCHEMA and run.get("status") == "complete"
            and run.get("source_parent_generated") is True and run.get("folded_child_proven") is False
            and run.get("mesh_sdf_pairs_complete") is False, "completed source-only mesh run required")
    settings = plan["generation"]
    require(run["generation"] == settings and all(settings.get(k) == v for k,v in generator.FIXED.items()),
            "unreviewed parent generation/extraction settings")
    require(type(settings["steps"]) is int and 2 <= settings["steps"] <= 50
            and type(settings["octree_resolution"]) is int and 8 <= settings["octree_resolution"] <= 256
            and 0 <= settings["capture_sigma"] <= 1, "invalid bounded generation/capture settings")
    source = capture.inspect_source(checkpoint, source_root)
    identity = source["source_identity"]
    for record in (pre, run):
        require(record["source_identity"] == identity and record["extra_source_sha256"] == generator.EXTRA_SOURCE
                and record["comfy_source_sha256"] == generator.COMFY_SOURCE, "native source identity mismatch")
    qfile = local_file(qimg, plan["qimg_receipt_file"], "parent_images_receipt.json")
    qrecord, qref = pins.json(qfile, plan["qimg_receipt_sha256"])
    require(qrecord.get("schema") == "neural_forge.qimg_parent_images.v1" and qrecord.get("status") == "complete",
            "completed native QIMG input receipt required")
    require(pre["qimg_receipt_sha256"] == run["qimg_receipt_sha256"] == qref["sha256"], "QIMG lineage mismatch")
    rows = plan["images"]
    require(len(rows) == len(run["samples"]) == len(objects["objects"]) == 4
            and [r["split"] for r in rows].count("fit") == [r["split"] for r in rows].count("held") == 2,
            "exact preregistered 2 fit + 2 held objects required")
    require(objects.get("schema") == "neural-forge.hy3d-capture-objects.v1", "wrong generator objects schema")
    qrows = {r["id"]:r for r in qrecord["images"]}
    require(len(qrows) == len(qrecord["images"]), "duplicate QIMG source identity")
    verified, seen_ids, seen_geom, seen_images = [], set(), set(), set()
    for ordinal, (row, item, old_object) in enumerate(zip(rows,run["samples"],objects["objects"])):
        name = row["id"]
        require(isinstance(name,str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}",name) and name not in seen_ids,
                "safe unique original object IDs required")
        seen_ids.add(name)
        require(type(row["seed"]) is int and 0 <= row["seed"] < 2**32, "original uint32 generation/capture seed required")
        rec, rec_ref = pins.json(Path(meshes)/(name+".json"), old_object["generation_receipt_sha256"])
        require(Path(old_object["generation_receipt_file"]).name == name+".json" and rec == item,
                "per-object receipt/run order mismatch")
        for key in ("id","split","seed","conditioning_sha256"):
            require(rec[key] == row[key], "original image/split/seed receipt mismatch")
        require(rec["generation"] == settings and rec["source_identity"] == identity
                and rec["plan_sha256"] == plan_ref["sha256"] and rec["qimg_receipt_sha256"] == qref["sha256"]
                and rec["source_parent_generated"] is True and rec["folded_child_proven"] is False
                and rec["field_kind"] == FIELD, "object source/extraction lineage mismatch")
        qi = qrows.get(name)
        require(qi is not None and rec["qimg_image"] == qi and qi["sha256"] == row["conditioning_sha256"]
                and {"fit":"fit","holdout":"held"}.get(qi["split"]) == row["split"], "QIMG pairing/split mismatch")
        image = local_file(qimg,row["conditioning_file"],name+".png")
        image_ref = pins.file(image,row["conditioning_sha256"],32 << 20)
        mesh = local_file(meshes,rec["object_file"],name+".ply")
        mesh_ref = pins.file(mesh,rec["object_sha256"])
        geometry = capture.geometry_identity(mesh,mesh_ref["sha256"])
        require(geometry == rec["geometry_identity"] == rec["extraction"]["geometry_identity"], "canonical native mesh geometry mismatch")
        require(geometry["sha256"] not in seen_geom and image_ref["sha256"] not in seen_images,
                "four distinct geometry and conditioning identities required")
        seen_geom.add(geometry["sha256"])
        seen_images.add(image_ref["sha256"])
        require(rec["extraction"]["native_winding_reversed"] is True and rec["extraction"]["trimesh_process"] is False,
                "unexpected geometry postprocess")
        for key,value in {"object_sha256":mesh_ref["sha256"],"geometry_identity":geometry,
                          "conditioning_sha256":image_ref["sha256"],"seed":row["seed"],
                          "split":row["split"],"sigma":settings["capture_sigma"]}.items():
            require(old_object[key] == value, f"generator object list mismatch: {key}")
        trpath = local_file(meshes,rec["trajectory"]["file"],name+".safetensors")
        tr = trajectory(trpath,rec["trajectory"],settings,pins)
        verified.append({"id":name,"ordinal":ordinal,"object_file":str(mesh),"object_sha256":mesh_ref["sha256"],
                         "conditioning_file":str(image),"conditioning_sha256":image_ref["sha256"],
                         "geometry_identity":geometry,"seed":row["seed"],"split":row["split"],
                         "sigma":settings["capture_sigma"],"generation_receipt":rec_ref,"trajectory":tr})
    pins.stable()
    output.mkdir(parents=True,exist_ok=False)
    object_file = output/"objects.json"
    capture.sealed_json(object_file,{"schema":"neural-forge.hy3d-capture-objects.v1","objects":verified})
    object_ref = {"path":str(object_file),"sha256":capture.digest_file(object_file)}
    result = {"schema":SCHEMA,"status":"prepared_not_captured","block_index":0,"source_identity":identity,
              "objects":object_ref,"samples":verified,"source_inputs":list(pins.files.values()),
              "parent_plan":plan_ref,"parent_run":run_ref,"preregistered":pre_ref,
              "generator_objects":objects_ref,"qimg_receipt":qref,"dit_probe":PROBE,
              "field_kind":FIELD,"parent_trajectory_used_as_capture_input":False,
              "folded_child_proven":False,"runtime_allocation":"root_owned_not_created_by_this_tool",
              "capture_argv":["c16_capture_hy3d.py","--parent-checkpoint",str(checkpoint),"--source-root",str(source_root),
                              "--objects",str(object_file),"--output",str(output/"capture"),"--block","0",
                              "--query-chunk","256","--device","cuda:0"],
              "code_sha256":{p.name:capture.digest_file(p) for p in (Path(__file__),Path(capture.__file__),Path(generator.__file__))}}
    capture.sealed_json(output/"capture-plan.json",result)
    return result


def verify_capture(plan_file, plan_sha, capture_dir):
    """Post-capture source lineage only; native replay/quality gates stay separate."""
    pins = Pins()
    plan, pref = pins.json(plan_file,plan_sha)
    require(plan.get("schema") == SCHEMA and plan.get("block_index") == 0 and plan.get("dit_probe") == PROBE,
            "not a prepared block0 parent bridge")
    for ref in plan["source_inputs"]:
        pins.file(ref["path"],ref["sha256"],max(ref["bytes"],1))
    objects,_ = pins.json(plan["objects"]["path"],plan["objects"]["sha256"])
    require(objects["objects"] == plan["samples"],"prepared object plan changed")
    directory = Path(capture_dir)
    run, run_ref = pins.json(directory/"capture-run.json")
    require(run.get("schema") == capture.SCHEMA and run.get("source_identity") == plan["source_identity"]
            and run.get("producer") == capture.PRODUCER and run.get("dit_probe") == PROBE
            and run.get("posterior") == "mode" and run.get("folded_child_proven") is False,
            "capture run source/semantics mismatch")
    require(len(run["samples"]) == 4,"complete4-object capture required")
    expected_samples = []
    for i,(original,actual) in enumerate(zip(plan["samples"],run["samples"])):
        receipt,_ = pins.json(directory/f"sample-{i:04d}.json")
        require(receipt == actual and actual["requested_sigma"] == original["sigma"],"capture sample/sigma order mismatch")
        expected = {k:original[k] for k in ("object_sha256","conditioning_sha256","geometry_identity","seed","split")}
        require(all(actual[k] == v for k,v in expected.items()),"captured object/image/geometry/seed/split mismatch")
        require(actual["file"] == f"sample-{i:04d}.safetensors","capture sample path mismatch")
        pins.file(directory/actual["file"],actual["file_sha256"],2 << 30)
        expected_samples.append(expected)
    block,_ = pins.json(directory/"block-00/capture.json")
    require(block["schema"] == CAPTURE_SCHEMA and block["block_index"] == 0
            and block["source_identity"] == plan["source_identity"] and block["samples"] == expected_samples,
            "complete native block0 lineage mismatch")
    source = {"source_identity":plan["source_identity"]}
    checked = validate_capture(directory/"block-00",source)
    require(run["dit_capture"] == checked,"capture-run/native block receipt mismatch")
    pins.stable()
    return {"schema":CHECK_SCHEMA,"status":"source_capture_lineage_verified_not_quality",
            "plan":pref,"capture_run":run_ref,"block":checked,"samples":plan["samples"],
            "dit_probe":PROBE,"parent_trajectory_used_as_capture_input":False,"folded_child_proven":False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    a = sub.add_parser("prepare")
    for name in ("parent-plan","meshes","qimg","output"):
        a.add_argument("--"+name,required=True)
    a.add_argument("--parent-checkpoint",default=MODEL_PATH)
    a.add_argument("--source-root",default=SOURCE_ROOT)
    b = sub.add_parser("verify-capture")
    for name in ("plan","plan-sha256","capture","output"):
        b.add_argument("--"+name,required=True)
    args = p.parse_args(argv)
    if args.command == "prepare":
        prepare(args.parent_plan,args.meshes,args.qimg,args.output,checkpoint=args.parent_checkpoint,source_root=args.source_root)
    else:
        result = verify_capture(args.plan,args.plan_sha256,args.capture)
        capture.sealed_json(Path(args.output),result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
