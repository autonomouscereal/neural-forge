"""Source-bound raw C11 image ingress, without learned targets or native readiness.

Only real rendered image pixels enter these artifacts. Clean mesh coordinates,
encoder queries, answers and teacher feature targets are not image inputs.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from . import audio_admission as A
    from . import geometry_prepare as G
except ImportError:
    import audio_admission as A
    import geometry_prepare as G

PINS={"media_b_teacher_cache.py":"d62d646c852fe24a26223c5097cfeef8f9cb9c3307b342f09fb7b1fedf48fc48",
      "s7_pairs_parent.py":"4339b1952d6c84c112e311afc25a36a505f66888d833aa0fcd0924d5986d6ca6"}
SCHEMA="recovery_geometry_conditioning.v1"


def pinned_function(path,name,namespace):
    """Execute only one unchanged function from a fully pinned actual source."""
    if Path(path).name not in PINS or A.file_sha(path)!=PINS[Path(path).name]:
        raise ValueError("reviewed original C11 source pin differs")
    tree=ast.parse(Path(path).read_text(encoding="utf-8"))
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name]
    if len(functions)!=1:
        raise ValueError("exact pinned source function required")
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),functions[0]],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),"exec"),namespace)
    return namespace[name]


def patchify(pixels):
    """NumPy equivalent of actual source C*T*16*16 merge-window packing."""
    pixels=np.asarray(pixels)
    if (pixels.ndim!=3 or pixels.shape[0]!=3 or pixels.shape[1]%32 or pixels.shape[2]%32 or
        pixels.dtype!=np.float32 or not np.isfinite(pixels).all()):
        raise ValueError("finite float32 normalized RGB factor32 pixels required")
    c,h,w=pixels.shape;gh,gw=h//16,w//16
    frames=np.broadcast_to(pixels[None],(2,c,h,w))
    flat=frames.reshape(1,2,c,gh//2,2,16,gw//2,2,16).transpose(0,3,6,4,7,2,1,5,8).reshape(gh*gw,1536)
    bh,bw,mh,mw=np.meshgrid(np.arange(gh//2),np.arange(gw//2),np.arange(2),np.arange(2),indexing="ij")
    yx=np.stack(((bh*2+mh).reshape(-1),(bw*2+mw).reshape(-1)),-1).astype(np.int64)
    grid=np.array([[1,gh,gw]],dtype=np.int64)
    positions=np.zeros((len(flat),4),dtype=np.float32)
    positions[:,1]=yx[:,0].astype(np.float32)/max(1,gh-1)
    positions[:,2]=yx[:,1].astype(np.float32)/max(1,gw-1)
    return np.ascontiguousarray(flat),grid,yx,positions


def unpack(flat,grid):
    gh,gw=map(int,np.asarray(grid)[0,1:])
    if np.asarray(flat).shape!=(gh*gw,1536) or gh%2 or gw%2:
        raise ValueError("complete duplicated-frame merge-window patches required")
    return flat.reshape(1,gh//2,gw//2,2,2,3,2,16,16).transpose(0,6,5,1,3,7,2,4,8).reshape(2,3,gh*16,gw*16)


def image_ingress(path,expected_sha,media_b_source):
    if A.file_sha(path)!=expected_sha:
        raise ValueError("actual source-bound corrected image bytes changed")
    resize=pinned_function(media_b_source,"smart_resize",{"math":math})
    with Image.open(path) as source:
        if source.mode!="RGBA":
            raise ValueError("authored corrected source requires exact RGBA interpretation")
        rgba=np.asarray(source)
        if np.any(rgba[rgba[:,:,3]==0,:3]) or not np.isin(rgba[:,:,3],[0,255]).all():
            raise ValueError("authored opaque-object/transparent-black image policy differs")
        rgb=source.convert("RGB")
        h,w=resize(rgb.height,rgb.width,max_pixels=512*512)
        image=rgb.resize((w,h),Image.Resampling.BICUBIC)
        pixels=np.asarray(image,dtype=np.float32).transpose(2,0,1)/np.float32(255.)
        pixels=(pixels-np.float32(.5))/np.float32(.5)
    flat,grid,yx,positions=patchify(pixels)
    recovered=unpack(flat,grid)
    if not np.array_equal(recovered[0],pixels) or not np.array_equal(recovered[1],pixels):
        raise ValueError("exact source pixels were lost in raw C11 patch packing")
    return {"patch_flat":flat,"grid_thw":grid,"yx":yx,"native_media_positions":positions}, {
        "asset_sha256":expected_sha,"resized_hw":[h,w],"grid_thw":grid[0].tolist(),"rows":len(flat),"feature_width":1536,
        "alpha_policy":"exact PIL.convert(RGB): discard alpha; original transparent RGB is verified zero (black)",
        "normalization":"(float32 RGB / 255 - 0.5) / 0.5", "resize":"actual pinned smart_resize factor32/max262144 + PIL BICUBIC",
        "packing":"actual s7 merge-window row order; duplicate temporal frame2; C*T*16*16 features",
        "native_type_id":6,"native_feature_band":[640,2176],"raw_pixel_input":True,
        "learned_teacher_features":False,"teacher_targets_present":False,"mesh_xyz_present":False,
        "duplicate_frame_inverse_bitwise_verified":True,"original_HY3D_DINO_preprocess_parity":False}


def preparation(manifest,expected_sha):
    if A.file_sha(manifest)!=expected_sha:
        raise ValueError("immutable complete geometry preparation pin differs")
    m=json.loads(manifest.read_text(encoding="utf-8"))
    if (m.get("schema")!=G.SCHEMA or m.get("sealed") is not True or
        A.digest({k:v for k,v in m.items() if k!="manifest_sha256"})!=m.get("manifest_sha256") or
        any(m.get(k) is not True for k in ("complete_source_traversal","frozen_exclusions_verified","connected_groups_verified")) or
        m.get("target_leakage_policy",{}).get("clean_mesh_xyz_in_generation_inputs") is not False or
        m.get("target_leakage_policy",{}).get("clean_encoder_query_xyz_in_generation_inputs") is not False):
        raise ValueError("verified global geometry source/frozen/group traversal required")
    # Recheck every immutable global input, not only boolean assertions.
    for name,spec in m["sources"].items():
        A.pinned(spec,name)
    graph=A.pinned(m["graph_manifest"],"global source graph")
    g=json.loads(graph.read_text(encoding="utf-8"))
    if (g.get("sealed") is not True or A.digest({k:v for k,v in g.items() if k!="manifest_sha256"})!=g.get("manifest_sha256")):
        raise ValueError("full sealed source graph changed")
    streams={}
    for name in ("source_ready","admission"):
        spec=m["outputs"][name];path=A.pinned(spec,name);rows=list(A.rows(path))
        if len(rows)!=spec["rows"] or len(rows)!=m["rows"] or len({x["id"] for x in rows})!=len(rows):
            raise ValueError("incomplete unique geometry source/admission inventory")
        streams[name]={row["id"]:row for row in rows}
    if set(streams["source_ready"])!=set(streams["admission"]):
        raise ValueError("geometry source/admission identity sets differ")
    return m,streams


def prepare(*,manifest,manifest_sha256,media_b_source,s7_source,output):
    if output.exists():
        raise ValueError("choose a fresh image-ingress namespace")
    m,streams=preparation(manifest,manifest_sha256)
    pinned_function(s7_source,"qwen_vl_patchify",{}) # compile only; no Torch/model forward
    output.mkdir(parents=True);(output/"shards").mkdir()
    receipts=[]
    for item_id,row in streams["source_ready"].items():
        admission=streams["admission"][item_id]
        images=[x for x in row["media"] if x["type"]=="image" and x["role"]=="conditioning"]
        if (len(images)!=1 or admission.get("source_row_sha256")!=A.digest(row) or
            admission.get("schema")!=G.ADMISSION or admission.get("item_id")!=item_id or
            admission.get("asset_sha256")!=images[0]["sha256"] or admission.get("split")!=row["split"] or
            (admission.get("train_admitted") and (row["split"]!="train" or admission.get("held_out") or admission.get("blockers")))):
            raise ValueError("source image/admission/group/held identity differs")
        arrays,policy=image_ingress(Path(images[0]["path"]),images[0]["sha256"],media_b_source)
        path=output/"shards"/(item_id+".npz");np.savez_compressed(path,**arrays)
        receipts.append({"schema":SCHEMA,"id":item_id,"item_id":item_id,"source_row_sha256":A.digest(row),
             "source_group_id":row["provenance"]["group_id"],"recovery_group_id":admission["recovery_group_id"],
             "split":row["split"],"train_source_eligible":admission["train_admitted"],
             "conditioning_image":images[0],"raw_image_ingress":{"path":str(path.resolve()),"sha256":A.file_sha(path)},
             "policy":policy,"native_train_ready":False,"model_forward_executed":False})
    rows=output/"receipts.jsonl";rows.write_bytes(b"".join(A.canonical(x)+b"\n" for x in receipts))
    result={"schema":SCHEMA,"sealed":True,"source_year":2026,"source_code_sha256":A.file_sha(Path(__file__)),
            "preparation":{"path":str(manifest.resolve()),"sha256":manifest_sha256},"source_pins":{
                 str(Path(media_b_source).name):{"path":str(media_b_source),"sha256":PINS[Path(media_b_source).name]},
                 str(Path(s7_source).name):{"path":str(s7_source),"sha256":PINS[Path(s7_source).name]}},
            "complete_source_traversal":True,"rows":len(receipts),"train_source_eligible":sum(x["train_source_eligible"] for x in receipts),
            "held_sources_retained":sum(x["split"]!="train" for x in receipts),"native_train_ready":False,
            "original_source_pins":m["sources"],"global_graph_manifest":m["graph_manifest"],
            "outputs":{"receipts":{"path":str(rows.resolve()),"sha256":A.file_sha(rows),"rows":len(receipts)}},
            "blockers":["Actual Torch parity with the pinned s7 function is a separate recorded CPU check.",
                        "Native complete-sequence insertion/query-chart/objective proof and actual clean VAE target admission remain required."],
            "teacher_feature_targets_generated":False,"training_started":False}
    result["manifest_sha256"]=A.digest(result)
    (output/"manifest.json").write_bytes(A.canonical(result)+b"\n")
    return result


def verify_ingress(*,manifest,manifest_sha256,media_b_source,s7_source,assets_root=None):
    """Actual original Torch patchify parity for every prepared source image."""
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("raw pixel input parity is CPU only")
    if A.file_sha(manifest)!=manifest_sha256:
        raise ValueError("prepared raw C11 ingress file seal differs")
    m=json.loads(manifest.read_text(encoding="utf-8"))
    if (m.get("schema")!=SCHEMA or m.get("sealed") is not True or
        A.digest({k:v for k,v in m.items() if k!="manifest_sha256"})!=m.get("manifest_sha256")):
        raise ValueError("complete sealed image ingress required")
    ref=m["outputs"]["receipts"];p=manifest.parent/Path(ref["path"].replace("\\","/")).name
    if A.file_sha(p)!=ref["sha256"]:
        raise ValueError("complete raw ingress receipt inventory changed")
    records=list(A.rows(p))
    if len(records)!=ref["rows"] or len(records)!=m["rows"] or len({x["id"] for x in records})!=len(records):
        raise ValueError("incomplete raw image ingress traversal")
    namespace={"os":os}
    pinned_function(s7_source,"_torch",namespace)
    native=pinned_function(s7_source,"qwen_vl_patchify",namespace)
    checks=[]
    for row in records:
        image=Path(row["conditioning_image"]["path"]) if assets_root is None else assets_root/(row["id"]+".png")
        expected,policy=image_ingress(image,row["conditioning_image"]["sha256"],media_b_source)
        shard=manifest.parent/"shards"/Path(row["raw_image_ingress"]["path"].replace("\\","/")).name
        if A.file_sha(shard)!=row["raw_image_ingress"]["sha256"] or A.digest(policy)!=A.digest(row["policy"]):
            raise ValueError("source image/packing policy/shard seal changed")
        with np.load(shard,allow_pickle=False) as stored:
            if set(stored.files)!=set(expected) or any(not np.array_equal(stored[k],v) for k,v in expected.items()):
                raise ValueError("raw image ingress differs from original pinned image")
        pixels=unpack(expected["patch_flat"],expected["grid_thw"])[0]
        flat,grid,yx=native(torch.from_numpy(pixels.copy()))
        if any(not np.array_equal(v.cpu().numpy(),expected[k]) for k,v in (("patch_flat",flat),("grid_thw",grid),("yx",yx))):
            raise ValueError("actual original Torch QVL patchify parity failed")
        position=torch.zeros(len(flat),4,dtype=torch.float32)
        position[:,1]=yx[:,0].float()/max(1,int(grid[0,1])-1)
        position[:,2]=yx[:,1].float()/max(1,int(grid[0,2])-1)
        if not np.array_equal(position.numpy(),expected["native_media_positions"]):
            raise ValueError("actual Torch C11 source positions differ")
        checks.append({"id":row["id"],"asset_sha256":row["conditioning_image"]["sha256"],
                       "source_row_sha256":row["source_row_sha256"],"split":row["split"],
                       "raw_pixel_patch_grid_order_positions_bitwise_equal":True,"rows":len(flat)})
    return {"schema":"recovery_geometry_conditioning_cpu_parity.v1","ingress_manifest_sha256":manifest_sha256,
            "torch_version":str(torch.__version__),"device":"cpu","actual_original_patchify_executed":True,
            "full_source_traversal":True,"checks":checks,"source_pins":PINS,"native_train_ready":False,
            "teacher_weights_loaded":False,"teacher_features_generated":False,"training_started":False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-manifest","media-b-source","s7-source","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--prepared-manifest-sha256",required=True)
    a=p.parse_args(argv)
    r=prepare(manifest=a.prepared_manifest,manifest_sha256=a.prepared_manifest_sha256,
              media_b_source=a.media_b_source,s7_source=a.s7_source,output=a.output)
    print(json.dumps({k:r[k] for k in ("rows","train_source_eligible","held_sources_retained","native_train_ready","manifest_sha256")}))


if __name__=="__main__":main()
