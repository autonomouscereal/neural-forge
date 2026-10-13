"""Prepare actual authored mesh/render targets with complete global source admission.

The surface is a target-side VAE input. It is never a generation query chart.
No teacher, native model, GPU, or historical command is executed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from . import audio_admission as A
except ImportError:
    import audio_admission as A

SCHEMA = "recovery_geometry_preparation.v1"
ADMISSION = "recovery_geometry_admission.v1"
SURFACE_POLICY = "operator_triangle_area_uniform_seeded_f16_roundtrip_xyz_normal_zero_flag_2026.v1"


def mesh(path):
    vertices, faces = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v" and len(fields) == 4:
            vertices.append([float(x) for x in fields[1:]])
        elif fields[0] == "f" and len(fields) == 4 and all(x.isdecimal() for x in fields[1:]):
            faces.append([int(x)-1 for x in fields[1:]])
        else:
            raise ValueError("only explicit finite triangular OBJ geometry is supported")
    v, f = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    if (v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or len(f) < 4 or
        not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v) or np.max(np.abs(v)) > 100):
        raise ValueError("invalid bounded physical mesh")
    lengths = np.ptp(v, axis=0)
    if np.any(lengths <= 1e-8):
        raise ValueError("planar/degenerate geometry cannot be a physical volume target")
    edges = Counter(tuple(sorted((int(face[i]), int(face[(i+1)%3])))) for face in f for i in range(3))
    oriented = Counter((int(face[i]),int(face[(i+1)%3])) for face in f for i in range(3))
    if any(n != 2 for n in edges.values()) or any(oriented[(a,b)] != oriented[(b,a)] for a,b in edges):
        raise ValueError("mesh must be closed with consistent oriented edge incidence")
    tri = v[f]
    cross = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    double_areas = np.linalg.norm(cross,axis=1)
    volume = float(np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
    if np.any(double_areas <= 1e-12) or volume <= 1e-10:
        raise ValueError("degenerate faces or nonpositive oriented volume")
    return v,f,{"vertex_count":len(v),"triangle_count":len(f),"bbox_min":v.min(0).tolist(),
                "bbox_max":v.max(0).tolist(),"volume":volume,"closed_manifold":True,
                "consistent_orientation":True,"surface_area":float(double_areas.sum()/2)}


def normalize(v):
    center=(v.min(0)+v.max(0))/2
    factor=2*.9999/np.ptp(v,axis=0).max()
    return (v-center)*factor,{"bbox_center":center.tolist(),"uniform_scale":float(factor),"native_normalize_mesh_scale":.9999}


def render(v,f,*,side=512,yaw=30.,elevation=20.,color=(90,155,205)):
    """Orthographic pixel-center barycentric rasterization, nearest camera depth."""
    if side < 32 or side > 1024:
        raise ValueError("bounded render size required")
    yaw,elevation=np.radians([yaw,elevation])
    right=np.array([np.cos(yaw),-np.sin(yaw),0.])
    up=np.array([-np.sin(yaw)*np.sin(elevation),-np.cos(yaw)*np.sin(elevation),np.cos(elevation)])
    near=np.array([np.sin(yaw)*np.cos(elevation),np.cos(yaw)*np.cos(elevation),np.sin(elevation)])
    x,y,z=v@right,v@up,v@near
    scale=side*.37/max(float(np.ptp(x)),float(np.ptp(y)))*2
    projected=np.stack([side/2+(x-(x.max()+x.min())/2)*scale,
                        side/2-(y-(y.max()+y.min())/2)*scale,z],axis=1)
    rgba=np.zeros((side,side,4),dtype=np.uint8)
    depth=np.full((side,side),-np.inf)
    light=np.array([.35,-.4,.85]);light/=np.linalg.norm(light)
    for face in f:
        p=projected[face]
        lo=np.maximum(np.floor(p[:,:2].min(0)).astype(int),0)
        hi=np.minimum(np.ceil(p[:,:2].max(0)).astype(int),side-1)
        if np.any(hi<lo):
            continue
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1)+.5,np.arange(lo[1],hi[1]+1)+.5)
        a,b,c=p
        den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-12:
            continue
        w0=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        w1=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        w2=1-w0-w1
        dz=w0*a[2]+w1*b[2]+w2*c[2]
        patch=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        take=(w0>=-1e-10)&(w1>=-1e-10)&(w2>=-1e-10)&(dz>patch)
        normal=np.cross(v[face[1]]-v[face[0]],v[face[2]]-v[face[0]])
        normal/=np.linalg.norm(normal)
        rgb=np.clip(np.asarray(color)*(.35+.65*max(0.,float(normal@light))),0,255).astype(np.uint8)
        patch[take]=dz[take]
        pixels=rgba[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        pixels[take,:3]=rgb;pixels[take,3]=255
    covered=np.isfinite(depth)
    if not .02 < covered.mean() < .85 or np.any(rgba[~covered]) or not np.array_equal(rgba[:,:,3]>0,covered):
        raise ValueError("render lacks meaningful exact object alpha/depth coverage")
    return rgba,{"projection":"orthographic","yaw_degrees":float(np.degrees(yaw)),
                 "elevation_degrees":float(np.degrees(elevation)),"side":side,"pixel_centers":True,
                 "depth_test":"largest camera-near coordinate wins","camera_right":right.tolist(),
                 "camera_up":up.tolist(),"camera_near":near.tolist(),"pixels_per_unit":float(scale),
                 "opaque_object_pixels":int(covered.sum()),"transparent_background":True,
                 "annotation_or_chart_present":False}


def sample_surface(v,f,seed,count=81920):
    if count != 81920:
        raise ValueError("reviewed HY3D2.1 surface requires exactly81920 uniform points")
    tri=v[f]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=1)
    normals=cross/area[:,None]
    rng=np.random.default_rng(seed)
    choice=rng.choice(len(f),count,p=area/area.sum())
    uv=rng.random((count,2));s=np.sqrt(uv[:,0]);bary=np.stack([1-s,s*(1-uv[:,1]),s*uv[:,1]],axis=1)
    points=np.einsum("ni,nij->nj",bary,tri[choice])
    # Native SharpEdgeSurfaceLoader uses this FP16 intermediate. This direct,
    # seeded uniform sampler is an explicit operator policy, not loader parity.
    surface=np.concatenate([points,normals[choice]],axis=1).astype(np.float16).astype(np.float32)
    surface=np.concatenate([surface,np.zeros((count,1),dtype=np.float32)],axis=1)[None]
    if surface.shape != (1,81920,7) or not np.isfinite(surface).all() or np.any(surface[:,:,6]):
        raise ValueError("invalid real mesh surface contract")
    return surface


def prepare(*,specs,asset_root,output):
    if output.exists():
        raise ValueError("choose a fresh geometry output directory")
    paths={k:A.pinned(v,k) for k,v in specs.items()}
    _,inventory=A.procedural_candidates(paths,specs,asset_root)
    output.mkdir(parents=True)
    sources=[];candidates=[]
    for r in inventory:
        if r["route"] != "volume_generate":
            continue
        images=[m for m in r["media"] if m["type"]=="image" and m["role"]=="conditioning"]
        targets=[m for m in r["media"] if m["type"]=="mesh" and m["role"]=="target"]
        if len(images)!=1 or len(targets)!=1:
            raise ValueError("actual image+mesh conditioning pair required")
        v,f,measurement=mesh(targets[0]["path"])
        truth=json.loads(Path(r["provenance"]["groundtruth_path"]).read_text())["groundtruth"]
        for name in ("vertex_count","triangle_count","closed_manifold"):
            if truth[name] != measurement[name]:
                raise ValueError("original authored geometry groundtruth changed")
        if not np.isclose(truth["volume"],measurement["volume"],rtol=1e-10):
            raise ValueError("actual mesh volume differs from original groundtruth")
        normalized,transform=normalize(v)
        seed=int(A.digest([r["id"],"geometry2026.v1"])[:8],16)
        rgba,camera=render(normalized,f,yaw=20+seed%65,elevation=15+(seed//65)%31,
                           color=tuple(75+(seed>>shift)%140 for shift in (0,8,16)))
        directory=output/"assets"/r["id"];directory.mkdir(parents=True)
        image=directory/"physical_object_rgba.png";Image.fromarray(rgba).save(image)
        copied_mesh=directory/"target_geometry.obj"
        copied_mesh.write_bytes(Path(targets[0]["path"]).read_bytes())
        if A.file_sha(copied_mesh)!=targets[0]["sha256"]:
            raise ValueError("original physical geometry copy differs")
        surface=directory/"target_surface.npy";np.save(surface,sample_surface(normalized,f,seed),allow_pickle=False)
        rec=json.loads(json.dumps(r))
        rec["id"]="geometry-corrected-"+r["id"]
        rec["media"][0]={"type":"image","role":"conditioning","path":str(image.resolve()),"sha256":A.file_sha(image),
                          "bytes":image.stat().st_size,"source_media_sha256":images[0]["sha256"],
                          "width":512,"height":512,"mode":"RGBA"}
        rec["media"][1]={**targets[0],"path":str(copied_mesh.resolve()),"source_media_sha256":targets[0]["sha256"]}
        rec["provenance"].update(original_row_sha256=A.digest(r),original_id=r["id"],
             derived_renderer_sha256=A.file_sha(Path(__file__)),physical_object_render=True,
             original_mesh_sha256=targets[0]["sha256"],render_camera=camera,mesh_normalization=transform)
        rec["quality"].update(train_ready=False,native_cache_status="actual_shape_vae_encoding_required",
                               readiness_scope="verified physical mesh+render and target surface; no native readiness")
        # The actual geometry target lives outside generation input media.
        target={"path":str(surface.resolve()),"sha256":A.file_sha(surface),"bytes":surface.stat().st_size,
                "shape":[1,81920,7],"dtype":"float32","policy":SURFACE_POLICY,"seed":seed,
                "normalization":transform,"native_loader_parity":False,"generation_input_allowed":False}
        rec["geometry_target"]={"mesh":rec["media"][1],"surface":target,"measure":measurement,
                                 "generation_query_xyz_allowed":False,"clean_query_points_are_target_only":True}
        sources.append(rec)
        candidates.append({"record":rec,"asset_path":str(image.resolve()),"asset_sha256":A.file_sha(image),
             "geometry_category":"physical_object_render","genuine_geometry_target":True,
             "source_is_chart_or_icon":False,"geometry_target":rec["geometry_target"],"groundtruth_sha256":r["provenance"]["groundtruth_sha256"],
             "measure":measurement})
    if not sources:
        raise ValueError("no complete actual geometry source pairs")
    sourcefile=output/"source_ready.jsonl";sourcefile.write_bytes(b"".join(A.canonical(r)+b"\n" for r in sources))
    graph_specs={**specs,"geometry_source":{"path":str(sourcefile.resolve()),"sha256":A.file_sha(sourcefile)}}
    graph=A.seal_admission(specs=graph_specs,candidates=candidates,inventory=inventory,output=output/"global_source_checks")
    admission=list(A.rows(Path(graph["outputs"]["admission"]["path"])))
    for r in admission:
        r.update(schema=ADMISSION,held_out=r["split"]!="train",encoder_status="actual_immutable_ShapeVAE_probe_required")
    ap=output/"admission.jsonl";ap.write_bytes(b"".join(A.canonical(r)+b"\n" for r in admission))
    manifest={"schema":SCHEMA,"created_utc":datetime.now(timezone.utc).isoformat(),"source_year":2026,"sealed":True,
              "complete_source_traversal":True,"frozen_exclusions_verified":True,"connected_groups_verified":True,
              "source_admission_schema":ADMISSION,"sources":graph["sources"],"graph_manifest":{"path":str((output/"global_source_checks/manifest.json").resolve()),"sha256":A.file_sha(output/"global_source_checks/manifest.json")},
              "outputs":{"source_ready":{"path":str(sourcefile.resolve()),"sha256":A.file_sha(sourcefile),"rows":len(sources)},
                         "admission":{"path":str(ap.resolve()),"sha256":A.file_sha(ap),"rows":len(admission)}},
              "rows":len(sources),"train_admitted":sum(r["train_admitted"] for r in admission),
              "split_counts":dict(Counter(r["split"] for r in sources)),"source_code_sha256":A.file_sha(Path(__file__)),
              "surface_policy":SURFACE_POLICY,"native_train_ready":False,"training_started":False,
              "target_leakage_policy":{"clean_mesh_xyz_in_generation_inputs":False,"clean_encoder_query_xyz_in_generation_inputs":False},
              "blockers":["Actual pinned ShapeVAE encode and geometry reconstruction proof pending.",
                          "Native HY3D unit dropped DINO cross attention; source-bound child QVL image features pending.",
                          "Parent has no generation XYZ/ordinal chart; dated target-independent query-chart amendment and validation pending."],
              "coverage":"24 small authored polyhedral object instances across three exclusive families; no natural-object claim"}
    manifest["manifest_sha256"]=A.digest(manifest)
    (output/"manifest.json").write_bytes(A.canonical(manifest)+b"\n")
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pins",type=Path,required=True);p.add_argument("--asset-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    result=prepare(specs=json.loads(a.pins.read_text(encoding="utf-8")),asset_root=a.asset_root,output=a.output)
    print(json.dumps({k:result[k] for k in ("rows","train_admitted","split_counts","outputs","manifest_sha256","native_train_ready")}))


if __name__=="__main__":
    main()
