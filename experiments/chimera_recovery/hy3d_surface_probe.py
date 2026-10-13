"""CPU-only, source-pinned real ShapeVAE target encode probe; no generation chart.

Preflight is the default. --execute loads only the actual original VAE tensors.
Target surface and captured query XYZ must never be used as generation inputs.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.machinery
import json
from pathlib import Path
import random
import sys
import time
import types

try:
    from . import audio_admission as A
    from . import geometry_prepare as G
except ImportError:
    import audio_admission as A
    import geometry_prepare as G

WEIGHT_SHA="5f21e98a6cb99b13b5e224abaee33929570fff7af2b6a0060001559a04ba9d72"
WEIGHT_BYTES=7365943290
PINS={
 "configs/dit_config_2_1.yaml":"a04b6a62f7084e1490eaba825355d340983bcbc88809bb36cad37b35979845aa",
 "hy3dshape/hy3dshape/surface_loaders.py":"c4a821fffd9ea9f121ac07a034faaa9e52350859cc667db955ae91c17ee1c001",
 "hy3dshape/hy3dshape/models/autoencoders/model.py":"b81a4586cc1148def7fe157c18071fb7dc50753cfbf1930932bbcbecbcb35829",
 "hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py":"4be3dccc02da6f773c979175280ee0618ce5cc58917016a16c554745fc051537",
 "hy3dshape/hy3dshape/preprocessors.py":"284dc24fabbf974bd6793ebd58e95b7700550d90371cacb96ad0889ec558de1e",
 "hy3dshape/hy3dshape/pipelines.py":"cf56918d62d5bee11a80641638dbc590bd541a1264c384e302a87d68a4cd285b",
 "hy3dshape/hy3dshape/models/autoencoders/attention_processors.py":"9a461ddde9a2bf5ca0cd3961c3c1b86f9c81772066fe5b815bc497e0b5a8f053",
 "hy3dshape/hy3dshape/models/autoencoders/volume_decoders.py":"95ffab6bd0a11034bc2b7415fef1379e27c4d2238638f831a0456019d90484eb",
 "hy3dshape/hy3dshape/models/autoencoders/surface_extractors.py":"203d1551ccef37e86f9809b9ac0fac68e8e4a2f633ca72a85408535595ae68e5",
 "hy3dshape/hy3dshape/utils/__init__.py":"0be86a4b0a9c7885550f80fd6f9900855246f6fb0d8df6000e07414258d6590f",
 "hy3dshape/hy3dshape/utils/utils.py":"09fda07cb7e7aafbbc2d0a0a808d312d6687bcb1a939c33cdf9f88dc07d42ca0",
 "hy3dshape/hy3dshape/utils/misc.py":"e8f41fef6db65993b87d64f9a540e75c3b4f8c85f6c703683929d6250ee3ec16"}


def source_pins(root):
    result={}
    for name,pin in PINS.items():
        p=root/name
        if A.file_sha(p)!=pin:
            raise ValueError("reviewed actual HY3D source changed: "+name)
        result[name]={"path":str(p.resolve()),"sha256":pin,"bytes":p.stat().st_size}
    return result


def source_item(manifest,item_id,surface,conditioning):
    body=json.loads(manifest.read_text(encoding="utf-8"))
    if (body.get("schema")!=G.SCHEMA or body.get("sealed") is not True or
        A.digest({k:v for k,v in body.items() if k!="manifest_sha256"})!=body.get("manifest_sha256") or
        body.get("target_leakage_policy",{}).get("clean_encoder_query_xyz_in_generation_inputs") is not False):
        raise ValueError("sealed target-independent geometry preparation required")
    # Resolve the immutable source JSONL next to the copied manifest. Explicit
    # remapped input paths are accepted only if bytes match the original seals.
    ref=body["outputs"]["source_ready"]
    p=manifest.parent/Path(str(ref["path"]).replace("\\","/")).name
    if A.file_sha(p)!=ref["sha256"]:
        raise ValueError("full geometry source inventory changed")
    records=list(A.rows(p))
    if len(records)!=ref["rows"] or len({r["id"] for r in records})!=len(records):
        raise ValueError("incomplete unique geometry source traversal")
    matches=[r for r in records if r["id"]==item_id]
    if len(matches)!=1:
        raise ValueError("exact source item required")
    r=matches[0];target=r["geometry_target"]["surface"]
    image=[m for m in r["media"] if m["type"]=="image" and m["role"]=="conditioning"]
    if (len(image)!=1 or A.file_sha(conditioning)!=image[0]["sha256"] or
        A.file_sha(surface)!=target["sha256"] or surface.stat().st_size!=target["bytes"] or
        target["shape"]!=[1,81920,7] or target["policy"]!=G.SURFACE_POLICY or
        target.get("generation_input_allowed") is not False):
        raise ValueError("actual surface/image target pairing changed")
    import numpy as np
    arr=np.load(surface,allow_pickle=False)
    if (arr.shape!=(1,81920,7) or arr.dtype!=np.float32 or not np.isfinite(arr).all() or
        np.any(arr[:,:,6]) or np.max(np.abs(arr[:,:,:3]))>1.0001 or
        not np.allclose(np.linalg.norm(arr[:,:,3:6],axis=-1),1.,atol=.001)):
        raise ValueError("real normalized XYZ/normal/zero-flag surface required")
    return r,arr,body


def actual_mesh_surface(record,arr,mesh_path):
    import numpy as np
    target=record["geometry_target"]
    if A.file_sha(mesh_path)!=target["mesh"]["sha256"]:
        raise ValueError("actual sealed physical target mesh changed")
    v,f,measure=G.mesh(mesh_path);v,transform=G.normalize(v)
    surface=target["surface"]
    if A.digest(measure)!=A.digest(target["measure"]) or A.digest(transform)!=A.digest(surface["normalization"]):
        raise ValueError("physical mesh measurement/normalization differs")
    if not np.array_equal(arr,G.sample_surface(v,f,surface["seed"])):
        raise ValueError("sealed target surface is not sampled from the actual physical mesh")
    return v,f


def convex_oracle(v,f):
    import numpy as np
    tri=v[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
    if np.any(np.einsum("vfi,fi->vf",v[:,None,:]-tri[None,:,0,:],normal)>1e-7):
        raise ValueError("bounded plane oracle applies only to actual convex authored meshes")
    grid=np.stack(np.meshgrid(*([np.linspace(-1.1,1.1,5)]*3),indexing="ij"),axis=-1).reshape(-1,3)
    signed=np.einsum("qfi,fi->qf",grid[:,None,:]-tri[None,:,0,:],normal).max(axis=1)
    keep=np.abs(signed)>1e-5
    return grid[keep].astype(np.float32),(signed[keep]<0)


def oracle_metrics(logits,inside):
    """Report both logit interpretations; class imbalance must remain visible."""
    import numpy as np
    values=np.asarray(logits).reshape(-1);truth=np.asarray(inside,dtype=bool).reshape(-1)
    if values.shape!=truth.shape or not np.isfinite(values).all() or not truth.any() or truth.all():
        raise ValueError("finite paired oracle with both classes required")
    result={"queries":len(truth),"inside":int(truth.sum()),"outside":int((~truth).sum())}
    for name,prediction in (("positive_logit_as_inside",values>0),("negative_logit_as_inside",values<0)):
        tp=int((prediction&truth).sum());tn=int((~prediction&~truth).sum())
        fp=int((prediction&~truth).sum());fn=int((~prediction&truth).sum())
        result[name]={"true_positive":tp,"true_negative":tn,"false_positive":fp,"false_negative":fn,
                      "accuracy":(tp+tn)/len(truth),"inside_recall":tp/(tp+fn),
                      "outside_recall":tn/(tn+fp),"balanced_accuracy":.5*(tp/(tp+fn)+tn/(tn+fp))}
    return result


def captured_surface(path,receipt_path,receipt_sha256,record,preparation_sha256):
    """Reuse one frozen actual native sample only through its complete source seal."""
    import numpy as np
    if not A.SHA.fullmatch(str(receipt_sha256)) or A.file_sha(receipt_path)!=receipt_sha256:
        raise ValueError("actual captured native surface receipt file pin differs")
    body=json.loads(receipt_path.read_text(encoding="utf-8"))
    seal=body.get("receipt_sha256");unsigned={k:v for k,v in body.items() if k!="receipt_sha256"}
    control=body.get("parent_surface_control",{})
    if (A.digest(unsigned)!=seal or body.get("schema")!="chimera_recovery.hy3d_clean_encode_probe.v1" or
        body.get("encode_executed") is not True or body.get("native_surface_loader_parity") is not True or
        body.get("weights",{}).get("sha256")!=WEIGHT_SHA or body.get("item_id")!=record["id"] or
        body.get("source_row_sha256")!=A.digest(record) or body.get("preparation_sha256")!=preparation_sha256 or
        body.get("mesh_sha256")!=record["geometry_target"]["mesh"]["sha256"] or
        control.get("loader")!="original SharpEdgeSurfaceLoader" or control.get("num_uniform_points")!=81920 or
        control.get("num_sharp_points")!=0 or control.get("unseeded_default_rng_unchanged") is not True or
        any(body.get("source_pins",{}).get(k,{}).get("sha256")!=v for k,v in PINS.items()) or
        A.file_sha(path)!=control.get("surface",{}).get("sha256")):
        raise ValueError("captured native surface provenance/source/asset pairing differs")
    arr=np.load(path,allow_pickle=False)
    if arr.shape!=(1,81920,7) or arr.dtype!=np.float32 or not np.isfinite(arr).all() or np.any(arr[:,:,6]):
        raise ValueError("captured native surface array contract differs")
    return arr,{"path":str(receipt_path),"sha256":receipt_sha256,"receipt_sha256":seal,
                "captured_surface_sha256":A.file_sha(path),"new_sampler_call":False}


def parent_surface(root,mesh_path,output,torch,seed,frozen=None):
    """Call unchanged native loader once; its unseeded default_rng is not replaced."""
    import numpy as np
    import trimesh
    module=importlib.import_module("_chimera_clean_hy3d_native.surface_loaders")
    loaded=trimesh.load(str(mesh_path),force="mesh",merge_primitives=True)
    bounds=loaded.bounds.copy();center=(bounds[0]+bounds[1])/2;scale=2*.9999/np.ptp(bounds,axis=0).max()
    normalized=module.normalize_mesh(loaded.copy())
    expected=(np.asarray(loaded.vertices)-center)*scale
    if not np.allclose(normalized.vertices,expected,rtol=0,atol=1e-12):
        raise ValueError("original loader normalized coordinate affine differs")
    if frozen is None:
        reset_cpu_rng(torch,seed)
        surface=module.SharpEdgeSurfaceLoader(num_uniform_points=81920,num_sharp_points=0)(str(mesh_path))
        arr=surface.cpu().numpy()
    else:
        arr=frozen[0]
    if (arr.shape!=(1,81920,7) or arr.dtype!=np.float32 or not np.isfinite(arr).all() or
        np.any(arr[:,:,6]) or not np.allclose(np.linalg.norm(arr[:,:,3:6],axis=-1),1.,atol=.002)):
        raise ValueError("actual parent surface dimensions/normals/sharp flag differ")
    v=np.asarray(normalized.vertices);f=np.asarray(normalized.faces)
    tri=v[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.linalg.norm(normal,axis=-1)[:,None]
    signed=np.einsum("qfi,fi->qf",arr[0,:,:3][:,None,:]-tri[None,:,0,:],normal).max(axis=1)
    if np.max(np.abs(signed))>.002:
        raise ValueError("native sampled surface does not cover the normalized actual target")
    path=output/"captured_parent_surface.npy";np.save(path,arr,allow_pickle=False)
    return arr,v,f,{"loader":"original SharpEdgeSurfaceLoader", "num_uniform_points":81920,"num_sharp_points":0,
                    "sharpedge_flag":True,"actual_flag_unique":np.unique(arr[:,:,6]).tolist(),
                    "surface":{"path":str(path),"sha256":A.file_sha(path),"shape":list(arr.shape)},
                    "sampler_repeatability_claimed":False,"unseeded_default_rng_unchanged":True,
                    "source_to_native_affine":{"bbox_min":bounds[0].tolist(),"bbox_max":bounds[1].tolist(),
                         "bbox_center":center.tolist(),"uniform_scale":float(scale),"scale_argument":.9999,
                         "formula":"(source_xyz - bbox_center) * uniform_scale"},
                    "normalized_bbox_min":v.min(0).tolist(),"normalized_bbox_max":v.max(0).tolist(),
                    "maximum_surface_plane_distance":float(np.max(np.abs(signed))),
                    "native_surface_loader_parity":True,"generation_input_allowed":False,
                    "prior_native_capture":frozen[1] if frozen else None}


def sampled_surface(v,f,count,seed):
    import numpy as np
    v=np.asarray(v);f=np.asarray(f)
    if (v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or len(f)==0 or
        f.min()<0 or f.max()>=len(v) or not np.isfinite(v).all() or not 64<=count<=8192):
        raise ValueError("bounded finite actual triangle mesh required")
    triangles=v[f];area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=-1)
    if not np.isfinite(area).all() or area.sum()<=0:
        raise ValueError("actual mesh has no finite surface area")
    rng=np.random.default_rng(seed);choice=rng.choice(len(f),count,p=area/area.sum())
    uv=rng.random((count,2));a=np.sqrt(uv[:,0]);bary=np.stack([1-a,a*(1-uv[:,1]),a*uv[:,1]],axis=-1)
    return np.einsum("ni,nij->nj",bary,triangles[choice])


def mesh_metrics(target_v,target_f,reconstructed_v,reconstructed_f,seed=20261013,count=4096):
    import numpy as np
    truth=sampled_surface(target_v,target_f,count,seed)
    observed=sampled_surface(reconstructed_v,reconstructed_f,count,seed+1)
    # Bounded exact nearest neighbor between sampled point sets, no new runtime
    # dependency. The continuous triangle surfaces are still only sampled.
    def distances(first,second):
        return np.concatenate([np.sqrt(((first[i:i+128,None,:]-second[None,:,:])**2).sum(-1).min(-1))
                               for i in range(0,len(first),128)])
    forward=distances(truth,observed);backward=distances(observed,truth)
    return {"metric":"two_direction_triangle_area_uniform_sampled_surface_nearest_neighbor",
            "samples_per_surface":count,"seed":seed,"units":"original loader normalized coordinate",
            "target_to_reconstruction_mean":float(forward.mean()),"reconstruction_to_target_mean":float(backward.mean()),
            "chamfer_l1_sum_of_means":float(forward.mean()+backward.mean()),
            "chamfer_l2_sum_of_means":float((forward**2).mean()+(backward**2).mean()),
            "target_to_reconstruction_p95":float(np.quantile(forward,.95)),
            "reconstruction_to_target_p95":float(np.quantile(backward,.95)),
            "exact_continuous_mesh_distance_claimed":False,
            "target_bbox_min":np.asarray(target_v).min(0).tolist(),"target_bbox_max":np.asarray(target_v).max(0).tolist(),
            "reconstruction_bbox_min":np.asarray(reconstructed_v).min(0).tolist(),
            "reconstruction_bbox_max":np.asarray(reconstructed_v).max(0).tolist()}


def dense_reconstruction(vae,decoded,vertices,faces,output,label,resolution,seed):
    if resolution not in (16,32,48,64):
        raise ValueError("bounded reviewed dense resolution required")
    started=time.monotonic()
    outputs=vae.latents2mesh(decoded,bounds=1.1,mc_level=0.,num_chunks=2048,
                             octree_resolution=resolution,enable_pbar=False)
    result={"method":"original VanillaVolumeDecoder + MCSurfaceExtractor", "resolution":resolution,
            "actual_grid_queries":(resolution+1)**3,"bounds":1.1,"mc_level":0.,"num_chunks":2048,
            "native_extractor_coordinate_divisor":resolution+1,"algorithm_changed":False,
            "original_dmc_convention":"sdf = -grid_logit / octree_resolution",
            "native_ready":False}
    if len(outputs)!=1 or outputs[0] is None:
        result.update(extracted=False,seconds=time.monotonic()-started,blocker="Original MC surface extraction returned no mesh.")
        return result
    v,f=outputs[0].mesh_v,outputs[0].mesh_f
    result["metrics"]=mesh_metrics(vertices,faces,v,f,seed=seed)
    path=output/(label+"_native_mc.obj")
    path.write_text("".join("v "+" ".join(map(str,x))+"\n" for x in v)+
                    "".join("f "+" ".join(str(int(i)+1) for i in x)+"\n" for x in f),encoding="utf-8")
    result.update(extracted=True,vertices=len(v),triangles=len(f),seconds=time.monotonic()-started,
                  mesh={"path":str(path),"sha256":A.file_sha(path)})
    return result


def reset_cpu_rng(torch,seed):
    # Genuine CPU torch_cluster fps uses libc rand for random_start, separately
    # from the Torch randperm generator. This affects only this owned process.
    if sys.platform!="linux":
        raise ValueError("reviewed CPU fps RNG policy requires Linux libc")
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)
    libc=ctypes.CDLL(None);libc.srand.argtypes=[ctypes.c_uint];libc.srand(seed)


def fps_dependency(torch,cluster):
    pin="e02265f20012f65467c0479b3bccecc62a043786330b9f223ddc1ebc64ee113d"
    path=Path(cluster.__file__).parent/"_fps_cpu.so"
    if (str(torch.__version__)!="2.6.0+cu124" or str(cluster.__version__)!="1.6.3+pt26cpu" or
        A.file_sha(path)!=pin):
        raise ValueError("reviewed actual CPU fps binary/runtime pin changed")
    points=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=torch.float32)
    runs=[]
    for _ in range(3):
        reset_cpu_rng(torch,20261013)
        runs.append(cluster.fps(points,ratio=.5,random_start=True).tolist())
    if any(x!=runs[0] for x in runs[1:]):
        raise ValueError("genuine CPU fps repeat failed despite both RNG resets")
    return {"path":str(path),"sha256":pin,"repeated_actual_cpu_fps":runs,"batch_size":1,
            "algorithm_changed":False}


def native_vae(root):
    base=root/"hy3dshape/hy3dshape";prefix="_chimera_clean_hy3d_native"
    for suffix in ("", ".models", ".models.autoencoders"):
        name=prefix+suffix;module=types.ModuleType(name)
        module.__path__=[str(base.joinpath(*suffix.strip(".").split("."))) if suffix else str(base)]
        module.__package__=name;module.__spec__=importlib.machinery.ModuleSpec(name,loader=None,is_package=True)
        module.__spec__.submodule_search_locations=module.__path__;sys.modules[name]=module
    return importlib.import_module(prefix+".models.autoencoders.model").ShapeVAE


def validate_config(config):
    p=config["vae"]["params"]
    expected={"num_latents":4096,"embed_dim":64,"width":1024,"heads":16,"num_encoder_layers":8,
              "num_decoder_layers":16,"point_feats":4,"pc_size":81920,"pc_sharpedge_size":0,
              "include_pi":False,"qkv_bias":False,"qk_norm":True,"scale_factor":1.0039506158752403}
    if any(p.get(k)!=v for k,v in expected.items()):
        raise ValueError("actual reviewed ShapeVAE geometry/config changed")
    return p


def probe(*,root,weights,manifest,manifest_sha256,item_id,surface,mesh_path,conditioning,output,execute=False,threads=8,seed=20261013,
          native_surface_control=False,dense_resolution=0,compare_posterior_sample=False,
          frozen_parent_surface=None,parent_surface_receipt=None,parent_surface_receipt_sha256=None,dense_sample_only=False):
    if output.exists():
        raise ValueError("choose a new diagnostic output directory")
    # Refuse an unwritable owned output before hashing/loading the parent or
    # performing costly native inference. Container UID must own this mount.
    output.mkdir(parents=True)
    pins=source_pins(root)
    if not A.SHA.fullmatch(str(manifest_sha256)) or A.file_sha(manifest)!=manifest_sha256:
        raise ValueError("immutable geometry preparation file pin differs")
    r,arr,prep=source_item(manifest,item_id,surface,conditioning)
    vertices,faces=actual_mesh_surface(r,arr,mesh_path)
    frozen=None
    reference_args=(frozen_parent_surface,parent_surface_receipt,parent_surface_receipt_sha256)
    if any(x is not None for x in reference_args):
        if not native_surface_control or any(x is None for x in reference_args):
            raise ValueError("complete original-loader frozen sample reference required")
        frozen=captured_surface(frozen_parent_surface,parent_surface_receipt,parent_surface_receipt_sha256,r,manifest_sha256)
    if dense_sample_only and (not compare_posterior_sample or not dense_resolution):
        raise ValueError("sample-only dense control requires explicit sampled posterior and resolution")
    if weights.stat().st_size!=WEIGHT_BYTES or A.file_sha(weights)!=WEIGHT_SHA:
        raise ValueError("original full HY3D checkpoint pin differs")
    import yaml
    params=validate_config(yaml.safe_load((root/"configs/dit_config_2_1.yaml").read_text()))
    errors=[];runtime={}
    for name in ("torch","torch_cluster","safetensors","trimesh","numpy"):
        try:
            m=importlib.import_module(name);runtime[name]={"version":str(getattr(m,"__version__","unknown")),"file":str(m.__file__)}
        except ImportError as e:
            errors.append(name+": "+str(e))
    report={"schema":"chimera_recovery.hy3d_clean_encode_probe.v1","source_year":2026,"source_pins":pins,
            "weights":{"path":str(weights),"sha256":WEIGHT_SHA,"bytes":WEIGHT_BYTES},"runtime":runtime,
            "preparation_sha256":A.file_sha(manifest),"source_row_sha256":A.digest(r),"item_id":item_id,
            "source_surface_sha256":A.file_sha(surface),"conditioning_image_sha256":A.file_sha(conditioning),
            "mesh_sha256":r["geometry_target"]["mesh"]["sha256"],"split":r["split"],"source_group_id":r["provenance"]["group_id"],
            "actual_mesh_surface_recomputed_equal":True,
            "native_train_ready":False,"training_started":False,"generation_query_chart_ready":False,
            "target_query_xyz_generation_input_allowed":False,"input_surface_shape":list(arr.shape),
            "surface_policy":G.SURFACE_POLICY,"native_surface_loader_parity":False,"device":"cpu",
            "cpu_seed_policy":"Torch.manual_seed + Linux libc.srand; genuine fps random_start algorithm unchanged",
            "blockers":errors,"encode_executed":False}
    if errors and execute:
        raise ValueError("genuine native CPU dependency unavailable: "+"; ".join(errors))
    if execute:
        import torch
        import torch_cluster
        from safetensors import safe_open
        from safetensors.torch import save_file
        if not 1<=threads<=16:
            raise ValueError("bounded CPU threads required")
        torch.set_num_threads(threads)
        report["fps_dependency"]=fps_dependency(torch,torch_cluster)
        vae=native_vae(root)(**params).to(device="cpu",dtype=torch.float32).eval().requires_grad_(False)
        expected=vae.state_dict()
        with safe_open(str(weights),framework="pt",device="cpu") as h:
            keys={k[4:] for k in h.keys() if k.startswith("vae.")}
            if keys!=set(expected) or len(keys)!=410:
                raise ValueError("exact original VAE tensor key set differs")
            state={}
            for name,ref in expected.items():
                value=h.get_tensor("vae."+name)
                if value.shape!=ref.shape:
                    raise ValueError("actual VAE tensor shape differs: "+name)
                state[name]=value.float()
        vae.load_state_dict(state,strict=True);del state,expected
        if native_surface_control:
            arr,vertices,faces,control=parent_surface(root,mesh_path,output,torch,seed,frozen=frozen)
            report.update(parent_surface_control=control,native_surface_loader_parity=True,
                          surface_policy="unchanged_original_SharpEdgeSurfaceLoader_captured_once")
        measured={}
        def observe(module,args,out):
            if list(args[0].shape)!=[1,81920,3] or list(args[1].shape)!=[1,81920,4]:
                raise ValueError("real native encoder input dimensions differ")
            measured["query_pc"]=out[1][0].detach().cpu().clone()
        hook=vae.encoder.register_forward_hook(observe)
        x=torch.from_numpy(arr.copy());started=time.monotonic()
        with torch.inference_mode():
            reset_cpu_rng(torch,seed);a=vae.encode(x,sample_posterior=False)
            qa=measured["query_pc"].clone()
            reset_cpu_rng(torch,seed);b=vae.encode(x,sample_posterior=False)
            qb=measured["query_pc"].clone()
            if list(a.shape)!=[1,4096,64] or not torch.isfinite(a).all() or not torch.equal(a,b) or not torch.equal(qa,qb):
                raise ValueError("real deterministic encoder mode/seed/query proof failed")
            modelspace=a*float(vae.scale_factor)
            inverse=modelspace/float(vae.scale_factor)
            if not torch.allclose(inverse,a,atol=2e-7,rtol=2e-7):
                raise ValueError("actual scale_factor model-space inverse differs")
            decoded=vae.decode(inverse)
            if list(decoded.shape)!=[1,4096,1024] or not torch.isfinite(decoded).all():
                raise ValueError("actual post-KL/transformer decoder failed")
            queries,inside=convex_oracle(vertices,faces)
            query_tensor=torch.from_numpy(queries).unsqueeze(0)
            logits=vae.geo_decoder(queries=query_tensor,latents=decoded)
            if list(logits.shape)!=[1,len(queries),1] or not torch.isfinite(logits).all():
                raise ValueError("genuine native geometry decoder query failed")
            report["oracle_confusion"]=oracle_metrics(logits[0,:,0].numpy(),inside)
            if dense_resolution and not dense_sample_only:
                report["mode_dense_reconstruction"]=dense_reconstruction(vae,decoded,vertices,faces,output,"mode",dense_resolution,seed)
            additional={}
            if compare_posterior_sample:
                reset_cpu_rng(torch,seed)
                sampled=vae.encode(x,sample_posterior=True)
                sample_query=measured["query_pc"].clone()
                reset_cpu_rng(torch,seed)
                repeated_sample=vae.encode(x,sample_posterior=True)
                if not torch.equal(sampled,repeated_sample) or not torch.equal(sample_query,measured["query_pc"]):
                    raise ValueError("native prescribed posterior.sample/query repeat failed")
                sample_decoded=vae.decode(sampled)
                sample_logits=vae.geo_decoder(queries=query_tensor,latents=sample_decoded)
                if not torch.isfinite(sampled).all() or not torch.isfinite(sample_logits).all():
                    raise ValueError("native posterior.sample produced nonfinite output")
                additional={"posterior_sample_raw_latents":sampled.contiguous(),"posterior_sample_geometry_logits":sample_logits.contiguous()}
                report["prescribed_posterior_sample_control"]={"sample_seed":seed,"samples_compared":1,"best_seed_selection":False,
                    "encoder_mode":"unchanged native posterior.sample()", "repeated_sample_bitwise_equal":True,
                    "repeated_sample_query_xyz_bitwise_equal":True,"oracle_confusion":oracle_metrics(sample_logits[0,:,0].numpy(),inside)}
                if dense_resolution:
                    report["prescribed_posterior_sample_control"]["dense_reconstruction"]=dense_reconstruction(
                        vae,sample_decoded,vertices,faces,output,"sample",dense_resolution,seed)
        hook.remove()
        tensorfile=output/"target_latents.safetensors"
        save_file({"raw_latents":a.contiguous(),"model_space_latents":modelspace.contiguous(),
                   "target_only_encoder_query_xyz":qa.contiguous(),"decoded_features":decoded.contiguous(),
                   "physical_oracle_queries":query_tensor.contiguous(),"native_geometry_logits":logits.contiguous(),
                   "physical_oracle_inside":torch.from_numpy(inside).contiguous(),**additional},str(tensorfile))
        report.update(encode_executed=True,encode_seconds=time.monotonic()-started,encoder_mode="posterior.mode()",
                      native_raw_latent_shape=list(a.shape),native_decoded_shape=list(decoded.shape),
                      repeated_encode_bitwise_equal=True,repeated_query_xyz_bitwise_equal=True,
                      model_space_policy="raw_latent * original ShapeVAE.scale_factor",scale_factor=float(vae.scale_factor),
                      model_space_inverse_verified=True,original_vae_tensors_strictly_loaded=410,
                      target_tensors={"path":str(tensorfile.resolve()),"sha256":A.file_sha(tensorfile)},
                      geometry_reconstruction_quality_verified=False,
                      bounded_geometry_oracle={"source":"independent closed convex mesh half-space containment",
                         "queries":len(queries),"inside":int(inside.sum()),"outside":int((~inside).sum()),
                         "positive_logit_inside_accuracy":float(((logits[0,:,0].numpy()>0)==inside).mean()),
                         "negative_logit_inside_accuracy":float(((logits[0,:,0].numpy()<0)==inside).mean()),
                         "parent_logit_sign_not_inferred":True,"dense_reconstructed_mesh_not_verified":True},
                      blockers=["Actual geometric reconstruction metrics remain unverified.",
                                "Genuine source-bound child image conditioning and target-independent native query-chart amendment remain required."])
    report["receipt_sha256"]=A.digest(report)
    (output/"probe.json").write_bytes(A.canonical(report)+b"\n")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source-root","weights","prepared-manifest","surface","mesh","conditioning","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--prepared-manifest-sha256",required=True)
    p.add_argument("--item-id",required=True);p.add_argument("--execute",action="store_true")
    p.add_argument("--threads",type=int,default=8);p.add_argument("--seed",type=int,default=20261013)
    p.add_argument("--native-surface-control",action="store_true")
    p.add_argument("--dense-resolution",type=int,default=0,choices=(0,16,32,48,64))
    p.add_argument("--compare-posterior-sample",action="store_true")
    p.add_argument("--frozen-parent-surface",type=Path)
    p.add_argument("--parent-surface-receipt",type=Path)
    p.add_argument("--parent-surface-receipt-sha256")
    p.add_argument("--dense-sample-only",action="store_true")
    a=p.parse_args(argv)
    result=probe(root=a.source_root,weights=a.weights,manifest=a.prepared_manifest,manifest_sha256=a.prepared_manifest_sha256,item_id=a.item_id,
                 surface=a.surface,mesh_path=a.mesh,conditioning=a.conditioning,output=a.output,execute=a.execute,threads=a.threads,seed=a.seed,
                 native_surface_control=a.native_surface_control,dense_resolution=a.dense_resolution,compare_posterior_sample=a.compare_posterior_sample,
                 frozen_parent_surface=a.frozen_parent_surface,parent_surface_receipt=a.parent_surface_receipt,
                 parent_surface_receipt_sha256=a.parent_surface_receipt_sha256,dense_sample_only=a.dense_sample_only)
    print(json.dumps({k:result[k] for k in ("item_id","encode_executed","native_train_ready","blockers","receipt_sha256")}))


if __name__=="__main__":
    main()
