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
    p=manifest.parent/Path(ref["path"]).name
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


def probe(*,root,weights,manifest,manifest_sha256,item_id,surface,mesh_path,conditioning,output,execute=False,threads=8,seed=20261013):
    if output.exists():
        raise ValueError("choose a new diagnostic output directory")
    pins=source_pins(root)
    if not A.SHA.fullmatch(str(manifest_sha256)) or A.file_sha(manifest)!=manifest_sha256:
        raise ValueError("immutable geometry preparation file pin differs")
    r,arr,prep=source_item(manifest,item_id,surface,conditioning)
    vertices,faces=actual_mesh_surface(r,arr,mesh_path)
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
        hook.remove()
        output.mkdir(parents=True)
        tensorfile=output/"target_latents.safetensors"
        save_file({"raw_latents":a.contiguous(),"model_space_latents":modelspace.contiguous(),
                   "target_only_encoder_query_xyz":qa.contiguous(),"decoded_features":decoded.contiguous(),
                   "physical_oracle_queries":query_tensor.contiguous(),"native_geometry_logits":logits.contiguous(),
                   "physical_oracle_inside":torch.from_numpy(inside).contiguous()},str(tensorfile))
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
    else:
        output.mkdir(parents=True)
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
    a=p.parse_args(argv)
    result=probe(root=a.source_root,weights=a.weights,manifest=a.prepared_manifest,manifest_sha256=a.prepared_manifest_sha256,item_id=a.item_id,
                 surface=a.surface,mesh_path=a.mesh,conditioning=a.conditioning,output=a.output,execute=a.execute,threads=a.threads,seed=a.seed)
    print(json.dumps({k:result[k] for k in ("item_id","encode_executed","native_train_ready","blockers","receipt_sha256")}))


if __name__=="__main__":
    main()
