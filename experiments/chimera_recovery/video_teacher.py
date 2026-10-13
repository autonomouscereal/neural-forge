"""Pinned temporal ViT13 teacher capture; no language trunk or synthetic targets.

The feature teacher is the actual available Qwen3.8 visual tower. Its lineage is
explicitly separate from the old QVL32 still-image cache. GPU language-answer
capture remains the original full teacher under the root allocator.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import time
import numpy as np

try:
    from . import video_cache as V
except ImportError:
    import video_cache as V
A = V.A
CONFIG_SHA = "4c95af1d45132ead5b63e540f08f32a351e20512e62438a3a78faf9881d1e7dc"
INDEX_SHA = "67dbe83b6c2c36225793a8b56b4ae79a318c882136142f58636ca388e26e623a"
MODEL_SOURCE_PINS = {
    "transformers.models.qwen3_5.modeling_qwen3_5":"395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6",
    "transformers.models.qwen3_5.configuration_qwen3_5":"f9d6330bc658363330167f8a8943dff48f040566e1ab0f296eddb2ffb524d367"}


def json_pin(path, expected, schema):
    p = A.pinned({"path":str(path),"sha256":expected},schema)
    r = json.loads(p.read_text())
    if (r.get("schema") != schema or r.get("sealed") is not True or
            A.digest({k:v for k,v in r.items() if k != "manifest_sha256"}) != r.get("manifest_sha256")):
        raise ValueError("sealed canonical manifest differs: "+schema)
    return r


def load_inputs(ingress, ingress_sha256, receipts):
    manifest = json_pin(ingress,ingress_sha256,V.SCHEMA)
    spec = manifest["outputs"]["receipts"]
    if (A.file_sha(receipts) != spec["sha256"] or manifest.get("all_frames_retained") is not True or
            manifest.get("model_weights_loaded") is not False or manifest.get("teacher_targets_generated") is not False):
        raise ValueError("actual complete CPU temporal replay required")
    rows = list(A.rows(receipts))
    if (len(rows) != 24 or manifest["rows"] != len(rows) or len({r['id'] for r in rows}) != len(rows) or
            any(r["all_eight_frames_retained"] is not True or r["parent_prefix_full_ids_reproduced"] is not True or
                r["metadata"]["frames_indices"] != list(range(8)) or r["grid_thw"] != [4,6,6] for r in rows)):
        raise ValueError("unchanged complete24 full-frame temporal replay required")
    return manifest,rows


def join_global(*, ingress, ingress_sha256, receipts, admission, admission_sha256, output):
    """Separate successor join; original processor receipt bytes stay immutable."""
    if output.exists():raise ValueError("fresh global-join output required")
    _,rows = load_inputs(ingress,ingress_sha256,receipts)
    manifest = json_pin(admission,admission_sha256,"recovery_audio_admission.v1")
    if any(manifest.get(k) is not True for k in ("complete_source_traversal","frozen_exclusions_verified","connected_groups_verified")):
        raise ValueError("complete frozen/group admission required")
    for name,spec in manifest["sources"].items():A.pinned(spec,"global input "+name)
    sources = {r["id"]:r for r in A.rows(A.pinned(manifest["sources"]["source"],"source rows"))}
    admitted = {r["id"]:r for r in A.rows(A.pinned(manifest["outputs"]["admission"],"source admission rows"))}
    result = []
    for r in rows:
        source,decision = sources[r["id"]],admitted[r["id"]]
        if (A.digest(source) != r["source_row_sha256"] or decision["source_row_sha256"] != r["source_row_sha256"] or
                source["route"] != "video_understand" or source["split"] != r["split"] or
                decision["asset_sha256"] != r["asset_sha256"] or decision["split"] != r["split"] or
                (decision["train_admitted"] and (r["split"] != "train" or decision["blockers"]))):
            raise ValueError("same video source/asset/split/frozen decision required")
        result.append({"id":r["id"],"source_row_sha256":r["source_row_sha256"],"asset_sha256":r["asset_sha256"],
            "split":r["split"],"recovery_group_id":decision["recovery_group_id"],"train_admitted":decision["train_admitted"],
            "blockers":decision["blockers"],"held":r["split"] != "train","source_route":"video_understand"})
    if len([r for r in result if r["train_admitted"]]) != 8 or len([r for r in result if r["held"]]) != 16:
        raise ValueError("expected8 train/16 held procedural interface scope differs")
    value = {"schema":"recovery_video_global_join.v1","sealed":True,"source_year":2026,
        "source_ingress":V.pin(ingress),"source_receipts":V.pin(receipts),"global_admission":V.pin(admission),
        "global_sources":manifest["sources"],"global_rows":manifest["outputs"]["admission"],"records":result,
        "source_train_eligible":8,"held":16,"full_candidates":len(admitted),"native_train_ready":False,
        "scope":"eight procedural training videos are an interface/regression baseline, insufficient for general video recovery",
        "code_sha256":A.file_sha(Path(__file__))}
    value["manifest_sha256"] = A.digest(value);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(A.canonical(value)+b"\n");return value


def source_catalog(model_dir):
    if A.file_sha(model_dir/"config.json") != CONFIG_SHA or A.file_sha(model_dir/"model.safetensors.index.json") != INDEX_SHA:
        raise ValueError("captured actual parent config/index changed")
    config = json.loads((model_dir/"config.json").read_text())["vision_config"]
    keys = ("model_type","depth","hidden_size","num_heads","patch_size","temporal_patch_size","spatial_merge_size","out_hidden_size")
    if tuple(config[k] for k in keys) != ("qwen3_5_vision",27,1152,16,16,2,2,5120):
        raise ValueError("captured actual visual architecture differs")
    index = json.loads((model_dir/"model.safetensors.index.json").read_text())["weight_map"]
    names = {k:v for k,v in index.items() if k.startswith("model.visual.")}
    if len(names) != 333 or set(names.values()) != {"model-00012-of-00012.safetensors"}:
        raise ValueError("captured actual333 visual tensor catalog differs")
    return names


def teacher_files(model_dir, names):
    return {name:{"sha256":A.file_sha(model_dir/name),"bytes":(model_dir/name).stat().st_size}
        for name in sorted({"config.json","model.safetensors.index.json",*names.values()})}


def shard_path(spec, mirror):
    if mirror is None:return A.pinned(spec,"authentic original CPU ingress")
    # Explicit relocation of one previously authenticated output/shards directory.
    # Every file still must match its original byte seal; this never rewrites it.
    p=mirror/Path(spec["path"]).name
    return A.pinned({"path":str(p),"sha256":spec["sha256"]},"explicit byte-identical ingress mirror")


def feature_cache(*, ingress, ingress_sha256, receipts, global_join, global_join_sha256,
                  model_dir, output, split, mirror=None):
    if output.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("fresh output and hidden CUDA required; this writer is CPU only")
    _,rows = load_inputs(ingress,ingress_sha256,receipts)
    join = json_pin(global_join,global_join_sha256,"recovery_video_global_join.v1")
    if join["source_ingress"]["sha256"] != ingress_sha256 or join["source_receipts"]["sha256"] != A.file_sha(receipts):
        raise ValueError("global decision identifies different temporal replay")
    decisions={r["id"]:r for r in join["records"]}
    if len(decisions) != 24:raise ValueError("full24 global decision required")
    selected=[r for r in rows if r['split']==split]
    if len(selected)!=8 or any((split=="train" and not decisions[r['id']]['train_admitted']) or
                              r['source_row_sha256']!=decisions[r['id']]['source_row_sha256'] or
                              r['asset_sha256']!=decisions[r['id']]['asset_sha256'] for r in selected):
        raise ValueError("source-admitted exact split8 required")
    # Validate every small source ingress before constructing any parent weights.
    paths={r['id']:shard_path(r['ingress'],mirror) for r in selected}
    names=source_catalog(model_dir)
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModel
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    if torch.cuda.is_initialized():raise ValueError("CPU capture cannot initialize CUDA")
    torch.set_num_threads(min(8,torch.get_num_threads()))
    torch.set_num_interop_threads(1)
    for name,expected in {**V.SOURCE_PINS,**MODEL_SOURCE_PINS}.items():
        if A.file_sha(importlib.import_module(name).__file__) != expected:
            raise ValueError("captured actual parent source differs")
    before=teacher_files(model_dir,names);started=time.perf_counter()
    config=Qwen3_5Config.from_pretrained(str(model_dir)).vision_config;config._attn_implementation="sdpa"
    model=AutoModel.from_config(config).to(device="cpu",dtype=torch.float32).eval()
    state={}
    with safe_open(str(model_dir/next(iter(names.values()))),framework="pt",device="cpu") as handle:
        for name in sorted(names):state[name[len("model.visual."):]]=handle.get_tensor(name)
    model.load_state_dict(state,strict=True,assign=False);del state
    for parameter in model.parameters():parameter.requires_grad_(False)
    if len(model.blocks)!=27:raise ValueError("actual visual tower layer count differs")
    # Bitwise parameter seals check in-process state independently of source files.
    def state_sha():
        import hashlib
        h=hashlib.sha256()
        for name,tensor in sorted(model.state_dict().items()):
            h.update(A.canonical([name,list(tensor.shape),str(tensor.dtype)]));h.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    state_before=state_sha();block=model.blocks[13];captured={}
    def pre(_module,args,kwargs):
        captured.update(x=args[0].detach().clone(),cu=kwargs['cu_seqlens'].detach().clone(),
            rope=tuple(x.detach().clone() for x in kwargs['position_embeddings']))
    def post(_module,_args,_kwargs,out):captured['y']=out.detach().clone()
    hooks=[block.register_forward_pre_hook(pre,with_kwargs=True),block.register_forward_hook(post,with_kwargs=True)]
    output.mkdir(parents=True);(output/'shards').mkdir();records=[]
    try:
        for r in selected:
            with np.load(paths[r['id']],allow_pickle=False) as data:
                arrays={k:np.array(data[k],copy=True) for k in data.files}
            flat=torch.from_numpy(arrays['patch_flat']);grid=torch.from_numpy(arrays['grid_thw'])
            if flat.shape!=(144,1536) or grid.tolist()!=[[4,6,6]] or not torch.isfinite(flat).all():
                raise ValueError("actual complete temporal ingress differs")
            with torch.inference_mode():
                model(flat,grid);first={k:(tuple(x.clone() for x in v) if isinstance(v,tuple) else v.clone()) for k,v in captured.items()}
                model(flat,grid);repeated=all(torch.equal(first[k],captured[k]) for k in ('x','y','cu')) and all(torch.equal(x,y) for x,y in zip(first['rope'],captured['rope']))
                replay=block(first['x'],cu_seqlens=first['cu'],position_embeddings=first['rope'])
            if (not repeated or not torch.equal(replay,first['y']) or first['x'].shape!=(144,1152) or
                    not torch.equal(first['cu'],torch.from_numpy(arrays['teacher_cu_seqlens']))):
                raise ValueError("actual repeated tower/block replay or temporal segment contract differs")
            tensors={'block_in':first['x'],'block_out':first['y'],'cu_seqlens':first['cu'],
                'rope_cos':first['rope'][0],'rope_sin':first['rope'][1],
                **{k:torch.from_numpy(arrays[k]) for k in ('patch_flat','grid_thw','tyx','native_media_positions','paired_timestamps_seconds')}}
            if any(not torch.isfinite(t).all() for t in tensors.values()):raise ValueError("nonfinite authentic feature target")
            path=output/'shards'/(r['id']+'.safetensors');save_file(tensors,str(path))
            records.append({**V.pin(path),'source_id':r['id'],'source_row_sha256':r['source_row_sha256'],
                'asset_sha256':r['asset_sha256'],'split':split,'held':split!='train','source_ingress':r['ingress'],
                'global_group_id':decisions[r['id']]['recovery_group_id'],'repeated_full_tower_bitwise':True,'captured_block_replay_bitwise':True,
                'feature_rows':144,'feature_width':1152})
    finally:
        for hook in hooks:hook.remove()
    after=teacher_files(model_dir,names);state_after=state_sha()
    if before!=after or state_before!=state_after:raise ValueError("actual feature teacher bytes/state moved")
    result={'schema':'recovery_video_vit13_features.v1','sealed':True,'source_year':2026,'block':13,'feature_width':1152,
        'source_pins':V.SOURCE_PINS,'vision_model_source_pins':MODEL_SOURCE_PINS,'model_forward_executed':True,
        'teacher_lineage':'actual Qwen3.8-27B-Uncensored visual tower only; separate from historical QVL32 still-image cache',
        'teacher_integrity':{'mode':'full','scope':'all complete files containing every visual tensor plus config/index; no language trunk loaded',
            'before':before,'after':after,'unchanged':True,'model_state_before':state_before,'model_state_after':state_after},
        'teacher_weights_unchanged':True,'all_finite':True,'dtype':'float32','attention':'sdpa','device':'cpu','threads':torch.get_num_threads(),
        'source_ingress':V.pin(ingress),'source_receipts':V.pin(receipts),'global_join':V.pin(global_join),'shards':records,
        'parameter_count':sum(p.numel() for p in model.parameters()),'elapsed_seconds':time.perf_counter()-started,
        'language_answer_targets_generated':False,'native_train_ready':False,'training_started':False,'code_sha256':A.file_sha(Path(__file__))}
    result['manifest_sha256']=A.digest(result);(output/'manifest.json').write_bytes(A.canonical(result)+b'\n');return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('join','features'):
        s=sub.add_parser(name)
        for field in ('ingress','receipts','output'):s.add_argument('--'+field,type=Path,required=True)
        s.add_argument('--ingress-sha256',required=True)
        if name=='join':
            s.add_argument('--admission',type=Path,required=True);s.add_argument('--admission-sha256',required=True)
        else:
            for field in ('global-join','model-dir'):s.add_argument('--'+field,type=Path,required=True)
            s.add_argument('--global-join-sha256',required=True);s.add_argument('--split',choices=('train','validation','test'),required=True)
            s.add_argument('--mirror',type=Path)
    args=vars(p.parse_args(argv));command=args.pop('command')
    result=join_global(**args) if command=='join' else feature_cache(**args)
    print(json.dumps({'schema':result['schema'],'native_train_ready':False}));return 0


if __name__=='__main__':raise SystemExit(main())
