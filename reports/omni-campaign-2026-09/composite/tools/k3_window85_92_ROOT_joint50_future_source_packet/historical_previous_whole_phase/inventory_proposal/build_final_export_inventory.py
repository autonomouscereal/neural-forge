"""Actual closed92 additive export inventory; full hash and bounded finite/load.

Use only after the real sole writer is terminal and the independent closed92
seal exists. No copies, deletion, model forward, allocation or provider action.
Current source/output payload bytes are re-read serially, never overwritten.
"""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import argparse,hashlib,json,math,os,re,stat,struct,sys

def require(value,message):
    if value is not True:raise ValueError(message)
def unique(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'duplicate JSON key');result[key]=value
    return result
def identity(path):
    s=Path(path).lstat()
    return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,mode=s.st_mode)
def no_alias(path,*,directory=False):
    p=Path(path)
    require(p.is_absolute() and '..' not in p.parts,'absolute canonical path required')
    for q in [*reversed(p.parents),p]:
        mode=q.lstat().st_mode
        require(not stat.S_ISLNK(mode),'symlink path refused')
        require(stat.S_ISDIR(mode) if q!=p or directory else stat.S_ISREG(mode),'wrong physical type')
    return p
def whole(path,expected):
    p=no_alias(path);before=identity(p)
    require(re.fullmatch(r'[0-9a-f]{64}',expected or '') is not None,'complete actual SHA required')
    with p.open('rb') as h:got=hashlib.file_digest(h,'sha256').hexdigest()
    require(got==expected and identity(p)==before,'current complete payload SHA/identity changed')
    return before
def pinned_doc(ref):
    p=no_alias(ref['path']);require(p.stat().st_size<=256<<20,'bounded fullraw metadata required')
    before=whole(p,ref['sha256']);raw=p.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==ref['sha256'] and identity(p)==before,'metadata changed')
    return json.loads(raw,object_pairs_hook=unique)


def header(path):
    with Path(path).open("rb") as handle:
        lead = handle.read(8)
        require(len(lead) == 8, "truncated safetensors header")
        length = struct.unpack("<Q", lead)[0]
        require(2 <= length <= 16 << 20 and 8 + length <= Path(path).stat().st_size, "invalid safetensors header length")
        result = json.loads(handle.read(length), object_pairs_hook=unique)
    require(isinstance(result, dict), "safetensors header object required")
    return result

def native_ffn_shapes(width, cfg):
    latent, hidden = cfg["routed_expert_hidden_size"], cfg["hidden_size"]
    shared = cfg["num_shared_experts"] * cfg["moe_intermediate_size"]
    prefix = "block_sparse_moe."
    return {"w1_s": [width, latent], "w3_s": [width, latent], "w2p": [latent, width], "channels": [width, 2],
            prefix + "routed_expert_down_proj.weight": [latent, hidden],
            prefix + "routed_expert_up_proj.weight": [hidden, latent],
            prefix + "routed_expert_norm.weight": [latent],
            prefix + "shared_experts.gate_proj.weight": [shared, hidden],
            prefix + "shared_experts.up_proj.weight": [shared, hidden],
            prefix + "shared_experts.down_proj.weight": [hidden, shared]}

def finite_library_readback(path, expected_shapes=None, expected_meta=None, expected_dtypes=None):
    """Read every stored coordinate in bounded actual safe_open CPU slices."""
    import torch
    from safetensors import safe_open

    path = Path(path)
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    hdr = header(path)
    metadata = hdr.pop("__metadata__", {})
    require(isinstance(metadata, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()),
            "invalid safetensors metadata")
    if expected_shapes is not None:
        require(set(hdr) == set(expected_shapes), "exact endpoint tensor keys required")
        if expected_dtypes is None:
            require(len({record.get("dtype") for record in hdr.values()}) == 1, "endpoint tensor dtypes disagree")
    if expected_meta is not None:
        for key, value in expected_meta.items():
            require(metadata.get(key) == value, "endpoint metadata mismatch: " + key)
    records = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        require(set(handle.keys()) == set(hdr), "actual library/header tensor set mismatch")
        for name, description in hdr.items():
            dimensions = description["shape"]
            require(isinstance(dimensions, list) and all(type(x) is int and x >= 0 for x in dimensions), "invalid tensor shape")
            if expected_shapes is not None:
                require(dimensions == expected_shapes[name], "endpoint shape mismatch: " + name)
                if expected_dtypes is None:
                    require(description["dtype"] in ("F16", "BF16", "F32"), "native endpoint float dtype required")
                else:
                    require(description["dtype"] == expected_dtypes[name], "native saved-output dtype mismatch: " + name)
            view = handle.get_slice(name)
            require(view.get_shape() == dimensions, "actual library tensor shape mismatch")
            # One first-axis slice stays near 4 MiB; no full residual stack is loaded.
            trailing = math.prod(dimensions[1:]) if dimensions else 1
            rows = max(1, (1 << 20) // max(1, trailing))
            count, nonfinite = 0, 0
            for start in range(0, dimensions[0] if dimensions else 1, rows):
                tensor = view[start:min(start + rows, dimensions[0])] if dimensions else handle.get_tensor(name)
                count += tensor.numel()
                if tensor.is_floating_point():
                    nonfinite += int((~torch.isfinite(tensor)).sum().item())
                del tensor
            require(count == math.prod(dimensions) and nonfinite == 0, "actual nonfinite or incomplete tensor: " + name)
            records[name] = {"shape": dimensions, "dtype": description["dtype"], "elements": count, "nonfinite": nonfinite}
    require(before == (path.stat().st_size, path.stat().st_mtime_ns), "file changed during actual library read")
    return {"path": str(path), "bytes": before[0], "tensors": records,
            "actual_library_load_verified": True, "all_coordinates_finite": True,
            "reader": "safetensors.safe_open framework=pt CPU slices; no full model load"}

def export(request_path,request_sha,out):
    request=pinned_doc({'path':str(request_path),'sha256':request_sha})
    require(request.get('schema')=='neural_forge.k3_actual_closed92_export_inventory_request.v1'
            and request.get('writer_terminal_verified') is True,'actual closed92 terminal request required')
    closed=pinned_doc(request['closed92_continuity'])
    require(closed.get('schema')=='neural_forge.k3_actual_window_continuity.v1'
            and closed.get('completed_window')=='85-92' and closed.get('full_prefix_layers')==92
            and closed.get('actual_layers_processed_this_terminal_run')==list(range(85,93))
            and closed.get('ffn_full_scan_complete') is True
            and closed.get('capability_accepted') is False and closed.get('training_eligible') is False,
            'actual full closed92 structural scanner/readback seal required')
    controller=pinned_doc(request['controller_terminal_result'])
    require(controller.get('stage')=='bounded_window_complete' and controller.get('full_prefix_layers')==92
            and controller.get('final_continuity_sha256')==request['closed92_continuity']['sha256'],
            'actual sole controller terminal must name this exact closed92 seal')
    execution=pinned_doc(request['fitter_execution'])
    require(execution.get('phase')=='native_fitter' and execution.get('exit_code')==0 and type(execution.get('pid')) is int and execution['pid']>0
            and execution.get('argv',[]).count('--layers')==1
            and execution['argv'][execution['argv'].index('--layers')+1]=='85-92'
            and execution.get('argv',[]).count('--acceptance')==1
            and execution['argv'][execution['argv'].index('--acceptance')+1]=='record',
            'genuine terminal fitter record-mode command/exit required')
    terminal=pinned_doc(request['terminal_receipt'])
    require(request['terminal_receipt']['sha256']==closed['receipt_sha256']
            and terminal.get('resume_context')==closed['resume_context']
            and closed['resume_context']['software']['tool_sha256']=='655e10cb06105a7933231a958e679d5e45a8b45ae1d9fb89a1e7940fd464f61b'
            and terminal.get('layers_processed_this_run')==list(range(85,93))
            and terminal.get('layers_folded')==list(range(1,93))
            and terminal.get('refusal') is None and terminal.get('end_of_stack') is not None,
            'actual full native terminal92/source/prefix/head diagnostics required')
    catalog=pinned_doc(request['source_catalog']);observation=pinned_doc(request['source_observation'])
    require(observation.get('source_catalog_sha256')==request['source_catalog']['sha256']
            and observation.get('first_new_layer')==85 and observation.get('last_new_layer')==92,
            'exact actual85 source catalog/observation required')
    origins=observation['source_roots']
    require(set(origins)=={'legacy_closed36','closed37_48','closed49_60','closed61_72','closed73_84'},
            'all five actual immutable old source origins required')
    rows={}
    for row in catalog['input_catalog']:
        logical=row.get('recorded_logical_path') or '/run-output/'+row['relative_path']
        require(logical not in rows and row['source_origin'] in origins,'unique old source origin/path required');rows[logical]=row
    new_root=no_alias(observation['output_root']['host_path'],directory=True)
    old_roots={name:no_alias(row['host_path'],directory=True) for name,row in origins.items()}
    for name,p in old_roots.items():
        require(origins[name]['read_only'] is True and p!=new_root and not p.is_relative_to(new_root)
                and not new_root.is_relative_to(p),'exact disjoint retained RO input roots required')
    root_stamps={str(p):identity(p) for p in [*old_roots.values(),new_root]}
    def physical(logical,expected):
        pp=PurePosixPath(logical)
        require(pp.is_relative_to('/run-output') and str(pp)==logical and '..' not in pp.parts,'canonical exact logical path required')
        rel=pp.relative_to('/run-output')
        if logical in rows:
            row=rows[logical];require(row['sha256']==expected,'old semantic source SHA changed')
            root=old_roots[row['source_origin']];p=root/rel
            require(str(p)==row['unchanged_host_input_path'],'immutable exact physical origin map differs')
        else:
            root=new_root;p=root/rel
            require(re.fullmatch(r'(?:fold/dense/L(?:8[5-9]|9[0-2])/(?:ffn.safetensors|done.json)|'
                r'body/L(?:8[5-9]|9[0-2])/body.safetensors|'
                r'fold/review/(?:k3_review_input_ids|k3_review_native_input_ids)_teacher_forced.safetensors)',str(rel)) is not None,
                'unlisted new terminal core artifact refused')
        no_alias(p);require(p.lstat().st_dev==root.lstat().st_dev,'output/source device alias differs')
        return p
    require(request['parent_config']['sha256']=='9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213'
            and closed['resume_context']['model']['config_sha256']==request['parent_config']['sha256'],
            'exact original accepted parent configuration required')
    cfg=pinned_doc(request['parent_config']);cfg=cfg.get('text_config',cfg)
    require(cfg['num_hidden_layers']==93 and cfg['first_k_dense_replace']==1,'exact93 layer architecture required')
    prefix=closed['prefix'];require(len(prefix)==92 and [r['layer'] for r in prefix]==list(range(1,93)),'actual full92 ordered prefix required')
    preserved={r['path']:r for r in closed['preserved_original_body_globals_L00']}
    specifications=[]
    for relative,role in [('fold/dense/L00/mlp.safetensors','original_L00_dense'),('body/L00/body.safetensors','original_L00_body'),
                          ('body/model_level.safetensors','original_model_globals')]:
        require(relative in preserved,'original L00/globals artifact missing')
        specifications.append(('/run-output/'+relative,preserved[relative]['sha256'],role,None))
    for row in prefix:
        layer=row['layer'];specifications.append((f'/run-output/fold/dense/L{layer:02d}/ffn.safetensors',row['ffn_sha256'],'native_MoE_FFN',layer))
    body=closed['body'];require([r['layer'] for r in body]==list(range(1,93)) and all(r['byte_exact'] is True for r in body),'all92 byte-exact native body rows required')
    for row in body:specifications.append((f"/run-output/body/L{row['layer']:02d}/body.safetensors",row['sha256'],'native_verbatim_body',row['layer']))
    reviews=terminal['end_of_stack']['review']['files']
    require(set(reviews)=={'k3_review_input_ids','k3_review_native_input_ids'},'both actual saved review outputs required')
    for name,row in reviews.items():specifications.append((row['path'],row['sha256'],'saved_native_teacher_forced_head_diagnostic',None))
    require(len(specifications)==189 and len({p for p,*_ in specifications})==189,'exact93 FFNs/dense+93 bodies+global+2reviews required')
    artifacts=[];dtype_counts={};coefficients=0
    for logical,expected,role,layer in specifications:
        path=physical(logical,expected);stamp=whole(path,expected);hdr=header(path);shapes=None;dtypes=None
        if role=='native_MoE_FFN':
            meta=hdr.get('__metadata__',{});width=int(meta['width_installed'])
            require(width>0 and int(meta['layer'])==layer and int(meta['K'])==17408,'actual native FFN metadata/accounting required')
            shapes=native_ffn_shapes(width,cfg)
            dtypes={name:'I32' if name=='channels' else 'F32' if name=='w2p' else 'BF16' for name in shapes}
        readback=finite_library_readback(path,shapes,expected_dtypes=dtypes)
        require(identity(path)==stamp,'artifact changed during serial finite/library readback')
        elements=sum(row['elements'] for row in readback['tensors'].values())
        float_count=sum(row['elements'] for row in readback['tensors'].values() if row['dtype'].startswith('F') or row['dtype']=='BF16')
        for row in readback['tensors'].values():dtype_counts[row['dtype']]=dtype_counts.get(row['dtype'],0)+row['elements']
        coefficients+=float_count
        artifacts.append({'logical_path':logical,'physical_path':str(path),'sha256':expected,'bytes':stamp['size'],
            'role':role,'layer':layer,'tensor_count':len(readback['tensors']),'stored_elements':elements,
            'floating_coefficients':float_count,'full_library_finite_readback':readback,'actual_six_field_identity':stamp})
    markers=[]
    done0='fold/dense/L00/done.json';require(done0 in preserved,'original L00 marker required')
    for layer,expected in [(0,preserved[done0]['sha256'])]+[(r['layer'],r['done_sha256']) for r in prefix]:
        logical=f'/run-output/fold/dense/L{layer:02d}/done.json';path=physical(logical,expected);stamp=whole(path,expected)
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==expected and identity(path)==stamp,'marker changed after complete SHA')
        marker=json.loads(raw,object_pairs_hook=unique)
        require(marker.get('layer')==layer and type(marker.get('layer')) is int,'actual marker layer identity differs')
        markers.append({'layer':layer,'logical_path':logical,'physical_path':str(path),'sha256':expected,'bytes':stamp['size']})
    require(all(identity(Path(p))==s for p,s in root_stamps.items()),'source/output root changed during final inventory')
    result={'schema':'neural_forge.k3_actual_complete93_layer_additive_export_inventory.v1',
        'at_utc':datetime.now(timezone.utc).isoformat(),'request_sha256':request_sha,
        'closed92_sha256':request['closed92_continuity']['sha256'],'actual_layers_including_L00':list(range(93)),
        'actual_MoE_layers':list(range(1,93)),'artifacts':artifacts,'markers':markers,
        'artifact_count':189,'marker_count':93,'total_artifact_bytes':sum(r['bytes'] for r in artifacts),
        'total_stored_floating_coefficients':coefficients,'stored_element_counts_by_dtype':dtype_counts,
        'exact_source_roots':origins,'exact_terminal_output_root':observation['output_root'],
        'terminal_native_diagnostic_receipt':request['terminal_receipt'],
        'all_current_payload_full_SHA_verified':True,'all_stored_coordinates_library_read_and_finite':True,
        'weights_copied_or_relocated':False,'historical_artifacts_modified':False,
        'whole_child_free_run_or_language_tool_media_capability_checked':False,
        'native_end_of_stack_is_teacher_forced_diagnostic':True,'capability_accepted':False,'training_eligible':False,
        'numerical_target_acceptance_required':False,'export_scope':'immutable-path inventory only; no model publication or source-stack assembly claim'}
    require(out.is_absolute() and '..' not in out.parts,'absolute canonical additive inventory output required')
    no_alias(out.parent,directory=True);require(not out.exists() and not out.is_symlink(),'exclusive additive inventory path required')
    require(all(not out.is_relative_to(p) for p in [*old_roots.values(),new_root]),'metadata output must be outside all retained model roots')
    encoded=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with out.open('xb') as h:h.write(encoded);h.flush();os.fsync(h.fileno())
    require(out.read_bytes()==encoded,'actual complete inventory readback differs')
    return {'path':str(out),'sha256':hashlib.sha256(encoded).hexdigest(),'bytes':len(encoded),'artifacts':189,'markers':93}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    try:print(json.dumps(export(args.request,args.request_sha256,args.out),sort_keys=True))
    except (ValueError,OSError,KeyError,TypeError,RuntimeError,ImportError) as error:parser.exit(2,'actual final inventory refused: '+str(error)+'\n')
if __name__=='__main__':main()
