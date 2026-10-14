"""Independent actual-video cache review and strict native consumer binding.

Raw cache/ViT13 inspection is useful while the stock control runs, but its
outputs explicitly remain optimizer-ineligible until that real control passes.
Original capture files and their stale FFmpeg-description field stay unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np
try:
    from . import video_answer as B
    from . import video_quantized_answer as Q
except ImportError:
    import video_answer as B
    import video_quantized_answer as Q
V, T, A = B.V, B.T, B.A
ANSWER_SOURCE_SHA = "055dc61f1593c2dd62e909b5078e0ef236aabeea9d8961c40c8df97222fb7726"
POLICY = "temporal_video_native_answer_feature_binding_2026-10-13.v1"


def json_pin(path, sha):
    return json.loads(A.pinned({"path":str(path),"sha256":sha},"video native source").read_text())


def mirror_file(entry, mirror):
    if mirror is None:
        return A.pinned(entry,"immutable source shard")
    path=Path(mirror)/Path(entry["path"]).name
    return A.pinned({"path":str(path),"sha256":entry["sha256"]},"explicit same-byte source mirror")


def safetensors_numpy(path):
    """Bounded read of already pinned numerical feature tensors, no execution."""
    kinds={"F32":np.dtype('<f4'),"F64":np.dtype('<f8'),"I64":np.dtype('<i8'),"I32":np.dtype('<i4'),
           "I16":np.dtype('<i2'),"U8":np.dtype('u1'),"BOOL":np.dtype('?')}
    payload=Path(path).read_bytes()
    if len(payload)<8:raise ValueError("truncated feature tensor header")
    size=struct.unpack('<Q',payload[:8])[0]
    if size>65536 or 8+size>len(payload):raise ValueError("invalid bounded feature header")
    header=json.loads(payload[8:8+size]);result={};spans=[]
    for name,meta in header.items():
        if name=='__metadata__':continue
        if meta['dtype'] not in kinds:raise ValueError("unsupported actual feature dtype")
        dtype=kinds[meta['dtype']];start,end=meta['data_offsets'];shape=tuple(meta['shape'])
        if (any(type(n) is not int or n<0 for n in shape) or type(start) is not int or type(end) is not int or
                not 0<=start<=end<=len(payload)-8-size or end-start!=int(np.prod(shape))*dtype.itemsize):
            raise ValueError("feature tensor offsets/shape differ")
        spans.append((start,end));result[name]=np.frombuffer(payload,dtype=dtype,count=int(np.prod(shape)),offset=8+size+start).reshape(shape).copy()
    cursor=0
    for start,end in sorted(spans):
        if start!=cursor:raise ValueError("feature tensor holes/overlap")
        cursor=end
    if cursor!=len(payload)-8-size or any(not np.isfinite(x).all() for x in result.values()):
        raise ValueError("feature tensor trailing/nonfinite data")
    return result


def actual_decoder(capture, ingress, request):
    proof=capture.get('decoder_proof',{})
    if (proof.get('alternate') is not True or proof.get('original_ffmpeg_cli_executed') is not False or
            proof.get('package_version')!=V.PYAV_VERSION or proof.get('wheel',{}).get('sha256')!=V.PYAV_WHEEL_SHA or
            ingress.get('decoder_proof',{}).get('alternate') is not True or
            ingress['decoder_proof'].get('wheel',{}).get('sha256')!=V.PYAV_WHEEL_SHA or
            request.get('schema')!='recovery_video_teacher_request.v1' or request.get('sealed') is not True or
            A.digest({k:v for k,v in request.items() if k!='manifest_sha256'})!=request.get('manifest_sha256') or
            request.get('inputs',{}).get('decoder_wheel',{}).get('sha256')!=V.PYAV_WHEEL_SHA or
            request.get('sources',{}).get('video_answer.py',{}).get('sha256')!=ANSWER_SOURCE_SHA or
            request.get('sources',{}).get('video_capture_job.py',{}).get('sha256')!='0e625db54859ed482783960b826e4360592f69de673153e75a724031295da07d'):
        raise ValueError("actual pinned PyAV execution/ingress/request required; stale FFmpeg label is not evidence")
    return {'policy':'actual source-pinned PyAV full-frame decode; original FFmpeg metadata string retained',
            'wheel_sha256':V.PYAV_WHEEL_SHA,'package_version':V.PYAV_VERSION,'ffmpeg_execution_claimed':False,
            'prior_full_parent_processor_replay_required':True,'captured_library_versions':proof['library_versions']}


def execution_binding(path, request_sha, request_record, job):
    receipt=json.loads(path.read_text())
    if (receipt.get('schema')!='recovery_video_teacher_execution.v1' or receipt.get('job')!=job or
            receipt.get('exit_code')!=0 or receipt.get('failure') is not None or
            receipt.get('request',{}).get('sha256')!=request_sha or
            receipt.get('code_sha256')!=request_record['sources']['video_capture_job.py']['sha256'] or
            receipt.get('training_started') is not False):
        raise ValueError('actual unchanged request/entry-source completed execution required')
    return V.pin(path)


def native_arrays(ingress, feature, parent_answer, lse):
    native=Q.rebase_answer(parent_answer,ingress,lse)
    pixels=ingress['patch_flat'];positions=ingress['native_media_positions'];indices=ingress['native_media_indices']
    if (pixels.shape!=(144,1536) or feature['block_out'].shape!=(144,1152) or feature['block_in'].shape!=(144,1152) or
            not np.array_equal(feature['patch_flat'],pixels) or not np.array_equal(feature['grid_thw'],ingress['grid_thw']) or
            not np.array_equal(feature['cu_seqlens'],ingress['teacher_cu_seqlens']) or
            not np.array_equal(feature['native_media_positions'],positions) or
            not np.array_equal(feature['tyx'],ingress['tyx']) or
            not np.array_equal(feature['paired_timestamps_seconds'],ingress['paired_timestamps_seconds']) or
            ingress['grid_thw'].tolist()!=[[4,6,6]] or ingress['teacher_cu_seqlens'].tolist()!=[0,36,72,108,144] or
            positions.shape!=(144,4) or indices.shape!=(1,144) or
            not np.all(ingress['input_ids'][0,indices[0]]==V.PLACEHOLDER)):
        raise ValueError("genuine complete temporal pixels/features/positions/placeholder packing differ")
    result={'input_ids':ingress['input_ids'].copy(),'native_media_features':np.zeros((1,144,6144),np.float32),
            'native_media_type_ids':np.full((1,144),6,np.int64),'native_media_indices':indices.copy(),
            'native_media_valid_mask':np.ones((1,144),np.bool_),'native_media_positions':positions[None].copy(),
            'native_media_noise_levels':np.zeros((1,144),np.float32),'teacher_features':feature['block_out'][None].copy(),
            'assistant_token_mask':native['assistant_token_mask'].copy(),
            'video_grid_thw':ingress['grid_thw'].copy(),'video_tyx':ingress['tyx'].copy(),
            'video_teacher_cu_seqlens':ingress['teacher_cu_seqlens'].copy(),
            'video_paired_timestamps_seconds':ingress['paired_timestamps_seconds'].copy()}
    result['native_media_features'][0,:,640:2176]=pixels
    result.update({'answer.'+k:v for k,v in native.items() if k not in ('input_ids','assistant_token_mask')})
    if any(not np.isfinite(x).all() for x in result.values()):raise ValueError("nonfinite native video binding")
    return result


def review_capture(*,cache_manifest,cache_manifest_sha256,capture_receipt,capture_receipt_sha256,
                   request,request_sha256,plan,plan_sha256,ingress,ingress_sha256,receipts,global_join,global_join_sha256,
                   features,features_sha256,ingress_mirror,feature_mirror,output):
    if output.exists():raise ValueError("fresh independent video review output required")
    if A.file_sha(Path(B.__file__))!=ANSWER_SOURCE_SHA:raise ValueError("original answer converter source changed")
    raw=json_pin(cache_manifest,cache_manifest_sha256);capture=json_pin(capture_receipt,capture_receipt_sha256)
    request_record=json_pin(request,request_sha256);prepared=B.load_plan(plan,plan_sha256)
    ingress_record,rows=T.load_inputs(ingress,ingress_sha256,receipts)
    join=T.json_pin(global_join,global_join_sha256,'recovery_video_global_join.v1')
    feature_manifest=T.json_pin(features,features_sha256,'recovery_video_vit13_features.v1')
    inner=capture.get('vl_capture',{});cfg=raw.get('cache_config',{})
    if (raw.get('schema')!='omni-distill-teacher-cache-manifest-v1' or raw.get('all_finite') is not True or
            raw.get('all_structural_ok') is not True or raw.get('positions_total')!=792 or raw.get('forward_tokens_total')!=1834 or
            raw.get('tool',{}).get('sha256')!=V.WRITER_SHA or raw.get('engine',{}).get('sha256')!=B.ENGINE_SHA or
            cfg.get('top_k')!=64 or cfg.get('vocab_rows')!=V.TEXT_VOCAB or cfg.get('dtype')!='bfloat16' or
            cfg.get('head_precision')!='fp32' or capture.get('schema')!='recovery_video_answer_capture.v1' or
            capture.get('exit_code')!=0 or capture.get('assistant_only') is not True or capture.get('code_sha256')!=ANSWER_SOURCE_SHA or
            capture.get('batch_policy')!=B.BATCH_POLICY or inner.get('cache_sealed_and_verified') is not True or
            inner.get('cache_manifest_sha256')!=cache_manifest_sha256 or inner.get('writer_exitcode')!=0 or inner.get('failure') is not None or
            inner.get('split')!='train' or capture['source_plan']['sha256']!=plan_sha256 or
            capture['source_ingress']['sha256']!=ingress_sha256 or prepared['global_join']['sha256']!=global_join_sha256):
        raise ValueError("actual complete original eight-row singleton capture identity differs")
    decoder=actual_decoder(capture,ingress_record,request_record)
    execution=execution_binding(capture_receipt.parent/'execution.json',request_sha256,request_record,'capture')
    if (request_record['inputs']['ingress']['sha256']!=ingress_sha256 or request_record['inputs']['plan']['sha256']!=plan_sha256 or
            request_record['inputs']['global_join']['sha256']!=global_join_sha256 or request_record['batch_policy']!=B.BATCH_POLICY or
            request_record['actual_vit13_manifest_sha256']!=features_sha256):
        raise ValueError("actual execution request differs from selected media/target source pins")
    integrity=B.full_integrity(raw)
    if (feature_manifest.get('all_finite') is not True or feature_manifest.get('teacher_weights_unchanged') is not True or
            feature_manifest.get('model_forward_executed') is not True or feature_manifest.get('feature_width')!=1152 or
            feature_manifest.get('block')!=13 or feature_manifest.get('source_pins')!=V.SOURCE_PINS or
            feature_manifest['source_ingress']['sha256']!=ingress_sha256 or
            feature_manifest['source_receipts']['sha256']!=A.file_sha(receipts) or
            feature_manifest['global_join']['sha256']!=global_join_sha256 or
            feature_manifest['teacher_integrity'].get('unchanged') is not True or
            feature_manifest['teacher_integrity']['before']!=feature_manifest['teacher_integrity']['after']):
        raise ValueError("actual full unchanged source-aligned ViT13 feature lineage required")
    decisions={r['id']:r for r in join['records']};row_map={r['id']:r for r in rows}
    feature_map={r['source_id']:r for r in feature_manifest['shards']}
    source=next(r for r in prepared['sources'] if r['split']=='train')
    corpus=list(A.rows(B.contained(plan.parent,source['path'])))
    if len(corpus)!=8 or [r['row_id'] for r in corpus]!=source['row_ids'] or len(feature_map)!=8:
        raise ValueError("unaltered complete eight-row training corpus/order required")
    media={}
    for row in corpus:
        for message in row['messages']:
            for item in message.get('content',[]):
                if item.get('type')=='video':
                    p=B.contained(plan.parent,item['path']);A.pinned({'path':str(p),'sha256':item['sha256']},'actual video asset')
                    media[item['path']]=A.file_sha(p)
    if A.digest({'jsonl_sha256':A.file_sha(plan.parent/source['path']),'media':sorted(media.items())})!=raw['corpus']['sha256']:
        raise ValueError("actual writer corpus/media content identity differs")
    if len(raw['shards'])!=1:raise ValueError("selected singleton eight-row one-shard policy differs")
    entry=raw['shards'][0];base=cache_manifest.parent/entry['shard']
    sidecar=json_pin(base/'shard.json',entry['shard_json_sha256'])
    if (sidecar['micro_batch']!=1 or sidecar['batches']!=8 or sidecar['batch_order']!='corpus' or
            sidecar['max_batch_tokens']!=8192 or sidecar['positions']!=792 or sidecar['forward_tokens']!=1834 or
            sidecar['files']!=entry['files'] or sidecar['checks'].get('finite') is not True):
        raise ValueError("actual singleton raw shard execution policy differs")
    arrays={}
    for name,dtype,shape in [('ids.u32','<u4',(792,)),('pos.u32','<u4',(792,)),('lse.f32','<f4',(792,)),
                             ('topk_ids.i32','<i4',(792,64)),('topk_logprobs.bf16','<u2',(792,64))]:
        spec=entry['files'][name];p=A.pinned({'path':str(base/name),'sha256':spec['sha256']},'actual raw '+name)
        if p.stat().st_size!=spec['bytes']:raise ValueError("raw file byte-size mismatch")
        arrays[name]=np.fromfile(p,dtype=dtype).reshape(shape)
    if not np.isfinite(arrays['lse.f32']).all():raise ValueError("nonfinite actual LSE")
    output.mkdir(parents=True);(output/'shards').mkdir();result=[]
    cursor=0
    for table,corpus_row in zip(sidecar['row_table'],corpus):
        ident=table['row_id'];r=row_map[ident];d=decisions[ident];f=feature_map[ident]
        if (ident!=corpus_row['row_id'] or table['offset']!=cursor or table['count']!=99 or r['split']!='train' or
                d['train_admitted'] is not True or d['held'] is not False or d['blockers'] or f['held'] is not False or
                any(x['source_row_sha256']!=r['source_row_sha256'] or x['asset_sha256']!=r['asset_sha256'] for x in (d,f))):
            raise ValueError("same held-excluded global source/V13/raw answer rows required")
        source_path=T.shard_path(r['ingress'],ingress_mirror)
        with np.load(source_path,allow_pickle=False) as z:original={k:np.array(z[k],copy=True) for k in z.files}
        if table['length']!=original['parent_input_ids'].shape[1]:raise ValueError("original complete parent length differs")
        stop=cursor+table['count']
        answer,rounding=Q.dense_answer(original['parent_input_ids'],original['parent_assistant_token_mask'],
            arrays['pos.u32'][cursor:stop],arrays['ids.u32'][cursor:stop],arrays['topk_ids.i32'][cursor:stop],
            arrays['topk_logprobs.bf16'][cursor:stop],arrays['lse.f32'][cursor:stop])
        feature_path=mirror_file(f,feature_mirror);feature=safetensors_numpy(feature_path)
        native=native_arrays(original,feature,answer,arrays['lse.f32'][cursor:stop])
        path=output/'shards'/(ident+'.npz');np.savez_compressed(path,**native)
        result.append({**V.pin(path),'source_id':ident,'source_row_sha256':r['source_row_sha256'],'asset_sha256':r['asset_sha256'],
            'split':'train','held':False,'global_group_id':d['recovery_group_id'],'source_ingress':V.pin(source_path),
            'source_feature_shard':V.pin(feature_path),'raw_row_table':table,'assistant_positions':99,'media_rows':144,
            'parent_length':int(original['parent_input_ids'].shape[1]),'native_length':int(native['input_ids'].shape[1]),
            'rounding_compatibility':rounding})
        cursor=stop
    if cursor!=792 or len(result)!=8:raise ValueError("complete actual raw answer coverage required")
    value={'schema':'recovery_video_capture_review.v1','sealed':True,'source_year':2026,'policy':POLICY,
        'source_cache':V.pin(cache_manifest),'source_capture':V.pin(capture_receipt),'source_request':V.pin(request),
        'source_capture_execution':execution,
        'source_plan':V.pin(plan),'source_ingress':V.pin(ingress),'source_receipts':V.pin(receipts),'global_join':V.pin(global_join),
        'source_features':V.pin(features),'teacher_integrity':integrity,'actual_decoder':decoder,'shards':result,
        'rows':8,'assistant_positions':792,'full_frames':8,'temporal_grid':[4,6,6],'media_rows_per_example':144,
        'all_finite':True,'raw_five_file_seals_verified':True,'complete_parent_masks_verified':True,
        'feature_pixel_time_alignment_verified':True,'teacher_targets_fabricated':False,
        'stock_control_passed':False,'optimizer_use_forbidden':True,'native_train_ready':False,
        'scope':'eight procedural videos; interface/regression baseline, insufficient for general video recovery',
        'blockers':['actual matched stock-control proof pending','selected temporal-video objective/amendment and actual consumer proof required'],
        'code_sha256':A.file_sha(Path(__file__)),'quantized_answer_source_sha256':A.file_sha(Path(Q.__file__)),
        'quantized_answer_policy':Q.POLICY,'exact_teacher_probability_normalization_claimed':False}
    value['manifest_sha256']=A.digest(value);(output/'manifest.json').write_bytes(A.canonical(value)+b'\n');return value


def bind_stock_control(review,control_root,producer_sources):
    """Read the unchanged source receipts/files; no asserted parity flag suffices."""
    production=json_pin(Path(review['source_cache']['path']),review['source_cache']['sha256'])
    request=json_pin(Path(review['source_request']['path']),review['source_request']['sha256'])
    control_path=control_root/'native-run.video-control.json';proof=json.loads(control_path.read_text())
    parity_path=control_root/'native-run.json';parity=json.loads(parity_path.read_text())
    manifests=list((control_root/'cache').rglob('manifest.json'))
    if len(manifests)!=1:raise ValueError('exactly one completed control cache required')
    manifest_path=manifests[0];raw=json.loads(manifest_path.read_text())
    execution=json.loads((control_root/'execution.json').read_text())
    execution_binding(control_root/'execution.json',review['source_request']['sha256'],request,'control')
    execution_binding(Path(review['source_capture_execution']['path']),review['source_request']['sha256'],request,'capture')
    A.pinned(review['source_capture_execution'],'original capture execution seal')
    if (execution.get('schema')!='recovery_video_teacher_execution.v1' or execution.get('exit_code')!=0 or
            execution.get('failure') is not None or execution.get('job')!='control' or
            execution.get('code_sha256')!=request['sources']['video_capture_job.py']['sha256'] or
            execution.get('request',{}).get('sha256')!=review['source_request']['sha256'] or
            proof.get('schema')!='recovery_video_answer_control.v1' or proof.get('exit_code')!=0 or proof.get('split')!='train' or
            proof.get('code_sha256')!=ANSWER_SOURCE_SHA or proof.get('batch_policy')!=B.BATCH_POLICY or
            proof.get('batch_shape_invariance_claimed') is not False or
            proof['source_plan']['sha256']!=review['source_plan']['sha256'] or
            proof['source_ingress']['sha256']!=review['source_ingress']['sha256'] or
            proof['original_parity_receipt']['sha256']!=A.file_sha(parity_path) or proof['original_parity']!=parity or
            proof['streamed_control_cache']['sha256']!=A.file_sha(manifest_path) or parity.get('passed') is not True or
            parity.get('tool',{}).get('sha256')!=V.WRITER_SHA or parity.get('engine',{}).get('sha256')!=B.ENGINE_SHA or
            parity.get('manifest_sha256')!=A.file_sha(manifest_path) or parity.get('corpus_sha256')!=production['corpus']['sha256'] or
            parity.get('top_k')!=64 or parity.get('vocab_rows')!=V.TEXT_VOCAB or
            parity.get('batching')!={'micro_batch':1,'max_batch_tokens':8192,'order':'corpus','rows_per_shard':8}):
        raise ValueError('actual completed same-source singleton stock control required')
    compared=parity.get('comparison_batched_same_inputs',{})
    if (compared.get('positions')!=792 or compared.get('same_positions_and_input_ids') is not True or
            compared.get('topk_ids_equal_all_positions') is not True or compared.get('topk_id_mismatch_positions')!=0 or
            compared.get('logprob_max_abs_diff_vs_reference_rounded_to_bf16')!=0 or compared.get('lse_max_abs_diff')!=0):
        raise ValueError('actual all792 ordered IDs/BF16 logprobs/FP32 LSE parity required')
    actual_decoder(proof,json_pin(Path(review['source_ingress']['path']),review['source_ingress']['sha256']),request)
    for name,spec in request['sources'].items():
        p=producer_sources/name
        A.pinned({'path':str(p),'sha256':spec['sha256']},'immutable producer source '+name)
    placement=proof.get('stock_placement',{})
    if (placement.get('reference_placement_only_changed') is not True or
            placement.get('original_forward_and_exact_id_gate_unchanged') is not True or
            len(placement.get('reference_facts',[]))!=1 or proof['stock_adapter']['sha256']!=request['sources']['video_stock_control.py']['sha256'] or
            proof.get('stock_teacher_after',{}).get('mode')!='full' or
            proof['stock_teacher_after'].get('tree_sha256')!=production['teacher']['tree_sha256_after'] or
            B.full_integrity(raw)['before']!=B.full_integrity(production)['before']):
        raise ValueError('authentic unchanged parent/stock placement source required')
    for key in ('dtype','head_precision','top_k','vocab_rows','rows_per_shard'):
        if raw['cache_config'][key]!=production['cache_config'][key]:raise ValueError('control numerical/processor policy differs')
    left_processor=json.loads(json.dumps(production['cache_config']['processor']))
    right_processor=json.loads(json.dumps(raw['cache_config']['processor']))
    stale_versions=[left_processor['video_decoder'].pop('version'),right_processor['video_decoder'].pop('version')]
    capture=json_pin(Path(review['source_capture']['path']),review['source_capture']['sha256'])
    if left_processor!=right_processor or capture['decoder_proof']!=proof['decoder_proof']:
        raise ValueError('actual decoder/processor controls differ')
    if len(raw['shards'])!=len(production['shards']):raise ValueError('control raw shard coverage differs')
    for left,right in zip(production['shards'],raw['shards']):
        if left['files']!=right['files'] or left['shard']!=right['shard']:raise ValueError('independent capture/control raw arrays differ')
        p=manifest_path.parent/right['shard'];sidecar=json_pin(p/'shard.json',right['shard_json_sha256'])
        if sidecar['files']!=right['files'] or sidecar['row_table']!=json.loads((Path(review['source_cache']['path']).parent/left['shard']/'shard.json').read_text())['row_table']:
            raise ValueError('control raw row tables/source IDs differ')
        for name,spec in right['files'].items():
            file=A.pinned({'path':str(p/name),'sha256':spec['sha256']},'independent stock-control raw '+name)
            if file.stat().st_size!=spec['bytes']:raise ValueError('control raw byte-size differs')
    return {'source_control':V.pin(control_path),'source_original_parity':V.pin(parity_path),
            'source_control_cache':V.pin(manifest_path),'source_control_execution':V.pin(control_root/'execution.json'),
            'all792_matched_singleton_positions':True,'raw_five_files_bitwise_equal':True,
            'stale_ffmpeg_versions_preserved':stale_versions,'stale_ffmpeg_version_used_as_decoder_evidence':False,
            'producer_sources':{name:{'path':str((producer_sources/name).resolve()),'sha256':spec['sha256']} for name,spec in request['sources'].items()},
            'actual_decoder':proof['decoder_proof'],'stock_teacher_after':proof['stock_teacher_after']}


def finalize(*,review,review_sha256,control_root,producer_sources,output):
    """Convert exact native preview arrays after authentic stock evidence joins.

    This seals usable tensors, not a training launch: selected amendment and
    real native forward/backward consumer proof remain separate requirements.
    """
    if output.exists():raise ValueError('fresh matched-control native binding output required')
    prepared=T.json_pin(review,review_sha256,'recovery_video_capture_review.v1')
    if (prepared.get('code_sha256')!=A.file_sha(Path(__file__)) or
            prepared.get('quantized_answer_source_sha256')!=A.file_sha(Path(Q.__file__)) or
            prepared.get('quantized_answer_policy')!=Q.POLICY or prepared.get('raw_five_file_seals_verified') is not True or
            prepared.get('complete_parent_masks_verified') is not True or prepared.get('feature_pixel_time_alignment_verified') is not True or
            prepared.get('rows')!=8 or prepared.get('assistant_positions')!=792):
        raise ValueError('actual complete source-pinned native review required')
    stock=bind_stock_control(prepared,control_root,producer_sources)
    from safetensors.numpy import save_file, load_file
    output.mkdir(parents=True);(output/'shards').mkdir();result=[]
    for entry in prepared['shards']:
        source=A.pinned(entry,'actual independently reviewed native row')
        with np.load(source,allow_pickle=False) as z:arrays={k:np.array(z[k],copy=True) for k in z.files}
        if (entry.get('split')!='train' or entry.get('held') is not False or entry.get('assistant_positions')!=99 or
                arrays['teacher_features'].shape!=(1,144,1152) or int(arrays['answer.q_w'].sum())!=99):
            raise ValueError('complete held-excluded temporal native source row required')
        path=output/'shards'/(entry['source_id']+'.safetensors');save_file(arrays,str(path))
        replay=load_file(str(path))
        if set(replay)!=set(arrays) or any(replay[k].dtype!=arrays[k].dtype or not np.array_equal(replay[k],arrays[k]) for k in arrays):
            raise ValueError('actual native serialization roundtrip differs')
        result.append({**entry,**V.pin(path),'source_review_shard':V.pin(source)})
    value={'schema':'chimera_recovery.temporal_video_native_binding.v1','sealed':True,'source_year':2026,
        'source_review':V.pin(review),'source_cache':prepared['source_cache'],'source_features':prepared['source_features'],
        'source_ingress':prepared['source_ingress'],'global_join':prepared['global_join'],'source_request':prepared['source_request'],
        'source_capture_execution':prepared['source_capture_execution'],
        'stock_control':stock,'shards':result,'rows':8,'assistant_positions':792,'heldout_groups_excluded':True,
        'complete_source_traversal':True,'stock_control_passed':True,'all_finite':True,'native_pixels_and_temporal_features_verified':True,
        'feature_lineage':'actual Qwen3.8 visual block13, distinct from historical QVL32 cache',
        'consumer_policy':POLICY,'rounding_policy':Q.POLICY,'source_rounding_compatible':True,
        'probabilities_changed':False,'exact_normalization_proven':False,'inherited_bucket_terms_loss_unchanged':True,
        'true_kl_or_nonnegative_exact_loss_claimed':False,'feature_weight':.1,'text_weight':1.,
        'native_unit':'qvl32_vit_L13','native_type':6,'feature_band':[640,2176],
        'observation_chart':V.POLICY,'parent_xyz_or_full_trunk_parity_claimed':False,
        'optimizer_use_forbidden':True,'native_train_ready':False,'training_started':False,
        'blockers':['selected explicit temporal-video amendment/config and native full-depth consumer proof required'],
        'code_sha256':A.file_sha(Path(__file__)),'quantized_answer_source_sha256':A.file_sha(Path(Q.__file__))}
    value['manifest_sha256']=A.digest(value);(output/'manifest.json').write_bytes(A.canonical(value)+b'\n');return value


def validate_native_row(*,batch,entry,manifest,manifest_path):
    """Pure CPU receiver check; neither a model nor optimizer is loaded."""
    if json.loads(Path(manifest_path).read_text())!=manifest:
        raise ValueError('consumer manifest object differs from actual file')
    if (manifest.get('schema')!='chimera_recovery.temporal_video_native_binding.v1' or manifest.get('sealed') is not True or
            A.digest({k:v for k,v in manifest.items() if k!='manifest_sha256'})!=manifest.get('manifest_sha256') or
            manifest.get('stock_control_passed') is not True or manifest.get('consumer_policy')!=POLICY or
            manifest.get('code_sha256')!=A.file_sha(Path(__file__)) or manifest.get('rounding_policy')!=Q.POLICY or
            manifest.get('quantized_answer_source_sha256')!=A.file_sha(Path(Q.__file__))):
        raise ValueError('matched-stock source-pinned temporal native binding required')
    review_path=A.pinned(manifest['source_review'],'original independent review')
    source_review=T.json_pin(review_path,manifest['source_review']['sha256'],'recovery_video_capture_review.v1')
    original=next((x for x in source_review['shards'] if x['source_id']==entry['source_id']),None)
    if original is None or any(entry[k]!=original[k] for k in ('source_row_sha256','asset_sha256','split','held','global_group_id')):
        raise ValueError('native row differs from original global source identity')
    if entry['held'] is not False or entry['split']!='train':raise ValueError('held rows cannot enter optimizer')
    A.pinned(entry,'sealed native tensor row')
    p=A.pinned(entry['source_review_shard'],'original native preview')
    with np.load(p,allow_pickle=False) as z:expected={k:np.array(z[k],copy=True) for k in z.files}
    supplied={k:(v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)) for k,v in batch.items()}
    if set(supplied)!=set(expected) or any(not np.array_equal(supplied[k],expected[k]) or supplied[k].dtype!=expected[k].dtype for k in expected):
        raise ValueError('consumer pixels/targets/masks/temporal rows differ from sealed actual preparation')
    # Reauthenticate evidence at the receiver: every source/response seal remains
    # pinned, including actual raw LSE/BF16 probability storage and stock receipts.
    for spec in ('source_cache','source_features','source_ingress','global_join','source_request'):
        A.pinned(manifest[spec],'native evidence '+spec)
    for spec in ('source_control','source_original_parity','source_control_cache','source_control_execution'):
        A.pinned(manifest['stock_control'][spec],'native stock evidence '+spec)
    A.pinned(manifest['source_capture_execution'],'native capture entry receipt')
    control_path=Path(manifest['stock_control']['source_control']['path'])
    sources=manifest['stock_control']['producer_sources']
    producer_root=Path(sources['video_answer.py']['path']).parent
    current_stock=bind_stock_control(source_review,control_path.parent,producer_root)
    if current_stock!=manifest['stock_control']:raise ValueError('native stock-control semantics differ')
    return {'verified':True,'source_id':entry['source_id'],'native_length':int(expected['input_ids'].shape[1]),
            'assistant_positions':int(expected['answer.q_w'].sum()),'media_rows':144,'teacher_feature_width':1152,
            'new_observation_chart':V.POLICY,'full_parent_trunk_parity_claimed':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--command',choices=('review','finalize'),default='review')
    if argv is None:
        import sys
        argv=sys.argv[1:]
    command=p.parse_known_args(argv)[0].command
    if command=='finalize':
        for field in ('review','control-root','producer-sources','output'):p.add_argument('--'+field,type=Path,required=True)
        p.add_argument('--review-sha256',required=True)
        args=vars(p.parse_args(argv));args.pop('command');r=finalize(**args)
        print(json.dumps({'rows':r['rows'],'positions':r['assistant_positions'],'native_train_ready':False}));return
    for field in ('cache-manifest','capture-receipt','request','plan','ingress','receipts','global-join','features','ingress-mirror','feature-mirror','output'):
        p.add_argument('--'+field,type=Path,required=True)
    for field in ('cache-manifest','capture-receipt','request','plan','ingress','global-join','features'):
        p.add_argument('--'+field+'-sha256',required=True)
    args=vars(p.parse_args(argv));args.pop('command');r=review_capture(**args);print(json.dumps({'rows':r['rows'],'positions':r['assistant_positions'],'native_train_ready':False}))


if __name__=='__main__':main()
