"""Actual VL writer entry point and strict sparse K64 answer conversion.

Conversion accepts only a complete original writer cache, a pinned recovery
assistant-mask capture receipt, and exact parent IDs from the CPU replay. It
cannot turn authored answers or visual features into fabricated teacher logits.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
try:
    from . import video_teacher as T
except ImportError:
    import video_teacher as T
V,A=T.V,T.A
ENGINE_SHA="77091f8ab1581c1edac3e3f1c1e5d727d60f8fb8465745c03a6eb6b12f0567da"
PLAN_SCHEMA='recovery_video_answer_plan.v1'
BATCH_POLICY={'micro_batch':1,'batch_order':'corpus','max_batch_tokens':8192,'rows_per_shard':8,
    'dtype':'bfloat16','head_precision':'fp32','top_k':64,'vocab_rows':V.TEXT_VOCAB,
    'head_chunk_rows':256,'attn_implementation':'sdpa','linear_attn_kernel':'auto',
    'host_resident':'pageable','teacher_digest':'full','digest_threads':4}


def contained(base,name):
    p=(base/name).resolve()
    if Path(name).is_absolute() or not p.is_relative_to(base.resolve()):
        raise ValueError('source path leaves the sealed private video preparation')
    return p


def prepare(*,vl_adapter,writer,plan,plan_sha256,inspection,inspection_sha256,ingress,ingress_sha256,
            receipts,global_join,global_join_sha256,output):
    """Subset authentic VL rows without editing messages, IDs, masks or media."""
    if output.exists():raise ValueError('fresh video-only plan output required')
    adapter=V.module(vl_adapter,V.VL_ADAPTER_SHA,'recovery_video_plan_original_adapter')
    original=adapter.load_writer(writer,V.WRITER_SHA)
    old=adapter.load_plan(plan,plan_sha256)
    replay,rows=T.load_inputs(ingress,ingress_sha256,receipts)
    proof=json.loads(A.pinned({'path':str(inspection),'sha256':inspection_sha256},'authentic processor inspection').read_text())
    join=T.json_pin(global_join,global_join_sha256,'recovery_video_global_join.v1')
    if (replay['plan']['sha256']!=plan_sha256 or replay['original_processor_inspection']['sha256']!=inspection_sha256 or
            proof.get('passed') is not True or proof.get('plan_sha256')!=plan_sha256 or
            proof.get('writer_sha256')!=V.WRITER_SHA or proof.get('adapter_sha256')!=V.VL_ADAPTER_SHA or
            join['source_ingress']['sha256']!=ingress_sha256 or join['source_receipts']['sha256']!=A.file_sha(receipts)):
        raise ValueError('exact original plan/inspection/24 temporal rows/global join required')
    decisions={r['id']:r for r in join['records']};replayed={r['id']:r for r in rows}
    inspected={r['row_id']:r for r in proof['rows']}
    selected={};assets={};records=[]
    for source in old['sources']:
        selected[source['split']]=[]
        for r in A.rows(contained(plan.parent,source['path'])):
            if r.get('route')!='video_understand':continue
            ident=r['row_id'];p=replayed[ident];d=decisions[ident]
            items=[i for m in r['messages'] for i in m.get('content',[]) if i.get('type')=='video']
            if (ident not in inspected or len(items)!=1 or items[0].get('num_frames')!=8 or
                    r['split']!=source['split'] or p['split']!=source['split'] or d['split']!=source['split'] or
                    r['source_row_sha256']!=p['source_row_sha256'] or d['source_row_sha256']!=p['source_row_sha256'] or
                    items[0]['sha256']!=p['asset_sha256'] or d['asset_sha256']!=p['asset_sha256'] or
                    (source['split']=='train' and not d['train_admitted']) or d['held']!=(source['split']!='train')):
                raise ValueError('unaltered admitted/held full-video source identity required')
            name=items[0]['path'];spec=old['assets'][name]
            A.pinned({'path':str(contained(plan.parent,name)),'sha256':spec['sha256']},'original video asset')
            assets[name]=spec;selected[source['split']].append(r)
            records.append({'id':ident,'split':source['split'],'held':d['held'],'train_admitted':d['train_admitted'],
                'source_row_sha256':p['source_row_sha256'],'teacher_row_sha256':A.digest(r),'asset_sha256':p['asset_sha256'],
                'global_group_id':d['recovery_group_id'],'processor_inspection':inspected[ident]})
    if (set(selected)!={'train','validation','test'} or any(len(rs)!=8 for rs in selected.values()) or
            len(records)!=24 or len({r['id'] for r in records})!=24 or len(assets)!=24):
        raise ValueError('complete24 video derivative with8 train/16 held required')
    output.mkdir(parents=True)
    for name,spec in assets.items():
        target=contained(output,name);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(contained(plan.parent,name),target)
        if A.file_sha(target)!=spec['sha256'] or target.stat().st_size!=spec['bytes']:
            raise ValueError('copied actual media bytes differ')
    sources=[]
    for split,rs in selected.items():
        target=output/(split+'.jsonl');target.write_bytes(b''.join(A.canonical(r)+b'\n' for r in rs))
        corpus=original.load_vl_corpus(target)
        sources.append({'path':target.name,'sha256':A.file_sha(target),'corpus_sha256':corpus['sha256'],
            'split':split,'rows':8,'row_ids':corpus['row_ids']})
    value={'schema':PLAN_SCHEMA,'sealed':True,'source_year':2026,'teacher':'q38-video-singleton-assistant-20261013-v1',
        'adapter_contract':old['adapter_contract'],'adapter_sha256':V.VL_ADAPTER_SHA,'writer_sha256':V.WRITER_SHA,
        'engine_sha256':ENGINE_SHA,'original_plan':V.pin(plan),'original_inspection':V.pin(inspection),
        'source_ingress':V.pin(ingress),'source_receipts':V.pin(receipts),'global_join':V.pin(global_join),
        'sources':sources,'assets':assets,'records':records,'batch_policy':BATCH_POLICY,'processor_options':old['processor_options'],
        'all_frames_retained':True,'complete_sequences':True,'assistant_only':True,'batch_shape_invariance_claimed':False,
        'source_train_eligible':8,'held':16,'scope':'procedural interface/regression baseline; insufficient for general video recovery',
        'teacher_targets_generated':False,'native_train_ready':False,'training_started':False,'code_sha256':A.file_sha(Path(__file__))}
    value['manifest_sha256']=A.digest(value);(output/'manifest.json').write_bytes(A.canonical(value)+b'\n');return value


def load_plan(path,sha):
    r=T.json_pin(path,sha,PLAN_SCHEMA)
    if (r['batch_policy']!=BATCH_POLICY or r['writer_sha256']!=V.WRITER_SHA or r['adapter_sha256']!=V.VL_ADAPTER_SHA or
            r['engine_sha256']!=ENGINE_SHA or r.get('code_sha256')!=A.file_sha(Path(__file__)) or
            r.get('all_frames_retained') is not True or r.get('complete_sequences') is not True or
            r.get('assistant_only') is not True or r.get('batch_shape_invariance_claimed') is not False):
        raise ValueError('source-pinned full-sequence singleton answer policy differs')
    for s in r['sources']:A.pinned({'path':str(contained(path.parent,s['path'])),'sha256':s['sha256']},'subset corpus')
    for name,s in r['assets'].items():
        p=A.pinned({'path':str(contained(path.parent,name)),'sha256':s['sha256']},'copied video')
        if p.stat().st_size!=s['bytes']:raise ValueError('video source byte size differs')
    return r


def writer_argv(plan,plan_path,split,model_dir,output,allocator_receipt):
    source=next(r for r in plan['sources'] if r['split']==split)
    return ['build-vl','--model-dir',str(model_dir),'--processor-dir',str(model_dir),'--teacher',plan['teacher'],
        '--out-root',str(output),'--corpus',str(contained(plan_path.parent,source['path'])),'--device','cuda:0',
        '--dtype','bfloat16','--top-k','64','--vocab-rows',str(V.TEXT_VOCAB),'--micro-batch','1',
        '--max-batch-tokens','8192','--head-chunk-rows','256','--rows-per-shard','8','--batch-order','corpus',
        '--head-precision','fp32','--attn-implementation','sdpa','--linear-attn-kernel','auto',
        '--teacher-digest','full','--digest-threads','4','--host-resident','pageable','--max-pixels','262144',
        '--chat-template-kwargs','{"enable_thinking":false}','--verify-existing',
        '--allocator-receipt',str(allocator_receipt),'--purpose','source-pinned full-sequence singleton assistant-only video K64',
        '--max-new-shards','0']


def original_inputs(vl_adapter,writer,plan,plan_sha256,inspection,inspection_sha256,ingress,ingress_sha256,model_dir,allocator_receipt):
    prepared=load_plan(plan,plan_sha256)
    replay=T.json_pin(ingress,ingress_sha256,V.SCHEMA)
    if (prepared['source_ingress']['sha256']!=ingress_sha256 or replay['decoder_proof'].get('alternate') is not True or
            prepared['original_inspection']['sha256']!=inspection_sha256 or not allocator_receipt.is_file()):
        raise ValueError('actual source-pinned alternate-decoder temporal proof and real allocation required')
    adapter=V.module(vl_adapter,V.VL_ADAPTER_SHA,'recovery_temporal_original_vl_adapter')
    original=adapter.load_writer(writer,V.WRITER_SHA)
    old=json.loads(A.pinned({'path':str(inspection),'sha256':inspection_sha256},'original48 CPU processor inspection').read_text())
    if (old.get('passed') is not True or old.get('adapter_sha256')!=V.VL_ADAPTER_SHA or
            old.get('writer_sha256')!=V.WRITER_SHA or old.get('plan_sha256')!=prepared['original_plan']['sha256'] or
            old['processor']['files_sha256']!=original.processor_files(model_dir) or
            old['model_config_sha256']!=A.file_sha(model_dir/'config.json')):
        raise ValueError('actual inspected processor/tokenizer/config/source pins changed')
    return prepared,adapter,original


def control(**kwargs):
    """Original streamed-versus-stock VL control, identical singleton corpus.

    This callable does not allocate a GPU. The root must first allocate capacity
    for the original stock full load as well as its streamed arm.
    """
    if kwargs['receipt'].exists() or kwargs['output'].exists():raise ValueError('fresh parity namespace required')
    plan,adapter,W=original_inputs(**{k:kwargs[k] for k in ('vl_adapter','writer','plan','plan_sha256','inspection',
        'inspection_sha256','ingress','ingress_sha256','model_dir','allocator_receipt')})
    argv=writer_argv(plan,kwargs['plan'],kwargs['split'],kwargs['model_dir'],kwargs['output'],kwargs['allocator_receipt'])
    args=W.parser().parse_args(argv);args.work_dir=str(kwargs['output']);args.receipt=str(kwargs['receipt'])
    args.corpus=args.corpus[0];args.parity_rows=8;args.parity_length=4096;args.seed=20261013
    corpus=W.load_vl_corpus(Path(args.corpus));source=next(r for r in plan['sources'] if r['split']==kwargs['split'])
    if corpus['sha256']!=source['corpus_sha256'] or corpus['row_ids']!=source['row_ids']:
        raise ValueError('same complete singleton control corpus required')
    with V.decoder_adapter(W,'pyav_full',kwargs['decoder_wheel']) as decoder:
        with adapter.assistant_mask_adapter(W):code=W.run_parity(args,'vl')
    if not kwargs['receipt'].is_file():raise ValueError('original stock control produced no receipt')
    original=json.loads(kwargs['receipt'].read_text());cache=Path(original['cache_dir'])/'manifest.json'
    value={'schema':'recovery_video_answer_control.v1','source_year':2026,'exit_code':code,
        'source_plan':V.pin(kwargs['plan']),'source_ingress':V.pin(kwargs['ingress']),
        'original_parity_receipt':V.pin(kwargs['receipt']),'original_parity':original,
        'streamed_control_cache':V.pin(cache),'batch_policy':BATCH_POLICY,'split':kwargs['split'],
        'decoder_proof':decoder,'adapter':V.pin(kwargs['vl_adapter']),'writer':V.pin(kwargs['writer']),
        'batch_shape_invariance_claimed':False,'native_train_ready':False,'code_sha256':A.file_sha(Path(__file__))}
    target=kwargs['receipt'].with_name(kwargs['receipt'].stem+'.video-control.json');target.write_bytes(A.canonical(value)+b'\n')
    if code or original.get('passed') is not True:raise ValueError('original same-singleton stock control failed; receipts preserved')
    return value


def dense_answer(ids,mask,positions,cached_ids,topk_ids,raw_bf16):
    ids,mask,positions,cached_ids,topk_ids,raw_bf16=map(np.asarray,(ids,mask,positions,cached_ids,topk_ids,raw_bf16))
    if ids.ndim!=2 or ids.shape[0]!=1 or ids.dtype!=np.int64 or mask.dtype!=np.bool_ or mask.shape!=(ids.shape[1],):
        raise ValueError("authentic parent IDs and target-indexed assistant mask required")
    length=ids.shape[1];owned=np.zeros(length,np.bool_);owned[:-1]=mask[1:]
    expected=np.flatnonzero(owned)
    if (not len(expected) or not np.array_equal(positions,expected) or cached_ids.shape!=expected.shape or
            not np.array_equal(cached_ids,ids[0,expected]) or topk_ids.shape!=(len(expected),64) or
            not np.issubdtype(topk_ids.dtype,np.integer) or raw_bf16.dtype!=np.uint16 or raw_bf16.shape!=topk_ids.shape):
        raise ValueError("actual raw cache must cover exactly the once-shifted assistant prediction positions")
    lp=(raw_bf16.astype(np.uint32)<<16).view(np.float32)
    if (np.any(topk_ids<0) or np.any(topk_ids>=V.TEXT_VOCAB) or not np.isfinite(lp).all() or np.any(lp>0) or
            any(len(set(row.tolist()))!=64 for row in topk_ids) or np.any(np.exp(lp.astype(np.float64)).sum(1)>1.001)):
        raise ValueError("actual native-vocabulary K64/BF16 distributions are invalid")
    result={'input_ids':ids.copy(),'assistant_token_mask':mask.copy(),'next_ids':np.zeros(length,np.int64),
        'ce_w':np.zeros(length,np.float32),'eos_w':np.zeros(length,np.float32),'q_w':owned.astype(np.float32),
        'q_ids':np.zeros((length,64),np.int64),'q_lp':np.zeros((length,64),np.float32)}
    result['next_ids'][expected]=ids[0,expected+1];result['q_ids'][expected]=topk_ids
    result['q_lp'][expected]=lp
    return result


def capture(*,vl_adapter,writer,plan,plan_sha256,inspection,inspection_sha256,ingress,ingress_sha256,
            model_dir,output,allocator_receipt,receipt,split,decoder_wheel):
    if receipt.exists() or output.exists():raise ValueError("fresh answer capture namespace required")
    prepared,adapter,original=original_inputs(vl_adapter,writer,plan,plan_sha256,inspection,inspection_sha256,
        ingress,ingress_sha256,model_dir,allocator_receipt)
    argv=writer_argv(prepared,plan,split,model_dir,output,allocator_receipt)
    args=original.parser().parse_args(argv);corpus=original.load_vl_corpus(Path(args.corpus[0]))
    source=next(r for r in prepared['sources'] if r['split']==split)
    if corpus['sha256']!=source['corpus_sha256'] or corpus['row_ids']!=source['row_ids']:
        raise ValueError('actual video-only corpus/media/order changed')
    failure=None;code=2
    with V.decoder_adapter(original,'pyav_full',decoder_wheel) as decoder:
        try:
            with adapter.assistant_mask_adapter(original):code=original.build_cache(args,'vl')
        except Exception as error:failure={'type':type(error).__name__,'message':str(error)}
    cache=output/prepared['teacher']/corpus['sha256'];sealed=(cache/'manifest.json').is_file()
    verified=sealed and original.verify_cache(cache)==0
    inner={'schema':'chimera_recovery.vl_teacher_run.v1','plan_sha256':plan_sha256,'inspection_sha256':inspection_sha256,
        'adapter_sha256':V.VL_ADAPTER_SHA,'writer_sha256':V.WRITER_SHA,'split':split,'argv':argv,'cache':str(cache),
        'writer_exitcode':code,'cache_sealed_and_verified':verified,'cache_manifest_sha256':A.file_sha(cache/'manifest.json') if sealed else None,
        'assistant_only':True,'teacher_forced_authored_reference':True,'native_student_training_started':False,
        'failure':failure,'allocator_receipt_sha256':A.file_sha(allocator_receipt)}
    receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_bytes(A.canonical(inner)+b'\n')
    value={'schema':'recovery_video_answer_capture.v1','source_year':2026,'exit_code':code,
        'source_ingress':V.pin(ingress),'vl_capture_receipt':V.pin(receipt),'vl_capture':inner,
        'decoder_proof':decoder,'adapter':V.pin(vl_adapter),'writer':V.pin(writer),
        'source_plan':V.pin(plan),'batch_policy':BATCH_POLICY,
        'assistant_only':True,'batch_shape_invariance_claimed':False,'native_train_ready':False,
        'code_sha256':A.file_sha(Path(__file__))}
    target=receipt.with_name(receipt.stem+'.video.json');target.write_bytes(A.canonical(value)+b'\n')
    if failure or code or not verified:raise ValueError('actual teacher build/verification failed; private receipts preserved')
    return value


def full_integrity(manifest):
    teacher=manifest.get('teacher',{});tree=teacher.get('tree_sha256_before');sessions=manifest.get('sessions',[])
    files=teacher.get('files',[])
    if (teacher.get('digest_mode')!='full' or teacher.get('equal_before_and_after_every_session') is not True or
            not tree or tree!=teacher.get('tree_sha256_after') or not sessions or not files or
            A.digest(files)!=tree or any(s.get('unchanged') is not True or
                any(s.get(k,{}).get('mode')!='full' or s[k].get('tree_sha256')!=tree for k in ('before','after')) for s in sessions) or
            any(not A.SHA.fullmatch(str(f.get('sha256'))) or type(f.get('bytes')) is not int or f['bytes']<=0 for f in files)):
        raise ValueError("original full teacher file identities must be equal before/after every complete session")
    # After is derived from the actual original full-digest tree equality, not
    # from a newly asserted flag or a post-hoc rewrite of its receipt.
    before={f['path']:{'sha256':f['sha256'],'bytes':f['bytes']} for f in files}
    if len(before)!=len(files):raise ValueError("teacher file list contains duplicates")
    return {'mode':'full','before':before,'after':before.copy(),'unchanged':True,
        'source_before_tree_sha256':tree,'source_after_tree_sha256':teacher['tree_sha256_after'],
        'derivation':'original complete full-digest session equality and canonical file-tree hash'}


def convert(*,cache_manifest,cache_manifest_sha256,capture_receipt,capture_receipt_sha256,
            control_receipt,control_receipt_sha256,ingress,ingress_sha256,receipts,global_join,global_join_sha256,writer,output,split,mirror=None):
    if output.exists():raise ValueError("fresh answer conversion output required")
    _,rows=T.load_inputs(ingress,ingress_sha256,receipts)
    join=T.json_pin(global_join,global_join_sha256,'recovery_video_global_join.v1');decisions={r['id']:r for r in join['records']}
    if join['source_ingress']['sha256']!=ingress_sha256 or join['source_receipts']['sha256']!=A.file_sha(receipts):
        raise ValueError("answer global join identifies a different temporal source")
    capture_path=A.pinned({'path':str(capture_receipt),'sha256':capture_receipt_sha256},'actual answer capture')
    capture_record=json.loads(capture_path.read_text());inner=capture_record.get('vl_capture',{})
    if (capture_record.get('schema')!='recovery_video_answer_capture.v1' or capture_record.get('exit_code')!=0 or
            capture_record.get('assistant_only') is not True or capture_record['source_ingress']['sha256']!=ingress_sha256 or
            inner.get('cache_sealed_and_verified') is not True or inner.get('split')!=split or
            inner.get('adapter_sha256')!=V.VL_ADAPTER_SHA or inner.get('writer_sha256')!=V.WRITER_SHA or
            inner.get('cache_manifest_sha256')!=cache_manifest_sha256 or capture_record.get('batch_policy')!=BATCH_POLICY or
            capture_record.get('code_sha256')!=A.file_sha(Path(__file__))):
        raise ValueError("genuine original assistant-only teacher capture receipt required")
    prepared=load_plan(Path(capture_record['source_plan']['path']),capture_record['source_plan']['sha256'])
    A.pinned(capture_record['vl_capture_receipt'],'unchanged original writer run receipt')
    path=A.pinned({'path':str(cache_manifest),'sha256':cache_manifest_sha256},'original raw teacher cache')
    raw=json.loads(path.read_text());cfg=raw.get('cache_config',{})
    if (raw.get('schema')!='omni-distill-teacher-cache-manifest-v1' or raw.get('all_finite') is not True or
            raw.get('all_structural_ok') is not True or cfg.get('top_k')!=64 or cfg.get('vocab_rows')!=V.TEXT_VOCAB or
            cfg.get('dtype')!='bfloat16' or cfg.get('head_precision')!='fp32' or cfg.get('rows_per_shard')!=8 or
            raw.get('tool',{}).get('sha256')!=V.WRITER_SHA or raw.get('engine',{}).get('sha256')!=ENGINE_SHA):
        raise ValueError("original source-pinned native-vocabulary K64/FP32-head/BF16 cache required")
    integrity=full_integrity(raw)
    control_path=A.pinned({'path':str(control_receipt),'sha256':control_receipt_sha256},'actual same-singleton stock control')
    proof=json.loads(control_path.read_text());parity=proof.get('original_parity',{})
    comparison=parity.get('comparison_batched_same_inputs',{})
    if (proof.get('schema')!='recovery_video_answer_control.v1' or proof.get('exit_code')!=0 or proof.get('split')!=split or
            proof.get('source_plan',{}).get('sha256')!=capture_record['source_plan']['sha256'] or
            proof.get('source_ingress',{}).get('sha256')!=ingress_sha256 or proof.get('batch_policy')!=BATCH_POLICY or
            proof.get('code_sha256')!=A.file_sha(Path(__file__)) or parity.get('passed') is not True or
            comparison.get('same_positions_and_input_ids') is not True or comparison.get('topk_ids_equal_all_positions') is not True or
            parity.get('tool',{}).get('sha256')!=V.WRITER_SHA or parity.get('engine',{}).get('sha256')!=ENGINE_SHA or
            parity.get('corpus_sha256')!=raw['corpus']['sha256'] or parity.get('top_k')!=64 or parity.get('vocab_rows')!=V.TEXT_VOCAB or
            parity.get('batching')!={'micro_batch':1,'max_batch_tokens':8192,'order':'corpus','rows_per_shard':8}):
        raise ValueError('genuine same-full-sequence singleton stock control required')
    A.pinned(proof['original_parity_receipt'],'unchanged original stock parity receipt')
    control_cache=A.pinned(proof['streamed_control_cache'],'unchanged original streamed control cache')
    control_raw=json.loads(control_cache.read_text())
    if (parity['manifest_sha256']!=proof['streamed_control_cache']['sha256'] or
            full_integrity(control_raw)['before']!=integrity['before']):
        raise ValueError('control and production teacher file identities differ')
    W=V.module(writer,V.WRITER_SHA,'recovery_video_answer_original_writer')
    for entry in control_raw['shards']:
        base=control_cache.parent/entry['shard']
        if A.file_sha(base/'shard.json')!=entry['shard_json_sha256'] or not W.verify_shard(base)['ok']:
            raise ValueError('actual stock-control raw shard/file seals differ')
    for key in ('dtype','head_precision','top_k','vocab_rows','rows_per_shard','processor','chat_template_kwargs'):
        if control_raw['cache_config'].get(key)!=cfg.get(key):
            raise ValueError('stock control and production numerical/processor policy differs')
    corpus=W.load_vl_corpus(Path(raw['corpus']['path']))
    source=next(r for r in prepared['sources'] if r['split']==split)
    if (corpus['sha256']!=raw['corpus']['sha256'] or corpus['kind']!='vl' or corpus['sha256']!=source['corpus_sha256'] or
            corpus['row_ids']!=source['row_ids'] or corpus['rows']!=8):raise ValueError("raw teacher corpus/media/order changed")
    # The streamed arm of the matched stock control must reproduce the actual
    # production raw arrays, including BF16 logprob and FP32-LSE file identities.
    def data_seals(m):
        return {s['shard']:{name:f['sha256'] for name,f in s['files'].items()} for s in m['shards']}
    if data_seals(control_raw)!=data_seals(raw):raise ValueError('singleton control and production raw arrays differ')
    row_lookup={}
    for entry in raw['shards']:
        base=path.parent/entry['shard'];meta=base/'shard.json'
        if A.file_sha(meta)!=entry['shard_json_sha256'] or not W.verify_shard(base)['ok']:
            raise ValueError("raw teacher shard/one of its five file seals differs")
        s=json.loads(meta.read_text())
        if s['micro_batch']!=1 or s['batch_order']!='corpus' or s['max_batch_tokens']!=8192:
            raise ValueError("raw teacher batch/padding policy differs; shape invariance is not assumed")
        for row in s['row_table']:
            ident=row['row_id']
            if ident in row_lookup or corpus['row_ids'][row['row']]!=ident:raise ValueError("raw teacher duplicate/misaligned row ID")
            row_lookup[ident]=(base,s,row)
    selected=[r for r in rows if r['split']==split]
    if len(selected)!=8 or set(row_lookup)!={r['id'] for r in selected}:raise ValueError("complete exact split8 videos required")
    from safetensors.torch import save_file
    import torch
    output.mkdir(parents=True);(output/'shards').mkdir();result=[]
    for r in selected:
        d=decisions[r['id']]
        if (split=='train' and not d['train_admitted']) or d['source_row_sha256']!=r['source_row_sha256'] or d['asset_sha256']!=r['asset_sha256']:
            raise ValueError("same admitted source or held eval identity required")
        with np.load(T.shard_path(r['ingress'],mirror),allow_pickle=False) as data:
            ingress_arrays={k:np.array(data[k],copy=True) for k in data.files}
        base,s,table=row_lookup[r['id']];p=int(s['positions']);k=int(s['top_k']);start=int(table['offset']);stop=start+int(table['count'])
        if table['length']!=ingress_arrays['parent_input_ids'].shape[1] or not 0<=start<stop<=p:raise ValueError("raw answer row length/offset differs")
        def array(name,dtype,shape):return np.fromfile(base/name,dtype=dtype).reshape(shape)[start:stop].copy()
        answer=dense_answer(ingress_arrays['parent_input_ids'],ingress_arrays['parent_assistant_token_mask'],
            array('pos.u32','<u4',(p,)),array('ids.u32','<u4',(p,)),array('topk_ids.i32','<i4',(p,k)),
            array('topk_logprobs.bf16','<u2',(p,k)))
        V.rebase_answer(answer,ingress_arrays)  # independently verify native once-shift conversion now
        shard=output/'shards'/(r['id']+'.safetensors');save_file({k:torch.from_numpy(v) for k,v in answer.items()},str(shard))
        result.append({**V.pin(shard),'source_id':r['id'],'source_row_sha256':r['source_row_sha256'],
            'asset_sha256':r['asset_sha256'],'split':split,'held':split!='train','global_group_id':d['recovery_group_id'],
            'raw_shard':V.pin(base/'shard.json'),'raw_row_table':table,'source_ingress':r['ingress'],
            'assistant_positions':int(answer['q_w'].sum())})
    value={'schema':'chimera_recovery.vision_answer_source.v1','sealed':True,'source_year':2026,'all_finite':True,
        'assistant_only':True,'native_text_vocab_rows':V.TEXT_VOCAB,'teacher_weights_unchanged':True,
        'teacher_integrity':integrity,'source_cache':V.pin(path),'source_capture':V.pin(capture_path),
        'source_stock_control':V.pin(control_path),'stock_singleton_control_passed':True,
        'source_ingress':V.pin(ingress),'global_join':V.pin(global_join),'shards':result,'batch_shape_invariance_claimed':False,
        'batch_policy':BATCH_POLICY,'native_train_ready':False,'training_started':False,'code_sha256':A.file_sha(Path(__file__))}
    value['manifest_sha256']=A.digest(value);(output/'manifest.json').write_bytes(A.canonical(value)+b'\n');return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('prepare','capture','control','convert'):
        s=sub.add_parser(name)
        for field in ('writer','ingress','output'):s.add_argument('--'+field,type=Path,required=True)
        s.add_argument('--ingress-sha256',required=True)
        if name!='prepare':s.add_argument('--split',choices=('train','validation','test'),required=True)
        if name in ('prepare','capture','control'):
            fields=('vl-adapter','plan','inspection')+ (('model-dir','allocator-receipt','receipt','decoder-wheel') if name in ('capture','control') else ('receipts','global-join'))
            for field in fields:s.add_argument('--'+field,type=Path,required=True)
            for field in ('plan-sha256','inspection-sha256'):s.add_argument('--'+field,required=True)
            if name=='prepare':s.add_argument('--global-join-sha256',required=True)
        else:
            for field in ('cache-manifest','capture-receipt','control-receipt','receipts','global-join'):s.add_argument('--'+field,type=Path,required=True)
            for field in ('cache-manifest-sha256','capture-receipt-sha256','control-receipt-sha256','global-join-sha256'):s.add_argument('--'+field,required=True)
            s.add_argument('--mirror',type=Path)
    args=vars(p.parse_args(argv));name=args.pop('command');result={'prepare':prepare,'capture':capture,'control':control,'convert':convert}[name](**args)
    print(json.dumps({'schema':result['schema'],'native_train_ready':False}));return 0


if __name__=='__main__':raise SystemExit(main())
