"""Global same-mux source admission and faithful sealed H3 native conversion.

CPU preparation only. The finite perturbation map defines an amended bridge
clock; it is not an exact receptive-field measurement. Parent PackedLayout and
packers are separately proved bitwise by the captured diagnostic tensors.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value
H = module('recovery_h3_native_capture', HERE/'h3_cache.py')
M = module('recovery_h3_native_storage', HERE/'media_cache.py')
R, Error = M.R, M.Error
BANDS = {2:{'feature':(384,480),'emission':(128,224)},3:{'feature':(480,512),'emission':(224,256)}}
SIGNS = {2:1.,3:1.}
SUFFIX = '\nNative recovery view: generate only the first 22/24 seconds on 22 frames at 24 fps; preserve the requested source event times and sample each visual event by latest-frame hold.'
COMMON = ('normalized','corpus_manifest','eval_rows','frozen_exclusions','prior_admission')

def pin(path):
    return {'path':str(Path(path).resolve()),'sha256':H.sha(path)}

def pinned(ref, root=Path('.'), aliases=None):
    """Explicit SHA-keyed read-only aliases preserve original source pin bytes."""
    expected = H.require_sha(ref.get('sha256'),'source SHA')
    path = Path(ref['path']); path = path if path.is_absolute() else root/path
    if not path.is_file() and aliases and expected in aliases:
        path = Path(aliases[expected])
    if not path.is_file() or H.sha(path) != expected:
        raise Error('source file missing or changed: '+str(path))
    return path.resolve()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def rows(path):
    result = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]
    if len({row['id'] for row in result}) != len(result):
        raise Error('source IDs must occur exactly once')
    return result

def source_candidate(record):
    """Verify original authored media, without altering its source row."""
    media, p = record.get('media',[]), record.get('provenance',{})
    if (record.get('schema')!='recovery_corpus.v1' or record.get('route')!='joint_av' or
            record.get('split') not in {'train','validation','test'} or p.get('source_year') not in {2025,2026} or
            len(media)!=2 or [m.get('type') for m in media]!=['video','audio'] or
            any(m.get('role')!='target' or m.get('source_verified') is not True for m in media) or
            [m.get('role') for m in record.get('messages',[])]!=['user','assistant'] or
            record.get('av_alignment',{}).get('same_origin') is not True or
            not p.get('group_id') or p.get('split')!=record['split']):
        raise Error('H3 source requires an unchanged authored same-mux AV target and explicit source split')
    mux = H.require_sha(media[0].get('sha256'),'physical mux')
    if any(m.get('source_media_sha256')!=mux or m.get('start_seconds')!=0. for m in media):
        raise Error('H3 source streams must identify the same physical origin and start')
    for m in media:
        pinned(m)
    groundtruth = pinned({'path':p['groundtruth_path'],'sha256':p['groundtruth_sha256']})
    g=read(groundtruth).get('groundtruth',{});answer=json.loads(record['messages'][-1]['content'])
    if (g.get('audio_video_muxed') is not True or g.get('same_origin_sha256')!=mux or
            R.digest({'count':g.get('count'),'events':g.get('events')}) != R.digest(answer)):
        raise Error('authored AV groundtruth differs from original sealed assistant metadata')
    streams = media[0].get('probe',{}).get('streams',[])
    video = next((s for s in streams if s.get('codec_type')=='video'),{})
    audio = next((s for s in streams if s.get('codec_type')=='audio'),{})
    if (video.get('avg_frame_rate')!='8/1' or video.get('width')!=96 or video.get('height')!=96 or
            float(video.get('duration',0))!=1. or float(audio.get('duration',0))!=1. or
            video.get('start_time')!='0.000000' or audio.get('start_time')!='0.000000'):
        raise Error('source lacks the measured one-second8fps96px physical AV stream')
    return {'record':record,'asset_path':media[0]['path'],'asset_sha256':mux,
            'audio_asset_sha256':media[1]['sha256'],'source_mux_sha256':mux,
            'groundtruth_sha256':p['groundtruth_sha256'],'source_scope':'same physical mux; native22/24s prefix'}

def prepare_source(joint_admission, expected_sha, helper_path, helper_sha, out):
    """Re-run the complete graph with225 prior candidates plus24 AV sources."""
    if out.exists():
        raise Error('H3 admission requires a fresh output; prior source seals remain immutable')
    parent_path=pinned({'path':str(joint_admission),'sha256':expected_sha})
    parent=read(parent_path)
    A=module('recovery_h3_native_global_admission',pinned({'path':str(helper_path),'sha256':helper_sha}))
    if (parent.get('schema')!='recovery_audio_admission.v1' or parent.get('sealed') is not True or
            parent.get('complete_source_traversal') is not True or parent.get('connected_groups_verified') is not True or
            R.digest({k:v for k,v in parent.items() if k!='manifest_sha256'})!=parent.get('manifest_sha256')):
        raise Error('prior225 global source admission seal differs')
    source=pinned(parent['sources']['source'],parent_path.parent)
    inventory=pinned(parent['sources']['source_inventory'],parent_path.parent)
    old_rows=rows(pinned(parent['outputs']['admission'],parent_path.parent)); old={r['id']:r for r in old_rows}
    originals=rows(source); full=rows(inventory); all_rows={r['id']:r for r in full}
    if set(old)!={r['id'] for r in originals} or any(R.digest(r)!=R.digest(all_rows.get(r['id'])) for r in originals):
        raise Error('prior candidates must match every unchanged original inventory row')
    candidates=[]
    for r in originals:
        entry=old[r['id']]
        if R.digest(r)!=entry['source_row_sha256'] or (not entry['train_admitted'] and not entry['blockers']):
            raise Error('prior source join/exclusion cannot be lifted')
        pinned({'path':entry['asset_path'],'sha256':entry['asset_sha256']})
        candidate={k:v for k,v in entry.items() if k not in {'schema','id','item_id','route','split','recovery_group_id',
            'connected_source_groups','source_group_id','source_row_sha256','authored_answer_sha256','prompt_sha256',
            'source_train_eligible','train_admitted','native_train_ready','encoder_status','source_pins'}}
        candidate.update(record=r,original_source_admission=pin(parent_path))
        candidates.append(candidate)
    av=[source_candidate(r) for r in full if r['route']=='joint_av']
    if not av or any(r['record']['id'] in old for r in av):
        raise Error('new complete AV scope is empty or overlaps prior source IDs')
    candidates.extend(av);out.mkdir(parents=True)
    candidate_source=out/'source_ready.jsonl'
    candidate_source.write_bytes(b''.join(R.canonical(c['record'])+b'\n' for c in candidates))
    specs={k:parent['sources'][k] for k in COMMON}
    specs.update(source=pin(candidate_source),source_inventory=pin(inventory),prior_joint_audio_admission=pin(parent_path),
                 prior_joint_audio_rows=parent['outputs']['admission'],source_admission_helper=pin(helper_path))
    manifest=A.seal_admission(specs=specs,candidates=candidates,inventory=full,output=out/'global_admission')
    current={r['id']:r for r in rows(out/'global_admission/admission.jsonl')}
    changed=[]
    for ident,old_row in old.items():
        new=current[ident]
        if set(old_row['blockers'])-set(new['blockers']) or (not old_row['train_admitted'] and new['train_admitted']):
            raise Error('new AV traversal lifted an old source exclusion')
        if (new['train_admitted'],new['recovery_group_id'])!=(old_row['train_admitted'],old_row['recovery_group_id']):
            changed.append({'id':ident,'old':old_row['recovery_group_id'],'new':new['recovery_group_id'],
                            'train_admitted':new['train_admitted'],'blockers':new['blockers']})
    result={'schema':'chimera_recovery.h3_source_preparation.v1','sealed':True,'source':pin(candidate_source),
        'source_admission':pin(out/'global_admission/manifest.json'),'prior_source_admission':pin(parent_path),
        'joint_av_source_ids':[c['record']['id'] for c in av],
        'joint_av_train_ids':[c['record']['id'] for c in av if current[c['record']['id']]['train_admitted']],
        'prior_native_source_changes':changed,'full_source_inventory':len(full),'full_candidates':len(candidates),
        'native_admission_granted':False,'training_started':False,'adapter_sha256':H.sha(Path(__file__))}
    (out/'preparation.json').write_bytes(R.canonical(result)+b'\n');return result

def geometry_proof(path, expected_sha, cache_sha, item_ids):
    proof=read(pinned({'path':str(path),'sha256':expected_sha}))
    if (proof.get('schema')!='chimera.h3-native-geometry-probe.v1' or proof.get('complete') is not True or
            proof.get('weights_unchanged') is not True or proof.get('shared_group_map_stable') is not True or
            proof.get('groups_nonempty_contiguous') is not True or
            proof.get('group_semantics')!='dominant per-frame finite-perturbation influence, not exclusive receptive fields' or
            proof.get('capture_preflight',{}).get('cache_manifest_sha256')!=cache_sha or
            proof.get('capture_preflight',{}).get('capture_contract_ok') is not True or
            proof.get('video_weight_before')!=proof.get('video_weight_after') or
            proof.get('model_state_before')!=proof.get('model_state_after')):
        raise Error('actual measured geometry, unchanged parent and complete target cache proof required')
    measurements=proof.get('measurements',[])
    for name in ('video_weight_before','video_weight_after','model_state_before','model_state_after'):
        H.require_sha(proof.get(name),'actual unchanged geometry parent '+name)
    if {m['id'] for m in measurements}!=set(item_ids) or len(measurements)!=len(item_ids):
        raise Error('geometry proof must cover every source exactly once')
    groups=proof['shared_groups']
    if [i for g in groups for i in g]!=list(range(22)) or any(not g for g in groups):
        raise Error('measured shared dominance groups must cover the22frame view')
    if any(m.get('groups')!=groups or m.get('base_bitwise_equal') is not True for m in measurements):
        raise Error('all source measurements must share the same clock and genuine clean encode')
    targets=proof.get('numerical_targets',[])
    if (len({p['target'] for p in targets})!=len(targets) or any(p.get('native_PackedLayout_bitwise_equal') is not True or
            p.get('native_packers_bitwise_equal') is not True or p.get('id') not in item_ids for p in targets)):
        raise Error('complete unique parent packer/layout diagnostic evidence required')
    return proof,{p['target']:p for p in targets}

def prepare_cache(cache, manifest_sha, items_path, admission_path, admission_sha, proof_path, proof_sha,
                  tokenizer, amendment_sha, out, *, aliases=None):
    import torch
    from safetensors.torch import load_file,save_file
    if out.exists():raise Error('native H3 output must be fresh')
    check=H.preflight(cache,read(cache/'manifest.json')['tool']['sha256'])
    if not check['capture_contract_ok'] or check['cache_manifest_sha256']!=manifest_sha:
        raise Error('actual H3 source cache preflight/seal differs')
    manifest=read(cache/'manifest.json');items=rows(items_path)
    if H.sha(items_path)!=manifest['corpus']['sha256']:raise Error('teacher item file differs from sealed capture')
    parent_path=pinned({'path':str(admission_path),'sha256':admission_sha})
    admission=read(parent_path)
    if (admission.get('schema')!='recovery_audio_admission.v1' or admission.get('sealed') is not True or
            admission.get('connected_groups_verified') is not True or admission.get('complete_source_traversal') is not True or
            R.digest({k:v for k,v in admission.items() if k!='manifest_sha256'})!=admission.get('manifest_sha256')):
        raise Error('full connected source admission required')
    source_path=pinned(admission['sources']['source'],parent_path.parent,aliases)
    source={r['id']:r for r in rows(source_path)}
    admitted={r['id']:r for r in rows(pinned(admission['outputs']['admission'],parent_path.parent,aliases))}
    av={i:r for i,r in admitted.items() if r['route']=='joint_av'}
    if not av or set(source)!=set(admitted):raise Error('native H3 source/admission scope differs')
    if {i for i,r in av.items() if r['train_admitted']} - {item['id'] for item in items}:
        raise Error('admitted AV source missing from the complete captured teacher scope')
    proof,diagnostics=geometry_proof(proof_path,proof_sha,manifest_sha,[i['id'] for i in items])
    for measured in proof['measurements']:
        item=next(i for i in items if i['id']==measured['id'])
        if (measured.get('source_sha256')!=item['sha256'] or
                measured.get('source_group_id')!=item['source_group_id']):
            raise Error('measured shared bridge clock belongs to a different physical source/group')
    bridge_file=pinned({'path':'bridge_positions.safetensors','sha256':proof['bridge_positions_sha256']},proof_path.parent)
    bridge=load_file(str(bridge_file))['native_media_positions']
    teacher=next(iter(manifest['teacher_integrity']['before'].values()))['sha256']
    H.require_sha(amendment_sha,'selected dated H3 amendment')
    out.parent.mkdir(parents=True,exist_ok=True);stage=out.with_name(out.name+'.partial')
    if stage.exists():raise Error('interrupted native output retained; select a fresh destination')
    stage.mkdir();accepted=[];excluded=[];seen=set()
    for shard in manifest['shards']:
        tensors=load_file(str(cache/shard['file']));side=read((cache/shard['file']).with_suffix('.json'))
        if any(t.is_floating_point() and not bool(torch.isfinite(t).all()) for t in tensors.values()):
            raise Error('actual source shard has nonfinite values')
        for index,target in enumerate(side['targets']):
            ident=target['id'];item=items[target['item_index']]
            if ident!=item['id'] or ident not in av or target['target'] in seen:
                raise Error('teacher source/target must join a unique globally reviewed AV row')
            seen.add(target['target']);entry=av[ident];record=source[ident]
            contract=target['h3_source_contract'];H.check_source_record(contract)
            if (contract['source_row_sha256']!=entry['source_row_sha256'] or entry['source_row_sha256']!=R.digest(record) or
                    contract['source_group_id']!=record['provenance']['group_id'] or contract['split']!=entry['split'] or
                    contract['video_sha256']!=entry['asset_sha256'] or
                    item['prompt']!=record['messages'][0]['content']+SUFFIX or
                    item.get('source_prompt_sha256')!=entry['prompt_sha256']):
                raise Error('teacher conditioning, physical origin, row, source group or split differs')
            pair=H.validate_pair(tensors,target,index,float(manifest['plan']['cfg']))
            prefix=M.prompt_prefix(tokenizer,item['prompt'])
            batch=H.pack_pair(pair,torch.tensor([prefix],dtype=torch.int64),bridge,media_token_id=M.MEDIA_PLACEHOLDER)
            diagnostic=diagnostics.get(target['target'])
            if not diagnostic or diagnostic['id']!=ident or diagnostic['prefix_ids']!=prefix or diagnostic['noise_sha256']!=pair['noise_sha256']:
                raise Error('genuine native prefix/noise/source diagnostic join differs')
            diagpath=pinned({'path':diagnostic['diagnostic_file'],'sha256':diagnostic['sha256']},proof_path.parent)
            expected=M.inflate_native_batch(load_file(str(diagpath)),diagnostic['storage'])
            if set(batch)!=set(expected) or any(not torch.equal(batch[k],expected[k]) for k in batch):
                raise Error('reconstructed batch differs bitwise from authentic parent packers/layout proof')
            if entry['train_admitted'] is not True:
                excluded.append({'target':target['target'],'source_id':ident,'split':entry['split'],'blockers':entry['blockers']});continue
            if entry['split']!='train' or entry['blockers']:raise Error('held/excluded AV source cannot enter optimizer data')
            name='h3-native-'+str(target['target']).zfill(8)
            stored,storage=M.compact_native_batch(batch,BANDS);save_file(stored,str(stage/(name+'.safetensors')))
            receipt={'schema':'chimera_recovery.native_flow_cache.v1','sealed':True,'all_finite':True,'all_structural_ok':True,
                'guided_teacher':True,'no_clean_target_in_inputs':True,'cache_sha256':H.sha(stage/(name+'.safetensors')),
                'corpus_sha256':admission_sha,'teacher_sha256':teacher,'recovery_amendment_sha256':amendment_sha,
                'target_kind':'canonical_velocity','prediction_kind':'raw','route':'joint_av','input_layout':'feature',
                'target_layout':'emission','parent_signs':{'2':1.,'3':1.},'noisy_feature_bands':{'2':[384,480],'3':[480,512]},
                'noisy_feature_bands_sha256':R.digest({'2':[384,480],'3':[480,512]}),'storage':storage,
                'av_alignment':{'same_origin':True,'source_media_sha256':entry['asset_sha256'],'start_seconds':0.,
                    'end_seconds':22/24,'window_sha256':contract['window_sha256']},
                'source':{'manifest_sha256':manifest_sha,'shard_sha256':shard['sha256'],'sidecar_sha256':shard['sidecar_sha256'],
                    'source_id':ident,'source_row_sha256':entry['source_row_sha256'],'recovery_group_id':entry['recovery_group_id'],
                    'target_index':target['target'],'item_index':target['item_index'],'cfg':manifest['plan']['cfg'],
                    'geometry_proof_sha256':proof_sha,'bridge_positions_sha256':proof['bridge_positions_sha256'],
                    'diagnostic_sha256':diagnostic['sha256'],'conditioning_sha256':R.digest(item['prompt'])},
                'h3_clock_contract':{'video_shift':12.,'audio_shift':3.,'sigma_video':pair['sigma_video'],
                    'sigma_audio':pair['sigma_audio'],'audio_flat_to_native_slope':pair['slope_audio'],
                    'audio_target':'captured v_audio_native; flat guided velocity divided by exact dual-clock slope',
                    'loss_audio_extra_slope':False,'packed_layout':'full_text|channel_major_stereo_audio|video_patches',
                    'bridge_clock_scope':proof['group_semantics'],'parent_PackedLayout_bitwise_equal':True,
                    'latent_audio_frames':37,'source_audio_samples':contract['measured']['audio_samples'],
                    'waveform_padding_samples':0,'latent_tail_scope':'retain actual VAE37frames; partial final nominal hop is not claimed as additional source time'},
                'conditioning_contract':{'kind':'native_parent_chat_prompt','template_sha256':R.digest(tokenizer.chat_template),
                    'prompt_sha256':entry['prompt_sha256'],'reference_answer_in_inputs':False},'bitwise_reconstruction':True}
            R.ReadyObjectiveAdapter(receipt).bind('joint_av',batch,BANDS,parent_signs=SIGNS)
            (stage/(name+'.json')).write_bytes(R.canonical(receipt)+b'\n')
            accepted.append({'id':name,'source_id':ident,'unit':'h3_dit_L25_curve6','route':'joint_av',
                'input_layout':'feature','target_layout':'emission','recovery_group_id':entry['recovery_group_id'],
                'parent_signs':{'2':1.,'3':1.},'storage':storage,'tensors':{'path':name+'.safetensors','sha256':receipt['cache_sha256']},
                'teacher_receipt':{'path':name+'.json','sha256':H.sha(stage/(name+'.json'))}})
    if len(seen)!=manifest['targets'] or seen!=set(diagnostics):raise Error('all sealed native targets must be traversed exactly once')
    result={'schema':'chimera_recovery.native_batches.v1','sealed':bool(accepted),'split':'train','heldout_groups_excluded':True,
        'complete_source_traversal':True,'corpus_sha256':admission_sha,'source_cache_manifest_sha256':manifest_sha,
        'source_items':pin(items_path),'source_admission':pin(admission_path),'source':pin(source_path),
        'geometry_proof':pin(proof_path),'batches':accepted,'excluded_heldout_targets':excluded,
        'uncached_held_source_ids':sorted(i for i,r in av.items() if not r['train_admitted'] and i not in {v['id'] for v in items}),
        'adapter_kind':'exact_h3_joint_native_dualclock_guided_velocity','adapter_sha256':H.sha(Path(__file__)),
        'production_training_started':False,'production_training_ready':False}
    (stage/'manifest.json').write_bytes(R.canonical(result)+b'\n');stage.rename(out);return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('prepare-source')
    for name in ('joint-admission','admission-helper','out'):s.add_argument('--'+name,type=Path,required=True)
    for name in ('joint-sha256','helper-sha256'):s.add_argument('--'+name,required=True)
    s=sub.add_parser('prepare-cache')
    for name in ('cache','items','admission','proof','tokenizer-dir','out'):s.add_argument('--'+name,type=Path,required=True)
    for name in ('manifest-sha256','admission-sha256','proof-sha256','amendment-sha256'):s.add_argument('--'+name,required=True)
    s.add_argument('--aliases',type=Path)
    a=p.parse_args(argv)
    if a.command=='prepare-source':result=prepare_source(a.joint_admission,a.joint_sha256,a.admission_helper,a.helper_sha256,a.out)
    else:
        from transformers import AutoTokenizer
        D=module('recovery_h3_native_tokenizer',HERE/'audio_dialogue.py')
        for name,value in D.TOKENIZER_PINS.items():pinned({'path':str(a.tokenizer_dir/name),'sha256':value})
        tokenizer=AutoTokenizer.from_pretrained(a.tokenizer_dir,local_files_only=True,use_fast=True,trust_remote_code=False)
        result=prepare_cache(a.cache,a.manifest_sha256,a.items,a.admission,a.admission_sha256,a.proof,a.proof_sha256,
            tokenizer,a.amendment_sha256,a.out,aliases=read(a.aliases) if a.aliases else None)
    print(json.dumps({'sealed':result['sealed'],'accepted':len(result.get('batches',result.get('joint_av_train_ids',[]))),
                      'training_started':False}));return 0

if __name__=='__main__':raise SystemExit(main())
