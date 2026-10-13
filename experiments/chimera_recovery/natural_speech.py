"""Source-preserving ASR prompt repair and explicit unadmitted inventory audit.

Collection dates are never treated as original data dates. This utility repairs
the authenticated transcript cue in supported existing registry rows; it does
not perform global admission, infer a parent teacher, or open model weights.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

SCHEMA='chimera_recovery.natural_speech_repair.v1'
POLICY='strip_verified_expected_transcript_cue.v1'
PROMPT='Transcribe the speech in this audio.'

class Error(ValueError):pass

def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()
def valid_sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def audio_assets(record):
    # Existing registry normalization retained kind/media aliases. Inventory
    # must count these authentic sidecars even when canonical type is missing.
    return [m for m in record.get('media',[]) if any(m.get(key) in {'audio','audio_input','audio_target'}
                                                   for key in ('type','kind','media'))]

def repair(record):
    """Derive a clean observation prompt with original answer/lineage preserved."""
    origin=record.get('provenance',{}).get('registry_origin',{})
    if (record.get('schema')!='recovery_corpus.v1' or record.get('route')!='audio_understand' or
            origin.get('source_dataset')!='openslr/librispeech_asr' or
            not origin.get('task_type','').startswith('speech_transcription_')):
        raise Error('unsupported original natural ASR source contract')
    messages=record.get('messages',[]);assets=audio_assets(record)
    if [m.get('role') for m in messages]!=['user','assistant'] or len(assets)!=1:
        raise Error('natural ASR requires one original waveform and terminal annotated answer')
    try:answer=json.loads(messages[-1]['content'])
    except (TypeError,ValueError):raise Error('annotated original ASR answer is not JSON') from None
    transcript=answer.get('transcript');asset=assets[0]
    if (not isinstance(transcript,str) or not transcript or not valid_sha(asset.get('sha256')) or
            answer.get('audio_sha256')!=asset['sha256'] or not isinstance(asset.get('path'),str) or
            type(asset.get('byte_size',asset.get('bytes'))) is not int or not origin.get('rid') or
            not valid_sha(origin.get('shard_sha256')) or not origin.get('shard_mirror') or
            not origin.get('license') or not record.get('split') or
            not record.get('provenance',{}).get('recovery_group_id')):
        raise Error('natural ASR original waveform/transcript/source/split identity incomplete')
    prompt=messages[0].get('content')
    marker='Expected transcript: '
    if not isinstance(prompt,str) or prompt.count(marker)!=1 or prompt.split(marker,1)[1]!=transcript:
        raise Error('expected transcript cue does not exactly match the original source annotation')
    parent_sha=digest(record);result=copy.deepcopy(record)
    result['id']='natural-asr-repair-'+digest({'source_id':record['id'],'source_row_sha256':parent_sha,'policy':POLICY})[:24]
    result['route']='speech_understand';result['coverage']=['speech_understand']
    result['messages']=[{'role':'user','content':PROMPT},{'role':'assistant','content':transcript}]
    result['supervision']={'assistant_only':True,'policy':'assistant_only','message_mask':[False,True]}
    result['media']=[{**asset,'type':'audio','role':'input','bytes':asset.get('byte_size',asset.get('bytes'))}]
    result['provenance'].update(parent_source_id=record['id'],parent_source_row_sha256=parent_sha,
        repair_policy=POLICY,original_user_prompt_sha256=digest(prompt),original_answer=answer,
        original_answer_sha256=digest(answer),original_audio_sha256=asset['sha256'],
        target_cue_removed=True,original_data_date_verified=False)
    blockers=list(dict.fromkeys(record.get('quality',{}).get('blockers',[])+[
        'original_data_date_2025_2026_unverified','global_source_admission_pending',
        'independent_waveform_transcript_check_pending','complete_h3_encoder_proof_pending']))
    result['quality']={**result.get('quality',{}),'train_ready':False,'source_train_eligible':False,'blockers':blockers}
    return result

def audit(source,expected_sha,out):
    source,out=Path(source),Path(out)
    if not valid_sha(expected_sha) or sha(source)!=expected_sha:raise Error('original normalized corpus pin differs')
    if out.exists():raise Error('preserve original audit; use a new immutable output')
    counts=Counter();audio_routes=Counter();tags=Counter();splits=Counter();quarantine=[];repaired=[];candidates=[];natural_assets=set()
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():continue
            r=json.loads(line);counts['all_rows']+=1
            assets=audio_assets(r)
            if assets:
                audio_routes[r.get('route','missing')]+=1
                for m in assets:tags[m.get('dataset','unspecified')]+=1
            origin=r.get('provenance',{}).get('registry_origin',{})
            if origin.get('source_dataset')!='openslr/librispeech_asr':continue
            counts['natural_speech_registry_rows']+=1;splits[r.get('split','missing')]+=1
            natural_assets.update(m['sha256'] for m in assets if valid_sha(m.get('sha256')))
            candidates.append({'id':r['id'],'route':r.get('route'),'source_row_sha256':digest(r),
                'assets':assets,'original_data_date_verified':False,
                'collection_dates':r.get('provenance',{}).get('source_dates',[]),
                'collection_date_basis':r.get('provenance',{}).get('source_date_basis'),
                'origin':origin,'train_admitted':False})
            if r.get('route')!='audio_understand':counts['generation_rows_retained_without_asr_rewrite']+=1;continue
            try:fixed=repair(r)
            except Error as e:quarantine.append({'id':r['id'],'reason':str(e),'source_row_sha256':digest(r)});continue
            repaired.append(fixed)
    if sha(source)!=expected_sha:raise Error('original normalized corpus changed during traversal')
    counts['natural_waveform_unique_source_hashes']=len(natural_assets)
    out.mkdir(parents=True)
    outputs={}
    for name,rows in [('blocked_repaired_asr.jsonl',repaired),('original_inventory.jsonl',candidates),('quarantine.jsonl',quarantine)]:
        path=out/name
        with path.open('xb') as f:
            for row in rows:f.write(canonical(row)+b'\n')
        outputs[name]={'path':name,'sha256':sha(path),'rows':len(rows)}
    manifest={'schema':SCHEMA,'source':{'path':str(source.resolve()),'sha256':expected_sha},
        'counts':dict(counts),'audio_media_routes':dict(audio_routes),'audio_asset_datasets':dict(tags),
        'natural_source_splits':dict(splits),'outputs':outputs,'policy':POLICY,
        'collection_dates_are_not_original_data_dates':True,'train_admitted':0,'native_ready':False,
        'original_media_and_registry_metadata_preserved':True,
        'limitations':['Repair removes an authenticated target cue; it supplies no independent transcript judgment.',
          'Original source dates need explicit 2025–2026 evidence before admission under the current user restriction.',
          'Generation metadata remains separate; these original rows are not new teacher outputs.']}
    manifest['seal_sha256']=digest(manifest);(out/'manifest.json').write_bytes(canonical(manifest)+b'\n')
    return manifest

def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--source-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv);print(json.dumps(audit(args.source,args.source_sha256,args.out)));return 0

if __name__=='__main__':raise SystemExit(main())
