"""Source-sealed image editing supervision, separate from guided image KD.

The native QIMG graft has no complete parent text/RoPE parity. This is an
explicitly amended clean-reference/noisy-output bridge and supervised flow.
Genuine before/after VAE targets authenticate the source; no after target is
used as clean input or converted into a caption.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from functools import lru_cache
try:
    from . import h3_native as N
except ImportError:
    import h3_native as N
R,M,Error=N.R,N.M,N.Error
HERE=Path(__file__).resolve().parent
SCHEMA='chimera_recovery.image_edit_batches.v1'
OBJECTIVE='supervised_image_edit_rectified_flow'
STAGE='D3b_image_edit_supervised_flow'
POLICY='qimg_noisy_output_then_clean_reference_index_zero_time_v1'
SIGMAS=[.05,.3,.7,1.]
SEED=20261013
ROWS=1024
BANDS={1:{'feature':(512,576),'emission':(256,320)}}
VAE_SHA='a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f'
QIMG_SOURCE_SHA='14c805af8da13d31094c2c704e413cc282ef43f4afe17f77de9070bdca301f28'
DEPENDENCIES=('h3_native.py','h3_cache.py','media_cache.py','training.py','audio_dialogue.py')

def dependency_pins():
    return {name:N.H.sha(HERE/name) for name in DEPENDENCIES}

def tokenizer_pins(tokenizer_dir):
    D=N.module('recovery_image_edit_tokenizer_pins',HERE/'audio_dialogue.py')
    return {name:N.pin(N.pinned({'path':str(tokenizer_dir/name),'sha256':h})) for name,h in D.TOKENIZER_PINS.items()}

@lru_cache(maxsize=4)
def _tokenizer(directory,files):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(directory,local_files_only=True,use_fast=True,trust_remote_code=False)

def validate_prefix(record,prefix,pins,root,aliases=None):
    D=N.module('recovery_image_edit_tokenizer_check',HERE/'audio_dialogue.py')
    if set(pins)!=set(D.TOKENIZER_PINS) or any(pins[n]['sha256']!=h for n,h in D.TOKENIZER_PINS.items()):
        raise Error('editing requires exact native parent tokenizer source pins')
    paths={n:N.pinned(p,root,aliases) for n,p in pins.items()}
    directory=paths['tokenizer.json'].parent
    if any(p.parent!=directory or p.name!=n for n,p in paths.items()):
        raise Error('editing tokenizer files must form one actual local parent tokenizer')
    tokenizer=_tokenizer(str(directory),tuple(sorted((n,pins[n]['sha256']) for n in pins)))
    if R.digest(pack_prefix(record,tokenizer))!=R.digest(prefix):
        raise Error('editing native prefix differs from authentic source-only parent rendering')
    return True

def source_messages(record):
    media=record.get('media',[]);messages=record.get('messages',[]);p=record.get('provenance',{})
    if (record.get('schema')!='recovery_corpus.v1' or record.get('route')!='image_edit' or
            [m.get('role') for m in messages]!=['user','assistant'] or len(media)!=2 or
            [m.get('type') for m in media]!=['image','image'] or [m.get('role') for m in media]!=['input','target'] or
            any(m.get('source_verified') is not True for m in media) or media[0].get('sha256')==media[1].get('sha256') or
            record.get('supervision',{}).get('message_mask') not in ([0,1],[False,True]) or
            record.get('split') not in {'train','validation','test'} or p.get('split')!=record['split'] or
            p.get('source_year') not in {2025,2026} or record.get('tools') or any(m.get('tool_calls') for m in messages)):
        raise Error('editing requires exact before-input/after-target roles and source-only user instruction')
    text=messages[0].get('content')
    if not isinstance(text,str) or not text.strip() or any(t in text for t in ('<|im_start|>','<|im_end|>')):
        raise Error('editing instruction must be the original unambiguous user text')
    return [{'role':'user','content':text}]

def source_candidate(record):
    source_messages(record)
    before,after=record['media'];p=record['provenance']
    for m in (before,after):N.pinned(m)
    oracle=N.read(N.pinned({'path':p['groundtruth_path'],'sha256':p['groundtruth_sha256']}))
    ground=oracle['groundtruth'];answer=json.loads(record['messages'][-1]['content'])
    expected={'after_objects':[{k:o[k] for k in ('color','shape','x','y')} for o in ground['after_objects']]}
    if (R.digest(answer)!=R.digest(expected) or ground.get('changed_pixels',0)<=0 or
            oracle.get('family')!=p.get('template_family')):
        raise Error('editing target metadata differs from the sealed transformation oracle')
    return {'record':record,'asset_path':before['path'],'asset_sha256':before['sha256'],
        'target_asset_path':after['path'],'target_asset_sha256':after['sha256'],
        'groundtruth_sha256':p['groundtruth_sha256'],'source_scope':'unchanged authored before/after image edit'}

def prepare_source(parent_path, parent_sha, helper_path, helper_sha, out):
    """Preserve prior admissions and re-run every original global graph node."""
    if out.exists():raise Error('editing admission needs a fresh immutable derivative')
    path=N.pinned({'path':str(parent_path),'sha256':parent_sha});parent=N.read(path)
    if (parent.get('sealed') is not True or parent.get('schema')!='recovery_audio_admission.v1' or
            parent.get('complete_source_traversal') is not True or parent.get('connected_groups_verified') is not True or
            R.digest({k:v for k,v in parent.items() if k!='manifest_sha256'})!=parent.get('manifest_sha256')):
        raise Error('complete globally screened prior source admission required')
    helper=N.module('recovery_image_edit_admission',N.pinned({'path':str(helper_path),'sha256':helper_sha}))
    originals=N.rows(N.pinned(parent['sources']['source'],path.parent))
    full=N.rows(N.pinned(parent['sources']['source_inventory'],path.parent));by_id={r['id']:r for r in full}
    old={r['id']:r for r in N.rows(N.pinned(parent['outputs']['admission'],path.parent))}
    if set(old)!={r['id'] for r in originals}:raise Error('prior source/admission scopes differ')
    candidates=[]
    for r in originals:
        row=old[r['id']]
        if R.digest(r)!=row['source_row_sha256'] or R.digest(r)!=R.digest(by_id[r['id']]):
            raise Error('earlier source row changed')
        keep={k:v for k,v in row.items() if k not in {'schema','id','item_id','route','split','recovery_group_id',
            'connected_source_groups','source_group_id','source_row_sha256','authored_answer_sha256','prompt_sha256',
            'source_train_eligible','train_admitted','native_train_ready','encoder_status','source_pins'}}
        candidates.append({**keep,'record':r,'original_source_admission':N.pin(path)})
    edits=[source_candidate(r) for r in full if r['route']=='image_edit']
    if not edits or any(c['record']['id'] in old for c in edits):raise Error('new complete edit scope is empty or overlapping')
    candidates.extend(edits);out.mkdir(parents=True)
    source=out/'source_ready.jsonl';source.write_bytes(b''.join(R.canonical(c['record'])+b'\n' for c in candidates))
    specs={k:parent['sources'][k] for k in N.COMMON}
    specs.update(source=N.pin(source),source_inventory=parent['sources']['source_inventory'],
                 prior_h3_admission=N.pin(path),prior_h3_rows=parent['outputs']['admission'],source_admission_helper=N.pin(helper_path))
    helper.seal_admission(specs=specs,candidates=candidates,inventory=full,output=out/'global_admission')
    now={r['id']:r for r in N.rows(out/'global_admission/admission.jsonl')};changes=[]
    for ident,prior in old.items():
        row=now[ident]
        if set(prior['blockers'])-set(row['blockers']) or (not prior['train_admitted'] and row['train_admitted']):
            raise Error('new edit traversal lifted a prior exclusion')
        if (prior['recovery_group_id'],prior['train_admitted'])!=(row['recovery_group_id'],row['train_admitted']):changes.append(ident)
    result={'schema':'chimera_recovery.image_edit_source_preparation.v1','sealed':True,'source':N.pin(source),
        'source_admission':N.pin(out/'global_admission/manifest.json'),'prior_source_admission':N.pin(path),
        'prior_native_source_changes':changes,'edit_source_ids':[c['record']['id'] for c in edits],
        'edit_train_ids':[c['record']['id'] for c in edits if now[c['record']['id']]['train_admitted']],
        'full_source_inventory':len(full),'full_candidates':len(candidates),'training_started':False,'native_train_ready':False}
    (out/'preparation.json').write_bytes(R.canonical(result)+b'\n');return result

def pack_prefix(record,tokenizer,*,max_length=8192):
    messages=source_messages(record);options=dict(add_generation_prompt=True,enable_thinking=False,tools=None)
    text=tokenizer.apply_chat_template(messages,tokenize=False,**options)
    ids=R.native_token_ids(tokenizer.apply_chat_template(messages,tokenize=True,**options))
    if (not isinstance(text,str) or not ids or ids!=R.native_token_ids(tokenizer(text,add_special_tokens=False)) or
            not getattr(tokenizer,'chat_template',None) or any(type(i) is not int or not 0<=i<R.TEXT_VOCAB for i in ids)):
        raise Error('editing requires exact authentic parent generation-prefix IDs')
    if type(max_length) is not int or len(ids)+2*ROWS>max_length:raise Error('editing full reference/output exceeds context; no truncation')
    return {'messages':messages,'parent_text_ids':ids,'input_ids':ids+[M.MEDIA_PLACEHOLDER]*(2*ROWS),
        'prompt_length':len(ids),'source_row_sha256':R.digest(record),'template_sha256':R.digest(tokenizer.chat_template),
        'rendered_sha256':hashlib.sha256(text.encode()).hexdigest(),'assistant_metadata_in_prefix':False,'policy':POLICY}

def noise_seed(record,sigma):
    if sigma not in SIGMAS or type(sigma) not in (int,float):raise Error('editing requires prescribed positive sigma')
    return int(R.digest({'source_row_sha256':R.digest(record),'sigma':sigma,'seed':SEED,'policy':POLICY})[:16],16)&((1<<63)-1)

def build_batch(record,prefix,before,after,*,sigma):
    import torch
    source_messages(record);n=prefix.get('prompt_length');ids=prefix.get('parent_text_ids')
    if (prefix.get('messages')!=source_messages(record) or prefix.get('source_row_sha256')!=R.digest(record) or
            prefix.get('policy')!=POLICY or prefix.get('assistant_metadata_in_prefix') is not False or
            type(n) is not int or n<1 or len(ids)!=n or prefix['input_ids']!=ids+[M.MEDIA_PLACEHOLDER]*(2*ROWS)):
        raise Error('editing prefix changed or clean after metadata leaked')
    for x in (before,after):
        if x.shape!=(1,ROWS,64) or x.dtype!=torch.float32 or x.device.type!='cpu' or x.requires_grad or not bool(torch.isfinite(x).all()):
            raise Error('editing needs detached genuine1024x64 CPU model-space image patches')
    epsilon=torch.randn(after.shape,generator=torch.Generator().manual_seed(noise_seed(record,sigma)),dtype=torch.float32)
    noisy=(1.-sigma)*after+sigma*epsilon
    features=torch.zeros(1,2*ROWS,6144);targets=torch.zeros_like(features)
    # Actual parent image ordering: generated/noisy output then reference.
    features[:,:ROWS,512:576]=noisy;features[:,ROWS:,512:576]=before
    targets[:,:ROWS,256:320]=epsilon-after
    positions=torch.zeros(1,2*ROWS,4);y,x=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
    for start,role in ((0,0),(ROWS,1)):
        positions[:,start:start+ROWS,0]=role
        positions[:,start:start+ROWS,1]=y.flatten().float()/31
        positions[:,start:start+ROWS,2]=x.flatten().float()/31
    levels=torch.zeros(1,2*ROWS);levels[:,:ROWS]=sigma
    weights=torch.zeros_like(levels);weights[:,:ROWS]=1
    return {'input_ids':torch.tensor([prefix['input_ids']],dtype=torch.int64),
        'native_media_features':features,'native_media_targets':targets,
        'native_media_type_ids':torch.ones(1,2*ROWS,dtype=torch.int64),
        'native_media_indices':torch.arange(n,n+2*ROWS,dtype=torch.int64).view(1,-1),
        'native_media_valid_mask':torch.ones(1,2*ROWS,dtype=torch.bool),
        'native_media_positions':positions,'native_media_noise_levels':levels,'native_media_loss_mask':weights}

def velocity_loss(prediction,batch):
    import torch
    if prediction.shape!=(1,2*ROWS,6144) or not prediction.is_floating_point() or not bool(torch.isfinite(prediction).all()):
        raise Error('editing requires a finite native raw image head')
    expected=torch.cat((torch.ones(1,ROWS),torch.zeros(1,ROWS)),1)
    if not torch.equal(batch['native_media_loss_mask'].cpu(),expected):raise Error('reference rows must receive zero direct flow loss')
    # Existing QIMG sign -1 means raw head is canonical epsilon-clean.
    return (prediction[:,:ROWS,256:320].float()-batch['native_media_targets'][:,:ROWS,256:320].to(prediction.device).float()).square().mean()

def verified_pairs(proof_path,proof_sha):
    import torch
    from safetensors.torch import load_file
    path=N.pinned({'path':str(proof_path),'sha256':proof_sha});proof=N.read(path)
    if (proof.get('schema')!='chimera_recovery.qimg_edit_vae_pairs.v1' or proof.get('sealed') is not True or
            proof.get('finite') is not True or proof.get('all48_repeated_encodes_bitwise') is not True or
            proof.get('all48_native_packs_bitwise') is not True or proof.get('vae_file_and_state_unchanged') is not True or
            proof.get('vae_file_sha256')!=VAE_SHA or len(proof.get('records',[]))!=24 or
            proof['source_and_code_pins'].get('/opt/ComfyUI/comfy/ldm/qwen_image/model.py')!=QIMG_SOURCE_SHA):
        raise Error('complete genuine before/after VAE and native image packing evidence required')
    result={}
    for item in proof['records']:
        if item['id'] in result:raise Error('editing source appears twice in VAE proof')
        data=load_file(str(N.pinned(item['tensors'],path.parent)))
        if any(not bool(torch.isfinite(v).all()) for v in data.values()):raise Error('editing genuine source tensor is nonfinite')
        for image in item['images']:
            role=image['role'];raw,model,packed=data[role+'.raw'],data[role+'.model'],data[role+'.packed']
            if (role not in {'before','after'} or raw.shape!=(1,16,1,64,64) or model.shape!=raw.shape or
                    packed.shape!=(1,ROWS,64) or image.get('repeat_encode_bitwise') is not True or
                    image.get('native_pack_bitwise') is not True or raw.dtype!=torch.float32 or model.dtype!=torch.float32 or
                    not torch.equal(model.reshape(1,16,1,32,2,32,2).permute(0,2,3,5,1,4,6).reshape(1,ROWS,64),packed)):
                raise Error('editing genuine model-space/nativepack contract changed')
            N.pinned(image['reconstruction_png'],path.parent)
        if {i['role'] for i in item['images']}!={'before','after'}:raise Error('both original image roles must be proved')
        result[item['id']]={'record':item,'data':data}
    return proof,result

def validate_batch(batch,receipt,root,*,aliases=None):
    import torch
    if (receipt.get('schema')!='chimera_recovery.image_edit_cache.v1' or receipt.get('sealed') is not True or
            receipt.get('objective')!=OBJECTIVE or receipt.get('guided_teacher') is not False or
            receipt.get('prediction_kind')!='raw' or receipt.get('parent_sign')!=-1. or
            receipt.get('no_clean_after_in_inputs') is not True or receipt.get('adapter_sha256')!=N.H.sha(Path(__file__)) or
            receipt.get('dependencies')!=dependency_pins() or receipt.get('policy')!=POLICY or
            receipt.get('stage')!=STAGE or receipt.get('supervised_units')!=ROWS*64 or
            receipt.get('denominator_kind')!='native_output_scalar_mse' or
            receipt.get('reference_rows_direct_loss')!=0. or receipt.get('noise_seed')!=noise_seed(
                N.read(N.pinned(receipt['source_record'],root,aliases)),receipt['sigma'])):
        raise Error('editing supervised native contract changed')
    R.require_sha(receipt.get('amendment_sha256'),'selected editing amendment')
    source=N.read(N.pinned(receipt['source_record'],root,aliases));prefix=N.read(N.pinned(receipt['prefix'],root,aliases))
    validate_prefix(source,prefix,receipt['tokenizer_files'],root,aliases)
    admission_path=N.pinned(receipt['source_admission'],root,aliases);admission=N.read(admission_path)
    if (admission.get('sealed') is not True or admission.get('complete_source_traversal') is not True or
            admission.get('connected_groups_verified') is not True or
            R.digest({k:v for k,v in admission.items() if k!='manifest_sha256'})!=admission.get('manifest_sha256')):
        raise Error('editing source admission seal changed')
    row=next(r for r in N.rows(N.pinned(admission['outputs']['admission'],admission_path.parent,aliases)) if r['id']==source['id'])
    original=next(r for r in N.rows(N.pinned(admission['sources']['source'],admission_path.parent,aliases)) if r['id']==source['id'])
    if (row['train_admitted'] is not True or row['split']!='train' or row['blockers'] or
            row['source_row_sha256']!=R.digest(source) or R.digest(original)!=R.digest(source) or
            receipt.get('source_id')!=source['id'] or receipt.get('source_row_sha256')!=R.digest(source) or
            row['recovery_group_id']!=receipt['recovery_group_id']):
        raise Error('editing source/group is held, excluded or changed')
    proof_path=N.pinned(receipt['vae_proof'],root,aliases);_,pairs=verified_pairs(proof_path,receipt['vae_proof']['sha256'])
    pair=pairs[source['id']];images={v['role']:v for v in pair['record']['images']}
    if (pair['record']['source_row_sha256']!=R.digest(source) or
            images['before']['source_sha256']!=source['media'][0]['sha256'] or
            images['after']['source_sha256']!=source['media'][1]['sha256']):raise Error('editing VAE belongs to a different original pair')
    expected=build_batch(source,prefix,pair['data']['before.packed'],pair['data']['after.packed'],sigma=receipt['sigma'])
    if set(batch)!=set(expected) or any(not torch.equal(batch[k].cpu(),v) for k,v in expected.items()):
        raise Error('editing batch differs from genuine before/noisy after/reference chart reconstruction')
    return True

class ImageEditAdapter:
    """Source authentication followed by native raw-head velocity MSE."""
    def __init__(self,receipt,root,*,aliases=None):self.receipt,self.root,self.aliases=receipt,Path(root),aliases
    def bind(self,batch):
        validate_batch(batch,self.receipt,self.root,aliases=self.aliases)
        return lambda prediction:velocity_loss(prediction,batch)

def full_depth_backward(module,hybrid,batch,compute,*,loss_scale=1.):
    """One streamed backward callback; creates no optimizer or teacher targets."""
    import math
    import torch
    try:from . import train_media as T
    except ImportError:import train_media as T
    if (type(loss_scale) not in (int,float) or not math.isfinite(loss_scale) or loss_scale<=0 or
            hybrid.bands[1]['feature']!=(512,576) or hybrid.bands[1]['emission']!=(256,320)):
        raise Error('editing full-depth callback needs exact native type1 bands and positive loss scale')
    types,mask=batch['native_media_type_ids'],batch['native_media_loss_mask']
    if (not bool((types==1).all()) or mask.shape!=types.shape or not bool(torch.isfinite(mask).all()) or
            bool(((mask!=0)&(mask!=1)).any()) or int(mask.sum())<1):
        raise Error('editing callback requires binary generated/reference row ownership')
    segment=T.make_segment(batch,media_placeholder=hybrid.media_placeholder)
    return module.full_depth_forward_backward(hybrid,segment,
        lambda normed,context:compute(hybrid.emission(normed,context)),loss_scale=float(loss_scale))[0]

def prepare(proof_path,proof_sha,source_path,admission_path,admission_sha,tokenizer,amendment_sha,out,*,tokenizer_dir,aliases=None):
    import torch
    from safetensors.torch import save_file
    if out.exists():raise Error('editing native preparation requires a fresh immutable output')
    R.require_sha(amendment_sha,'selected edit amendment')
    proof,pairs=verified_pairs(proof_path,proof_sha)
    token_pins=tokenizer_pins(tokenizer_dir)
    admission=N.read(N.pinned({'path':str(admission_path),'sha256':admission_sha}))
    if (admission.get('sealed') is not True or admission.get('complete_source_traversal') is not True or
            admission.get('connected_groups_verified') is not True or
            R.digest({k:v for k,v in admission.items() if k!='manifest_sha256'})!=admission.get('manifest_sha256')):
        raise Error('editing needs the complete global admission seal')
    originals={r['id']:r for r in N.rows(source_path)}
    if N.H.sha(source_path)!=admission['sources']['source']['sha256']:raise Error('editing original source set changed')
    admitted={r['id']:r for r in N.rows(N.pinned(admission['outputs']['admission'],admission_path.parent,aliases))}
    edits={i:r for i,r in originals.items() if r['route']=='image_edit'}
    if set(edits)!=set(pairs):raise Error('editing requires complete all24 authentic VAE targets')
    out.parent.mkdir(parents=True,exist_ok=True);temporary=Path(tempfile.mkdtemp(prefix='.image-edit-',dir=out.parent))
    batches=[];excluded=[]
    for ident,record in edits.items():
        row=admitted[ident];pair=pairs[ident];images={i['role']:i for i in pair['record']['images']}
        if (row['source_row_sha256']!=R.digest(record) or pair['record']['source_row_sha256']!=R.digest(record) or
                images['before']['source_sha256']!=record['media'][0]['sha256'] or
                images['after']['source_sha256']!=record['media'][1]['sha256']):raise Error('editing source/admission/genuine pair joins differ')
        prefix=pack_prefix(record,tokenizer)
        validate_prefix(record,prefix,token_pins,out.parent,aliases)
        if not row['train_admitted']:
            excluded.append({'id':ident,'split':row['split'],'blockers':row['blockers']});continue
        for name,value in ((ident+'.source.json',record),(ident+'.prefix.json',prefix)):
            (temporary/name).write_bytes(R.canonical(value)+b'\n')
        for index,sigma in enumerate(SIGMAS):
            batch=build_batch(record,prefix,pair['data']['before.packed'],pair['data']['after.packed'],sigma=sigma)
            name=ident+'.sigma'+str(index);stored,storage=M.compact_native_batch(batch,BANDS)
            save_file(stored,str(temporary/(name+'.safetensors')))
            receipt={'schema':'chimera_recovery.image_edit_cache.v1','sealed':True,'objective':OBJECTIVE,'stage':STAGE,
                'route':'image_edit','guided_teacher':False,'teacher_kd_weight':0.,'supervised_flow_weight':1.,'ce_weight':0.,
                'prediction_kind':'raw','target_kind':'canonical_velocity','parent_sign':-1.,'no_clean_after_in_inputs':True,
                'source_record':{'path':ident+'.source.json','sha256':N.H.sha(temporary/(ident+'.source.json'))},
                'prefix':{'path':ident+'.prefix.json','sha256':N.H.sha(temporary/(ident+'.prefix.json'))},
                'source_admission':N.pin(admission_path),'vae_proof':N.pin(proof_path),'source_id':ident,
                'source_row_sha256':R.digest(record),'recovery_group_id':row['recovery_group_id'],'sigma':sigma,
                'noise_seed':noise_seed(record,sigma),'policy':POLICY,'native_parent_full_parity':False,
                'reference_time_policy':'clean reference sigma0; explicit index_timestep_zero-style native convention',
                'reference_rows_direct_loss':0.,'supervised_units':ROWS*64,'denominator_kind':'native_output_scalar_mse',
                'amendment_sha256':amendment_sha,'adapter_sha256':N.H.sha(Path(__file__)),
                'dependencies':dependency_pins(),'tokenizer_files':token_pins,
                'tensors_sha256':N.H.sha(temporary/(name+'.safetensors')),'storage':storage}
            (temporary/(name+'.json')).write_bytes(R.canonical(receipt)+b'\n')
            batches.append({'id':name,'source_id':ident,'route':'image_edit','unit':'qimg_mmdit_L30_img',
                'recovery_group_id':row['recovery_group_id'],'storage':storage,
                'tensors':{'path':name+'.safetensors','sha256':receipt['tensors_sha256']},
                'teacher_receipt':{'path':name+'.json','sha256':N.H.sha(temporary/(name+'.json'))}})
    if not batches:raise Error('no globally admitted edit supervision')
    result={'schema':SCHEMA,'sealed':True,'stage':STAGE,'objective':OBJECTIVE,'complete_source_traversal':True,
        'source':N.pin(source_path),'source_admission':N.pin(admission_path),'vae_proof':N.pin(proof_path),
        'batches':batches,'excluded':excluded,'sigma_grid':SIGMAS,'amendment_sha256':amendment_sha,
        'adapter_sha256':N.H.sha(Path(__file__)),'dependencies':dependency_pins(),
        'tokenizer_files':token_pins,'guided_teacher':False,'native_parent_full_parity':False,
        'training_started':False,'production_training_ready':False}
    (temporary/'manifest.json').write_bytes(R.canonical(result)+b'\n');temporary.rename(out);return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare-source')
    for name in ('parent-admission','admission-helper','out'):a.add_argument('--'+name,type=Path,required=True)
    for name in ('parent-sha256','helper-sha256'):a.add_argument('--'+name,required=True)
    a=sub.add_parser('prepare-cache')
    for name in ('proof','source','admission','tokenizer-dir','out'):a.add_argument('--'+name,type=Path,required=True)
    for name in ('proof-sha256','admission-sha256','amendment-sha256'):a.add_argument('--'+name,required=True)
    a.add_argument('--aliases',type=Path)
    args=p.parse_args(argv)
    if args.command=='prepare-source':report=prepare_source(args.parent_admission,args.parent_sha256,args.admission_helper,args.helper_sha256,args.out)
    else:
        from transformers import AutoTokenizer
        D=N.module('recovery_image_edit_tokenizer',HERE/'audio_dialogue.py')
        for name,h in D.TOKENIZER_PINS.items():N.pinned({'path':str(args.tokenizer_dir/name),'sha256':h})
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_dir,local_files_only=True,use_fast=True,trust_remote_code=False)
        report=prepare(args.proof,args.proof_sha256,args.source,args.admission,args.admission_sha256,tokenizer,
            args.amendment_sha256,args.out,tokenizer_dir=args.tokenizer_dir,aliases=N.read(args.aliases) if args.aliases else None)
    print(json.dumps({'sealed':report['sealed'],'accepted':len(report.get('batches',report.get('edit_train_ids',[]))),'training_started':False}))
    return 0

if __name__=='__main__':raise SystemExit(main())
