"""Bounded speech-teacher corpus generation and independent transcript checks."""
import argparse, datetime, hashlib, json, os, re, resource, time, wave
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,data):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    os.replace(tmp,path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['tts','asr']);ap.add_argument('--root',required=True);ap.add_argument('--limit',type=int,required=True);args=ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(261013)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    root=Path(args.root).resolve();parent=root.parent
    plan=json.loads((root/'plan.json').read_text());assert sha(root/'plan.json')=='8d73be8dc81f8c72c89e08fe85faf65d202561afb87cc980086e8a11bc897653'
    records=plan['records'][:args.limit]
    teachers=Path(os.environ['CHIMERA_SPEECH_TEACHERS_DIR'])
    download=json.loads((teachers/'completion.json').read_text());assert download['complete']
    spec=next(m for m in download['models'] if ('TTS' if args.mode=='tts' else 'ASR') in m['model_id']);modeldir=teachers/Path(spec['path']).name
    for item in spec['files']:assert sha(modeldir/item['path'])==item['sha256']
    name=f'{args.mode}-{args.limit}.json';result={'schema':'chimera.speech-teacher-campaign.v1','mode':args.mode,'rows':[],'teacher':spec['model_id'],'revision':spec['revision'],'plan_sha256':sha(root/'plan.json'),'synthetic':True,'training_started':False,'production_admitted':False,'human_audio_reviewed':False,'dtype':'bfloat16','device':'cuda:0','gpu_name':torch.cuda.get_device_name(0),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':False}
    begin=time.monotonic()
    try:
        if args.mode=='tts':
            from qwen_tts import Qwen3TTSModel
            model=Qwen3TTSModel.from_pretrained(str(modeldir),device_map='cuda:0',dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True)
            shapes=[];original=model.model.generate
            def measured(*a,**kw):
                output=original(*a,**kw);shapes.append([list(c.shape) for c in output[0]]);return output
            model.model.generate=measured
            for source in records:
                record=dict(source);target=root/'assets'/source['id'];target.mkdir(parents=True,exist_ok=True);receipt=target/'tts.json'
                if receipt.exists():
                    record=json.loads(receipt.read_text());assert record['text']==source['text'] and record['speaker']==source['speaker'];assert sha(record['audio'])==record['audio_sha256'] and sha(record['h3_audio'])==record['h3_audio_sha256']
                else:
                    start=time.monotonic();waves,rate=model.generate_custom_voice(text=source['text'],language='English',speaker=source['speaker'],instruct='',do_sample=False,subtalker_dosample=False,max_new_tokens=256)
                    values=np.asarray(waves[0],dtype=np.float32);assert int(rate)==24000 and values.ndim==1 and np.isfinite(values).all() and values.size>6000 and np.max(np.abs(values))<=1
                    assert len(shapes[-1])==1 and shapes[-1][0][0]<256,'teacher hit generation cap'
                    raw=target/'original.wav';sf.write(raw,values,24000,subtype='PCM_16')
                    with wave.open(str(raw),'rb') as inp:params=inp.getparams();pcm=inp.readframes(inp.getnframes())
                    padding=(-values.size)%600;derived=target/'h3-aligned.wav'
                    with wave.open(str(derived),'wb') as out:out.setparams(params);out.writeframes(pcm+b'\0\0'*padding)
                    with wave.open(str(derived),'rb') as check:full=check.readframes(check.getnframes());assert full[:len(pcm)]==pcm and full[len(pcm):]==b'\0\0'*padding
                    record.update({'audio':str(raw),'audio_sha256':sha(raw),'h3_audio':str(derived),'h3_audio_sha256':sha(derived),'source_samples':int(values.size),'source_rate':24000,'channels':1,'derived_samples':int(values.size+padding),'zero_tail_samples':int(padding),'source_pcm_prefix_bitwise_preserved':True,'source_trimmed_samples':0,'h3_target_samples':int((values.size+padding)*4//3),'expected_h3_frames':int((values.size+padding)//600),'codec_shape':shapes[-1][0],'hit_cap':False,'rms':float(np.sqrt(np.mean(values.astype(np.float64)**2))),'peak':float(np.max(np.abs(values))),'inference_seconds':time.monotonic()-start,'speech_teacher_revision':spec['revision']})
                    assert record['rms']>1e-4;save(receipt,record)
                result['rows'].append(record);save(root/name,result);print(json.dumps({'mode':'tts','id':record['id'],'rows':len(result['rows']),'seconds':record['inference_seconds']}),flush=True)
        else:
            from qwen_asr import Qwen3ASRModel
            source_manifest=json.loads((root/f'tts-{args.limit}.json').read_text());assert source_manifest['complete']
            model=Qwen3ASRModel.from_pretrained(str(modeldir),dtype=torch.bfloat16,device_map='cuda:0',max_inference_batch_size=1,max_new_tokens=256,attn_implementation='eager',local_files_only=True)
            normal=lambda s:re.findall('[a-z0-9]+',s.lower())
            for source in source_manifest['rows']:
                assert sha(source['h3_audio'])==source['h3_audio_sha256']
                values,rate=sf.read(source['h3_audio'],dtype='float32');start=time.monotonic();outputs=model.transcribe(audio=(values,int(rate)),language='English');assert len(outputs)==1
                text=outputs[0].text;exact=normal(source['text'])==normal(text)
                record={'id':source['id'],'group_id':source['group_id'],'split':'train','text':source['text'],'transcript':text,'language':outputs[0].language,'audio_sha256':source['h3_audio_sha256'],'transcript_exact_after_case_punctuation':exact,'eligible_for_global_admission':exact,'failure_reason':None if exact else 'independent teacher transcript mismatch','inference_seconds':time.monotonic()-start}
                result['rows'].append(record);save(root/name,result);print(json.dumps(record),flush=True)
            result['eligible_count']=sum(r['eligible_for_global_admission'] for r in result['rows']);result['quarantined_count']=len(result['rows'])-result['eligible_count']
            if args.limit==4:assert result['eligible_count']==4,'four-speaker GPU canary must match before full expansion'
        result['complete']=len(result['rows'])==len(records)
    except Exception as error:
        result['failure']=type(error).__name__+': '+str(error);raise
    finally:
        result['elapsed_seconds']=time.monotonic()-begin;result['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;result['peak_cuda_bytes']=torch.cuda.max_memory_allocated();result['peak_reserved_cuda_bytes']=torch.cuda.max_memory_reserved();result['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(root/name,result)


if __name__=='__main__':main()
