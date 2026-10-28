"""Capture public official speech metadata only; never download model weights."""
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import sys
from pathlib import Path
import urllib.request

MODELS=['Qwen/Qwen3-ASR-0.6B','Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice']
def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'chimera-recovery-source-audit/2026','Accept':'application/json'})
    with urllib.request.urlopen(req,timeout=12) as stream:
        body=stream.read(2000001)
        if len(body)>2000000:
            raise ValueError('metadata_bound')
    return json.loads(body),hashlib.sha256(body).hexdigest()
def model(model_id):
    url='https://huggingface.co/api/models/'+model_id+'?blobs=true'
    doc,api_sha=fetch(url)
    revision=doc['sha']
    files=[{'path':item['rfilename'],'size':item.get('size'),'blob_id':item.get('blobId'),'lfs':item.get('lfs')} for item in doc['siblings']]
    configs={}
    for name in ['config.json','preprocessor_config.json','generation_config.json','speech_tokenizer/config.json','speech_tokenizer/preprocessor_config.json']:
        if any(item['path']==name for item in files):
            config,config_sha=fetch('https://huggingface.co/'+model_id+'/resolve/'+revision+'/'+name)
            configs[name]={'sha256':config_sha,'value':config}
    return {'model_id':model_id,'revision':revision,'api_url':url,'api_sha256':api_sha,'last_modified':doc.get('lastModified'),
            'created_at':doc.get('createdAt'),'gated':doc.get('gated'),'private':doc.get('private'),
            'safetensors':doc.get('safetensors'),'files':files,'configs':configs}
def repo(name):
    doc,api_sha=fetch('https://api.github.com/repos/'+name+'/commits/main')
    return {'repo':name,'commit':doc['sha'],'api_sha256':api_sha,'commit_date':doc['commit']['committer']['date']}
with ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(model,name) for name in MODELS]+[pool.submit(repo,name) for name in ['QwenLM/Qwen3-ASR','QwenLM/Qwen3-TTS']]
    rows=[]
    for future in futures:
        try:
            rows.append(future.result(timeout=55))
        except Exception as error:
            rows.append({'error_type':type(error).__name__})
result={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'public_metadata_only':True,'rows':rows}
path=Path(sys.argv[1] if len(sys.argv)>1 else 'proof_outputs/speech_official_model_metadata_oct13.json')
blob=(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode()
with path.open('xb') as stream:
    stream.write(blob)
print(json.dumps({'path':str(path.resolve()),'sha256':hashlib.sha256(blob).hexdigest(),'successful':sum('error_type' not in r for r in rows),'rows':len(rows)}))
