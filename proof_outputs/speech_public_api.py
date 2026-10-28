"""Pin official speech package metadata and bounded inference source; no weights/install."""
import datetime
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request

def raw(url):
    request=urllib.request.Request(url,headers={'User-Agent':'chimera-recovery-source-audit/2026'})
    with urllib.request.urlopen(request,timeout=12) as stream:
        body=stream.read(1500001)
    if len(body)>1500000:
        raise ValueError('source_bound')
    return body
def package(name):
    body=raw('https://pypi.org/pypi/'+name+'/json')
    doc=json.loads(body)
    return {'package':name,'metadata_sha256':hashlib.sha256(body).hexdigest(),'version':doc['info']['version'],
            'requires_python':doc['info']['requires_python'],'requires_dist':doc['info']['requires_dist'],
            'files':[{'filename':item['filename'],'size':item['size'],'sha256':item['digests']['sha256'],'url':item['url'],'uploaded':item['upload_time_iso_8601']} for item in doc['urls']]}
def source(repo,commit,paths):
    tree=json.loads(raw('https://api.github.com/repos/'+repo+'/git/trees/'+commit+'?recursive=1'))
    existing={item['path'] for item in tree['tree']}
    rows=[]
    for path in paths:
        if path in existing:
            url='https://raw.githubusercontent.com/'+repo+'/'+commit+'/'+path
            body=raw(url)
            rows.append({'path':path,'url':url,'sha256':hashlib.sha256(body).hexdigest(),'text':body.decode('utf-8')})
    return {'repo':repo,'commit':commit,'files':rows,'candidate_paths':[path for path in existing if any(v in path for v in ['inference','utils.py','setup.py','pyproject.toml'])]}
tasks=[(package,('qwen-asr',)),(package,('qwen-tts',)),
       (source,('QwenLM/Qwen3-ASR','7c6daf77a2421100f5fb066495372c00129d39ff',['pyproject.toml','setup.py','qwen_asr/__init__.py','qwen_asr/inference/qwen3_asr.py','qwen_asr/inference/utils.py'])),
       (source,('QwenLM/Qwen3-TTS','022e286b98fbec7e1e916cb940cdf532cd9f488e',['pyproject.toml','setup.py','qwen_tts/__init__.py','qwen_tts/inference/qwen3_tts_model.py']))]
with ThreadPoolExecutor(max_workers=4) as pool:
    jobs=[pool.submit(func,*args) for func,args in tasks]
    rows=[]
    for job in jobs:
        try:
            rows.append(job.result(timeout=55))
        except Exception as error:
            rows.append({'error_type':type(error).__name__})
doc={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'public_source_only':True,'rows':rows}
blob=(json.dumps(doc,ensure_ascii=False,indent=2)+'\n').encode()
path=Path('proof_outputs/speech_official_api_oct13.json')
with path.open('xb') as stream:
    stream.write(blob)
print(json.dumps({'path':str(path.resolve()),'sha256':hashlib.sha256(blob).hexdigest(),'successful':sum('error_type' not in r for r in rows)}))
