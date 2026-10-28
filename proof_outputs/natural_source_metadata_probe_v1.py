"""Bounded metadata only: immutable public cards, memory leads, existing source profile."""
import datetime
import hashlib
import json
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'natural_recorded_sources_oct13_v1'
OUT.mkdir(exist_ok=True)
CAP = 2 * 1024 * 1024
records = []

def save(name, raw, metadata):
    p = OUT / name
    if p.exists() and p.read_bytes() != raw:
        raise RuntimeError('refuse replacing source metadata: ' + str(p))
    p.write_bytes(raw)
    metadata.update(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    records.append(metadata)
    return raw

def fetch(name, url):
    req = urllib.request.Request(url, headers={'User-Agent': 'ChimeraRecovery/2026-10-13 metadata audit'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read(CAP + 1)
        if len(raw) > CAP:
            raise RuntimeError('metadata exceeds bounded cap')
        return save(name, raw, {'kind': 'public_metadata', 'url': url, 'resolved_url': resp.url, 'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'etag': resp.headers.get('ETag')})

api = json.loads(fetch('omnilingual_api.json', 'https://huggingface.co/api/datasets/facebook/omnilingual-asr-corpus'))
revision = api['sha']
fetch('omnilingual_readme.md', f'https://huggingface.co/datasets/facebook/omnilingual-asr-corpus/resolve/{revision}/README.md')
fetch('common_voice_v27_v5.html', 'https://community.mozilladatacollective.com/commonvoice-27/')
fetch('common_voice_v5_timeline.json', 'https://discourse.mozilla.org/t/dataset-release-timeline-for-mcv-scripted-speech-v27-0-and-spontaneous-speech-v5-0/149314.json')
fetch('sssd_project.html', 'https://wavlab-speech.github.io/SSSD/')
fetch('sssd_org.json', 'https://huggingface.co/api/datasets?author=SSSD-CMU&limit=10')

for num, query in enumerate(('Omnilingual ASR Corpus Common Voice26 natural speech', 'Common Voice speech data human recordings2025 2026')):
    cmd = ['C:/Python314/python.exe', 'C:/Users/cereal/.claude/memory_backend/agent_memory.py', '--json', 'search', query, '--all-spaces', '--limit', '5']
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError('memory retrieval failed: ' + result.stderr.decode(errors='replace')[-600:])
    save(f'memory-{num}.json', result.stdout, {'kind': 'primary_postgresql_memory_search', 'query': query, 'argv': cmd, 'returned': 5})

sys.path.insert(0, 'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig, SSHClient
program = r'''from pathlib import Path
import json,hashlib
p=Path('/home/cereal/omnicoder_2026_work/profiles/dataset_curation_2026.json')
raw=p.read_bytes();x=json.loads(raw)
rows=[]
def walk(v,path):
 if isinstance(v,dict):
  descriptor=' '.join(str(v.get(k,'')) for k in ('id','name','dataset','dataset_id','title','url','hf_id','repo_id','dataset_key','source','source_url')).lower()
  if any(s in descriptor for s in ('speech','voice','audio','omnilingual','commonvoice','common-voice','openslr','sssd','vaani','waxal')):
   rows.append({'json_path':path,'metadata':v});return
  for k,w in v.items():walk(w,path+'/'+str(k))
 elif isinstance(v,list):
  for n,w in enumerate(v):walk(w,path+'/'+str(n))
walk(x,'')
print(json.dumps({'source_path':str(p),'source_sha256':hashlib.sha256(raw).hexdigest(),'source_bytes':len(raw),'rows':rows},ensure_ascii=False))
'''
c = SSHClient.from_config(ServerConfig().get('ai'), control=True)
if not c.connect():
    raise RuntimeError('profile read connection failed')
try:
    rc, out, err = c.execute('python3 -c ' + shlex.quote(program), timeout=10)
    if rc:
        raise RuntimeError('profile read failed: ' + err[-600:])
    save('existing_profile_speech.json', out.encode(), {'kind': 'existing_read_only_profile_metadata', 'server': 'ai', 'remote_source': '/home/cereal/omnicoder_2026_work/profiles/dataset_curation_2026.json'})
finally:
    c.close()

receipt = {'schema': 'chimera_recovery.natural_recorded_source_metadata.v1', 'dated': '2026-10-13', 'metadata_files': records, 'metadata_only': True, 'audio_downloaded': False, 'weights_downloaded': False, 'teacher_launched': False, 'train_admitted_rows': 0, 'per_request_bytes_limit': CAP}
(OUT / 'source-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
print(json.dumps({'receipt': str(OUT / 'source-receipt.json'), 'metadata_files': len(records), 'omnilingual_revision': revision, 'profile_rows': len(json.loads(out)['rows'])}))
