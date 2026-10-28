import datetime
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, 'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import SSHClient, ServerConfig

OUT = Path(__file__).resolve().parent
PROGRAM = r'''
import datetime, hashlib, json, os, subprocess, time
from pathlib import Path
def command(argv, seconds):
    start=time.monotonic()
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=seconds)
        return {'argv':argv,'exit_code':p.returncode,'seconds':time.monotonic()-start,'stdout':p.stdout[-30000:],'stderr':p.stderr[-3000:]}
    except subprocess.TimeoutExpired as e:
        return {'argv':argv,'timed_out':True,'seconds':time.monotonic()-start}
def file_meta(p, contents=False):
    p=Path(p)
    if not p.is_file():return {'path':str(p),'exists':False}
    b=p.read_bytes()
    r={'path':str(p),'exists':True,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'mtime_ns':p.stat().st_mtime_ns}
    if contents:r['tail']=b.decode('utf-8',errors='replace').splitlines()[-30:]
    return r
out={'schema':'chimera_recovery.music_capture_readonly_status.v1','captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_or_resource_mutations':False}
paths=['/home/cereal/omni-campaign/distill/tools/_superseded/media_b_teacher_cache_1.0.8.py','/home/cereal/omni-campaign/distill/staging_p40/media_b_teacher_cache.py','/home/cereal/omni-campaign/distill/tools/media_b_teacher_cache.py']
out['writer_candidates']=[file_meta(p) for p in paths]
out['gpus']=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits','-i','1,2,3'],4)
out['containers']=command(['docker','ps','--format','{{json .}}'],4)
names=[]
for line in out['containers'].get('stdout','').splitlines():
    try:r=json.loads(line)
    except ValueError:continue
    if 'ace-lm' in r.get('Names',''):names.append(r['Names'])
if names:
    fmt='{{json .Id}} {{json .Name}} {{json .State}} {{json .Config.Cmd}} {{json .HostConfig.DeviceRequests}} {{json .Mounts}}'
    out['ace_worker_metadata']=command(['docker','inspect','--format',fmt,*names],4)
out['queue_status']=file_meta('/home/cereal/omni-campaign/distill/queues/p40.status',True)
st=Path('/home/cereal/omni-campaign/distill/queues/p40.state')
out['rank_exits']={p.name:p.read_text().strip() for p in sorted(st.glob('lm-*.rc'))}
out['rank_claim_dirs']=[p.name for p in sorted(st.glob('lm-*')) if p.is_dir()]
base=Path('/mnt/lexar/omni-campaign/teacher_cache/ace/ace_lm_codes')
out['caches']=[]
for d in sorted(base.glob('*/*')):
    if not d.is_dir():continue
    entries=list(d.iterdir())
    shard_names=sorted(p.name for p in entries if p.name.startswith('shard-') and p.suffix=='.safetensors')
    sidecars=sorted(p.name for p in entries if p.name.startswith('shard-') and p.suffix=='.json')
    r={'path':str(d),'shards_present':len(shard_names),'sidecars_present':len(sidecars),'first_shard':shard_names[0] if shard_names else None,'last_shard':shard_names[-1] if shard_names else None,'manifest':file_meta(d/'manifest.json'),'manifest_sidecar':file_meta(d/'manifest.json.sha256',True),'cache_config':file_meta(d/'cache_config.json')}
    if (d/'manifest.json').is_file():
        m=json.loads((d/'manifest.json').read_text())
        r['sealed_metadata']={k:m.get(k) for k in ('schema','teacher','tool','sealed_at_utc','run_config','config_sha256','shard_tool_versions','compat','checks','corpus','totals','finite','capability_claim')}
        r['teacher_integrity_summary']={k:m.get('teacher_integrity',{}).get(k) for k in ('mode','equal')}
        expected=m.get('shards',[])
        r['manifest_all_shards_present']=all((d/s['name']).is_file() for s in expected)
        r['manifest_all_shard_sizes_match']=all((d/s['name']).is_file() and (d/s['name']).stat().st_size==s['bytes'] for s in expected)
        r['manifest_all_sidecars_present']=all((d/Path(s['name']).with_suffix('.json')).is_file() for s in expected)
        r['full_raw_shard_sha_finite_verification_performed']=False
    for name in ('cache_config.json',):
        if (d/name).is_file():r[name]=json.loads((d/name).read_text())
    if sidecars:
        for label,name in (('first',sidecars[0]),('last',sidecars[-1])):
            p=d/name
            r[label+'_sidecar']={'file':file_meta(p),'metadata':json.loads(p.read_text())}
        r['newest_shard_mtime_utc']=datetime.datetime.fromtimestamp(max((d/n).stat().st_mtime for n in shard_names),datetime.timezone.utc).isoformat()
    out['caches'].append(r)
logs=Path('/home/cereal/omni-campaign/distill/media_b/logs_p40')
out['recent_logs']=[file_meta(p,True) for p in sorted(logs.glob('*ace-lm*.log'),key=lambda p:p.stat().st_mtime,reverse=True)[:4]]
rec=Path('/home/cereal/omni-campaign/distill/media_b/receipts_p40')
out['check_receipts']=[]
for p in sorted(rec.glob('*ace-lm*')):
    if p.is_file() and p.stat().st_size<1000000:
        r=file_meta(p)
        try:r['body']=json.loads(p.read_text())
        except ValueError:pass
        out['check_receipts'].append(r)
print(json.dumps(out,ensure_ascii=False))
'''

if __name__ == '__main__':
    c = SSHClient.from_config(ServerConfig().get('ai'), control=True)
    try:
        if not c.connect():raise RuntimeError('server-manager connection failed')
        rc, stdout, stderr = c.execute('python3 -c ' + shlex.quote(PROGRAM), timeout=23)
    finally:
        c.close()
    (OUT/'remote_status.stdout.json').write_text(stdout,encoding='utf-8')
    (OUT/'remote_status.stderr.txt').write_text(stderr,encoding='utf-8')
    proof={'exit_code':rc,'stdout_sha256':hashlib.sha256(stdout.encode()).hexdigest(),'captured_at_local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (OUT/'remote_command_receipt.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    if rc:print(json.dumps(proof));print(stderr[-2000:]);sys.exit(rc)
    d=json.loads(stdout)
    print(json.dumps({'captured_at_utc':d['captured_at_utc'],'writers':d['writer_candidates'],'gpus':d['gpus'],'rank_exits':d['rank_exits'],'caches':[{'path':r['path'],'shards':r['shards_present'],'manifest':r['manifest'],'sealed_metadata':r.get('sealed_metadata'),'teacher_integrity':r.get('teacher_integrity_summary')} for r in d['caches']]},indent=2))
