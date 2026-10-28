import base64, hashlib, json, shlex, sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig, SSHClient
root=Path(r'C:/Users/cereal/Documents/ai training/.agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-memory-corpus-oct13-9c19be622841/proof_outputs/k3_b2_l2_source_audit_v1')
root.mkdir(parents=True,exist_ok=True)
receipt=json.loads(Path(r'C:/Users/cereal/Documents/ai training/reviews/2026-10-13-folding/recovery-preparation/original_teacher_lane_followup_20261013T232456Z.json').read_text())['data']['files']['distill/parity/k3cache-2026-10-13-bisect-gpu6/probe/probe_receipt.json']['json']
e=receipt['engine'];paths={e['path']:e['sha256']}
for n,h in e['lane_sha256'].items():paths[e['lane_dir']+'/'+n]=h
paths['/home/cereal/omni-campaign/k3-run-oct09-claude/receipts/k3_dense_pass_receipt.json']=receipt['integrity']['fold_receipt']['receipt_sha256']
paths['/home/cereal/omni-campaign/distill/queues/frozen/k3_teacher_cache-gpu6.py']=receipt['tool_sha256']
probe="""import hashlib, json
from pathlib import Path
paths="""+repr(paths)+"""
out=[]
for path,pin in paths.items():
 p=Path(path);s=p.stat()
 if s.st_size>(16000000 if path.endswith('k3_dense_pass_receipt.json') else 2000000):raise RuntimeError('bounded source read refused: '+path)
 b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
 if h!=pin:raise RuntimeError('source pin mismatch: '+path)
 out.append({'path':path,'sha256':h,'bytes':len(b)})
print(json.dumps(out))
"""
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connection failed')
try:
 rc,out,err=c.execute('/usr/bin/python3 -B -c '+shlex.quote(probe),timeout=15)
 if rc!=0:raise RuntimeError(str((rc,out,err)))
 manifest=json.loads(out)
 for item in manifest:
  p=root/Path(item['path']).name
  if not p.exists():
   if not c.download_file(item['path'],str(root)):raise RuntimeError('download failed')
  if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('download/source hash mismatch')
  item['local']=str(p)
finally:c.close()
(root/'source_capture.json').write_text(json.dumps({'schema':'source_pinned_k3_l2_audit.v1','files':manifest},indent=2)+'\n')
print(json.dumps({'capture':str(root/'source_capture.json'),'files':len(manifest),'total_bytes':sum(x['bytes'] for x in manifest)}))
