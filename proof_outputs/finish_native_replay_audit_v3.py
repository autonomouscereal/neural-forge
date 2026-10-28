"""Read completed private root-owned cache seals and run only the final audit."""
import json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
REMOTE='/home/cereal/omni-campaign/distill/codex-native-audio-consumer-modality-v2'
HERE=Path(__file__).resolve().parent/'native_audio_consumer_v2'
program='''from pathlib import Path
import hashlib,json,subprocess
p=Path(REMOTE);log=p/'logs-audit-repair-v3';assert not log.exists();log.mkdir()
assert (p/'logs-mountfix-v2/chain.exitcode').read_text().strip()=='1'
assert 'Permission denied' in (p/'logs-mountfix-v2/chain.log').read_text()
assert all((p/'logs-mountfix-v2'/n).read_text().strip()=='0' for n in ('speech-ce.exitcode','speech-flow.exitcode','compose.exitcode','joint-ce.exitcode'))
pins={}
for name in ('native-speech-ce-v2','native-speech-flow-v2','joint-clean-cache-v2','native-audio-ce-v2'):
 f=p/name/'manifest.json';m=json.loads(f.read_text());assert m['sealed'] is True
 pins[name]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'entries':len(m.get('batches',m.get('entries',[])))}
assert pins['native-speech-flow-v2']['entries']==744 and pins['native-audio-ce-v2']['entries']==194
line=next(l for l in (p/'run_chain_mountfix_v2.sh').read_text().splitlines() if l.startswith('docker run') and 'audio_replay_index.py' in l)
line=line.replace('codex-audio-consumer-replay-audit-modality-mountfix-v2','codex-audio-consumer-replay-audit-modality-auditfix-v3')
line=line.replace('logs-mountfix-v2/replay-audit.log','logs-audit-repair-v3/replay-audit.log')
script='set -euo pipefail\\ncd '+str(p)+"\\ntrap 'rc=$?; printf \\\"%s\\\\n\\\" \\\"$rc\\\" > logs-audit-repair-v3/exitcode' EXIT\\n"
script+='ce_sha='+pins['native-audio-ce-v2']['sha256']+'\\nflow_sha='+pins['native-speech-flow-v2']['sha256']+'\\n'+line+'\\n'
(p/'run_audit_repair_v3.sh').write_text(script)
subprocess.run(['bash','-n',str(p/'run_audit_repair_v3.sh')],check=True)
receipt={'original_phase_exitcodes_preserved':True,'fix_scope':'read actual sealed root-owned cache hashes without rewriting cache bytes',
 'input_manifests':pins,'script_sha256':hashlib.sha256(script.encode()).hexdigest(),'training_started':False,'gpu_exposed':False}
(log/'input_receipt.json').write_text(json.dumps(receipt)+'\\n')
output=open(log/'chain.log','wb');job=subprocess.Popen(['timeout','7300','bash',str(p/'run_audit_repair_v3.sh')],stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':job.pid,'receipt':receipt}))
'''.replace('REMOTE',repr(REMOTE))
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 rc,out,err=c.execute('sudo -n python3 -c '+shlex.quote(program),timeout=10)
 (HERE/'audit-repair-v3-launch.json').write_text(json.dumps({'exit_code':rc,'stdout':out,'stderr':err},indent=2)+'\n')
 print(out)
 if rc:raise RuntimeError(err)
finally:c.close()
