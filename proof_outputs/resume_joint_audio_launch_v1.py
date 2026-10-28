"""Resume only the fully verified prelaunch stage, after SSH read timeout."""
import hashlib,json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
HERE=Path(__file__).resolve().parent/'joint_audio_native_prepare_v1'
REMOTE='/home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((HERE/'launch_plan.json').read_text())
program='''from pathlib import Path
import hashlib,json,subprocess
p=Path(REMOTE);expected_plan=PLAN_SHA;expected_inputs=INPUTS_SHA;expected_source=SOURCE_SHA;expected_chain=CHAIN_SHA
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (p/'started').exists()
assert sha(p/'launch_plan.json')==expected_plan and sha(p/'inputs_manifest.json')==expected_inputs
assert sha(p/'source.tar.gz')==expected_source and sha(p/'run_chain.sh')==expected_chain
plan=json.loads((p/'launch_plan.json').read_text());verified=json.loads((p/'inputs_verified.json').read_text())
assert verified=={'complete':True,'files':265,'training_started':False}
for item in plan['files']:assert sha(p/'source'/item['path'])==item['sha256']
assert subprocess.run(['bash','-n',str(p/'run_chain.sh')]).returncode==0
(p/'started').touch()
log=open(p/'logs/chain.log','wb')
proc=subprocess.Popen(['timeout','18000','bash',str(p/'run_chain.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(p/'launch_resume.json').write_text(json.dumps({'pid':proc.pid,'verified_prelaunch_resumed':True,'training_started':False})+'\\n')
print(json.dumps({'pid':proc.pid,'verified_prelaunch_resumed':True,'training_started':False}))
'''.replace('REMOTE',repr(REMOTE)).replace('PLAN_SHA',repr(sha(HERE/'launch_plan.json'))).replace('INPUTS_SHA',repr(sha(HERE/'inputs_manifest.json'))).replace('SOURCE_SHA',repr(sha(HERE/'source.tar.gz'))).replace('CHAIN_SHA',repr(plan['chain_sha256']))
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('server-manager connect failed')
try:
 rc,out,err=c.execute('python3 -c '+shlex.quote(program),timeout=10)
 result={'remote':REMOTE,'launch_exit':rc,'stdout':out,'stderr':err,'launch_plan_sha256':sha(HERE/'launch_plan.json'),
         'source_archive_sha256':sha(HERE/'source.tar.gz'),'prelaunch_resume':True,'training_started':False,'gpu_visible':False}
 (HERE/'launch.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
 if rc:raise RuntimeError('verified prelaunch continuation failed')
finally:c.close()

