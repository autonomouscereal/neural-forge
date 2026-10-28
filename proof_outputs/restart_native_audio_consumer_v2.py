"""Preserve genuine encoder evidence; fresh audited consumer/output capsule."""
import hashlib,json,shlex,subprocess,sys,tarfile
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
OWN=Path(__file__).resolve().parents[1]
ROOT=OWN.parent/'chimera-training-recovery-oct13-eae1b72e2c3f'
HERE=OWN/'proof_outputs/native_audio_consumer_v2';HERE.mkdir()
REMOTE='/home/cereal/omni-campaign/distill/codex-native-audio-consumer-modality-v2'
SPEECH='/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1'
JOINT='/home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1'
PROC='/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013'
IMAGE='sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164'
AMEND='312706e85efa8fbb996b67f220856adc76236bc516af1ba30f03a365d5b64055'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={str(p.relative_to(ROOT)).replace('\\','/'):p for p in (ROOT/'experiments/chimera_recovery').glob('*.py')}
for name in ('h3_audio_chart.py','audio_replay_index.py'):
 files['experiments/chimera_recovery/'+name]=OWN/'experiments/chimera_recovery'/name
hashes={name:sha(p) for name,p in files.items()}
with tarfile.open(HERE/'source.tar','w') as archive:
 for name,p in sorted(files.items()):archive.add(p,arcname=name)
assert hashes=={name:sha(p) for name,p in files.items()},'source changed while snapshotting'
plan={'schema':'chimera_recovery.native_audio_consumer_repair.v1','remote':REMOTE,'source_hashes':hashes,
 'root_baseline':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'consumer_fix_commit':'2f0219bf','source_tar_sha256':sha(HERE/'source.tar'),'image':IMAGE,
 'encoder_evidence_reused_unchanged':True,'failure':'binary floating endpoint equality disagreed with exact integer PCM coverage on53/201 genuine receipts',
 'old_speech':SPEECH,'old_joint':JOINT,'amendment_sha256':AMEND,'cpu_threads':8,'memory_gb':32,
 'one_serial_consumer_at_a_time':True,'training_started':False,'gpu_exposed':False}
(HERE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
mounts=['-v',REMOTE+':/recovery:rw','-v',REMOTE+'/source:/recovery/source:ro',
 '-v',SPEECH+':/work:ro','-v',JOINT+':/joint:ro',
 '-v',PROC+'/clean-audio-cache-v1:/work/clean-audio-cache-v1:ro',
 '-v',PROC+'/audio-cache-inputs-v1:/work/audio-cache-inputs-v1:ro',
 '-v','/home/cereal/model-surgery-lab/venv:/opt/teacher-runtime:ro',
 '-v','/home/cereal/models/Qwen3.8-27B-Uncensored:/model:ro']
base=['docker','run','--network','none','--cpus','8','--memory','32g','--no-healthcheck','--entrypoint','/usr/bin/timeout',
 '-e','CUDA_VISIBLE_DEVICES=','-e','TRANSFORMERS_OFFLINE=1','-e','HF_HUB_OFFLINE=1',
 '-e','OMP_NUM_THREADS=8','-e','MKL_NUM_THREADS=8','-e','PYTHONDONTWRITEBYTECODE=1',
 '-e','PYTHONPATH=/opt/teacher-runtime/lib/python3.12/site-packages:/recovery/source',*mounts]
commands=[]
def phase(name,module,argv,timeout=3600):
 cmd=[*base,'--name','codex-audio-consumer-'+name+'-modality-v2',IMAGE,str(timeout),'/usr/bin/python3.12',
      '/recovery/source/experiments/chimera_recovery/'+module,*argv]
 commands.append(' '.join(shlex.quote(x) if not x.startswith('$') else '"'+x+'"' for x in cmd)+f' > logs/{name}.log 2>&1\nprintf \'0\\n\' > logs/{name}.exitcode')
source201='/work/inputs/admission/files/a4db8d110e2590ed8b7f103b4fd4afee2cdf50f79e2862a80bfa01d8ebe088d0__source_ready.jsonl'
source225='/joint/inputs/admission/files/096270c869454e62b70b65cc97b30cf1ddc76449da7f19a513dec4021b0dd1bc__source_ready.jsonl'
common=['--source',source201,'--clean-manifest','/work/clean-speech-cache-v1/manifest.json','--clean-manifest-sha256','$speech_clean_sha',
 '--admission','/work/inputs/admission/manifest.json','--admission-sha256','9578be63cbc8e4c87b5d7cf6999826785d0a3173412ab8ffd7eb1fe533f48287',
 '--tokenizer-dir','/model','--charted-attention','/work/inputs/charted_attention.py','--amendment-sha256',AMEND]
phase('speech-ce','audio_cache.py',['prepare',*common,'--out','/recovery/native-speech-ce-v2'])
phase('speech-flow','speech_flow.py',[*common,'--out','/recovery/native-speech-flow-v2'])
phase('compose','audio_sources.py',['compose-cache','--source',source225,'--admission','/joint/inputs/admission/manifest.json',
 '--admission-sha256','a07c0139fd073647d8482ef4af4b94c9d677f9a195534e1350adc45aa399ae8c',
 '--clean-manifests','/recovery/clean_pins.json','--out','/recovery/joint-clean-cache-v2'])
commands.append('joint_clean_sha=$(sha256sum joint-clean-cache-v2/manifest.json | cut -d\' \' -f1)')
phase('joint-ce','audio_cache.py',['prepare','--source',source225,'--clean-manifest','/recovery/joint-clean-cache-v2/manifest.json',
 '--clean-manifest-sha256','$joint_clean_sha','--admission','/joint/inputs/admission/manifest.json',
 '--admission-sha256','a07c0139fd073647d8482ef4af4b94c9d677f9a195534e1350adc45aa399ae8c',
 '--tokenizer-dir','/model','--charted-attention','/joint/inputs/charted_attention.py','--amendment-sha256',AMEND,
 '--out','/recovery/native-audio-ce-v2'])
commands.append('ce_sha=$(sha256sum native-audio-ce-v2/manifest.json | cut -d\' \' -f1)\nflow_sha=$(sha256sum native-speech-flow-v2/manifest.json | cut -d\' \' -f1)')
phase('replay-audit','audio_replay_index.py',['--ce-manifest','/recovery/native-audio-ce-v2/manifest.json','--ce-sha256','$ce_sha',
 '--ce-source-dir','/recovery/source/experiments/chimera_recovery','--flow-manifest','/recovery/native-speech-flow-v2/manifest.json',
 '--flow-sha256','$flow_sha','--flow-source-dir','/recovery/source/experiments/chimera_recovery',
 '--amendment-sha256',AMEND,'--out','/recovery/replay-index.json'],7200)
script='set -euo pipefail\ncd '+shlex.quote(REMOTE)+"\ntrap 'rc=$?; printf \"%s\\n\" \"$rc\" > logs/chain.exitcode' EXIT\nspeech_clean_sha=$(sha256sum "+shlex.quote(SPEECH+'/clean-speech-cache-v1/manifest.json')+" | cut -d' ' -f1)\n"+'\n'.join(commands)+'\n'
(HERE/'run_chain.sh').write_bytes(script.encode())
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 program="from pathlib import Path;p=Path("+repr(REMOTE)+");assert not p.exists();p.mkdir();(p/'source').mkdir();(p/'logs').mkdir()"
 rc,out,err=c.execute('python3 -c '+shlex.quote(program),timeout=10)
 if rc:raise RuntimeError(err)
 for name in ('source.tar','plan.json','run_chain.sh'):
  if not c.upload_file(str(HERE/name),REMOTE+'/'+name):raise RuntimeError('upload failed')
 code='''from pathlib import Path
import hashlib,json,subprocess,sys,os,signal
p=Path(REMOTE);plan=json.loads((p/'plan.json').read_text());s=Path(SPEECH)
assert hashlib.sha256((p/'source.tar').read_bytes()).hexdigest()==plan['source_tar_sha256']
subprocess.run(['tar','xf',str(p/'source.tar'),'-C',str(p/'source')],check=True)
for name,value in plan['source_hashes'].items():assert hashlib.sha256((p/'source'/name).read_bytes()).hexdigest()==value
subprocess.run(['bash','-n',str(p/'run_chain.sh')],check=True)
sys.path.insert(0,str(p/'source'))
from experiments.chimera_recovery.h3_audio_chart import validate_encoder_proof
proofs=list((s/'clean-speech-cache-v1').glob('[0-9]*.json'));assert len(proofs)==201
for f in proofs:validate_encoder_proof(json.loads(f.read_text()))
state=json.loads(subprocess.check_output(['docker','inspect','codex-speech-native-ce-modality-v1','--format','{{json .State}}']))
assert state['Running'] is False and state['ExitCode']!=0
prior=[]
for pid,namespace in [(2018524,'codex-joint-audio-preparation-modality-01a1326f-v1'),(2057526,'codex-native-replay-audit-modality-v1')]:
 f=Path('/proc')/str(pid)/'cmdline'
 if f.exists():
  cmd=f.read_bytes().replace(b'\\0',b' ').decode();assert namespace in cmd and 'run_chain.sh' in cmd;assert os.getpgid(pid)==pid
  prior.append({'pid':pid,'cmdline':cmd,'terminated_owned_waiter':True});os.killpg(pid,signal.SIGTERM)
receipt={'all201_exact_pcm_proofs_validated':True,'encoder_manifest_sha256':hashlib.sha256((s/'clean-speech-cache-v1/manifest.json').read_bytes()).hexdigest(),
 'failed_original_consumer_state':state,'old_owned_waiters':prior,'proof_bytes_unchanged':True,'training_started':False}
(p/'repair-validation.json').write_text(json.dumps(receipt)+'\\n')
pins=[{'path':'/work/clean-audio-cache-v1/manifest.json','sha256':'ae08c35f02458b9e36a19431e2d9ff9ffedaf29d4d3ed9118b2d00a9b0572b47'},
 {'path':'/work/clean-speech-cache-v1/manifest.json','sha256':receipt['encoder_manifest_sha256']}]
(p/'clean_pins.json').write_text(json.dumps(pins)+'\\n')
assert not(p/'started').exists();(p/'started').touch();log=open(p/'logs/chain.log','wb')
proc=subprocess.Popen(['timeout','23000','bash',str(p/'run_chain.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':proc.pid,'validation':receipt,'training_started':False,'gpu_exposed':False}))
'''.replace('REMOTE',repr(REMOTE)).replace('SPEECH',repr(SPEECH))
 rc,out,err=c.execute('python3 -c '+shlex.quote(code),timeout=15)
 result={'remote':REMOTE,'exit_code':rc,'stdout':out,'stderr':err,'plan_sha256':sha(HERE/'plan.json')}
 (HERE/'launch.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 if rc:raise RuntimeError('repair validation/launch failed')
finally:c.close()
