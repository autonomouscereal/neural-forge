"""Fresh private CPU audit queued after owned native preparation completes."""
import hashlib,json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
LOCAL=Path(__file__).resolve().parents[1]
HERE=LOCAL/'proof_outputs/native_replay_audit_v1'
HERE.mkdir()
REMOTE='/home/cereal/omni-campaign/distill/codex-native-replay-audit-modality-v1'
SPEECH='/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1'
JOINT='/home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1'
PROC='/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013'
IMAGE='sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164'
AMENDMENT='312706e85efa8fbb996b67f220856adc76236bc516af1ba30f03a365d5b64055'
adapter=LOCAL/'experiments/chimera_recovery/audio_replay_index.py'
plan={'schema':'chimera_recovery.native_replay_actual_audit_job.v1','remote':REMOTE,
 'source':{'path':str(adapter),'sha256':hashlib.sha256(adapter.read_bytes()).hexdigest()},
 'ce_source_dir':'/joint/source/experiments/chimera_recovery','flow_source_dir':'/work/source/experiments/chimera_recovery',
 'ce_manifest':'/joint/native-audio-ce-v1/manifest.json','flow_manifest':'/work/native-speech-flow-v1/manifest.json',
 'image':IMAGE,'amendment_sha256':AMENDMENT,'cpu_threads':8,'memory_gb':32,
 'audit_timeout_seconds':7200,'wait_for_owned_joint_chain':True,'source_mounts_readonly':True,
 'training_started':False,'gpu_exposed':False,'expected_counts_are_not_proof':{'audio_ce':194,'speech_flow':744,'encoder_sources':225}}
(HERE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
script=f'''set -euo pipefail
cd {shlex.quote(REMOTE)}
trap 'rc=$?; printf "%s\\n" "$rc" > logs/chain.exitcode' EXIT
deadline=$((SECONDS+10800))
while [ ! -f {shlex.quote(JOINT+'/logs/chain.exitcode')} ]; do
 if [ "$SECONDS" -ge "$deadline" ]; then printf 'joint native seal deadline exceeded\\n' >&2; exit 2; fi
 sleep 15
done
test "$(cat {shlex.quote(JOINT+'/logs/chain.exitcode')})" = 0
python3 - <<'PY'
from pathlib import Path
import hashlib,json
p=Path({REMOTE!r});ce=Path({(JOINT+'/native-audio-ce-v1/manifest.json')!r});flow=Path({(SPEECH+'/native-speech-flow-v1/manifest.json')!r})
def pin(path):return {{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
(p/'manifest_pins.json').write_text(json.dumps({{'ce':pin(ce),'flow':pin(flow)}})+'\\n')
PY
ce_sha=$(sha256sum {shlex.quote(JOINT+'/native-audio-ce-v1/manifest.json')} | cut -d' ' -f1)
flow_sha=$(sha256sum {shlex.quote(SPEECH+'/native-speech-flow-v1/manifest.json')} | cut -d' ' -f1)
docker run --name codex-native-replay-audit-modality-v1 --network none --cpus 8 --memory 32g --no-healthcheck \\
 --entrypoint /usr/bin/timeout -e CUDA_VISIBLE_DEVICES= -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \\
 -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e PYTHONDONTWRITEBYTECODE=1 \\
 -e PYTHONPATH=/opt/teacher-runtime/lib/python3.12/site-packages \\
 -v /home/cereal/model-surgery-lab/venv:/opt/teacher-runtime:ro \\
 -v /home/cereal/models/Qwen3.8-27B-Uncensored:/model:ro \\
 -v {shlex.quote(REMOTE+':/audit:rw')} -v {shlex.quote(REMOTE+'/source:/audit/source:ro')} \\
 -v {shlex.quote(JOINT+':/joint:ro')} -v {shlex.quote(SPEECH+':/work:ro')} \\
 -v {shlex.quote(PROC+'/clean-audio-cache-v1:/work/clean-audio-cache-v1:ro')} \\
 -v {shlex.quote(PROC+'/audio-cache-inputs-v1:/work/audio-cache-inputs-v1:ro')} \\
 {IMAGE} 7200 /usr/bin/python3.12 /audit/source/audio_replay_index.py \\
 --ce-manifest /joint/native-audio-ce-v1/manifest.json --ce-sha256 "$ce_sha" \\
 --ce-source-dir /joint/source/experiments/chimera_recovery \\
 --flow-manifest /work/native-speech-flow-v1/manifest.json --flow-sha256 "$flow_sha" \\
 --flow-source-dir /work/source/experiments/chimera_recovery \\
 --amendment-sha256 {AMENDMENT} --out /audit/replay-index.json > logs/audit.log 2>&1
printf '0\\n' > logs/audit.exitcode
'''
(HERE/'run_chain.sh').write_bytes(script.encode())
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('server-manager connect failed')
try:
 code='from pathlib import Path;p=Path('+repr(REMOTE)+');assert not p.exists();p.mkdir();(p/"source").mkdir();(p/"logs").mkdir()'
 rc,out,err=c.execute('python3 -c '+shlex.quote(code),timeout=10)
 if rc:raise RuntimeError('fresh audit namespace failed: '+err)
 for path,target in ((adapter,REMOTE+'/source/audio_replay_index.py'),(HERE/'plan.json',REMOTE+'/plan.json'),(HERE/'run_chain.sh',REMOTE+'/run_chain.sh')):
  if not c.upload_file(str(path),target):raise RuntimeError('audit source upload failed')
 verify='from pathlib import Path;import hashlib,subprocess;p=Path('+repr(REMOTE)+');assert hashlib.sha256((p/"source/audio_replay_index.py").read_bytes()).hexdigest()=='+repr(plan['source']['sha256'])+';subprocess.run(["bash","-n",str(p/"run_chain.sh")],check=True)'
 rc,out,err=c.execute('python3 -c '+shlex.quote(verify),timeout=10)
 if rc:raise RuntimeError('actual audit source or launcher verification failed: '+err)
 launch='from pathlib import Path;import subprocess,json;p=Path('+repr(REMOTE)+');assert not(p/"started").exists();(p/"started").touch();log=open(p/"logs/chain.log","wb");proc=subprocess.Popen(["timeout","19000","bash",str(p/"run_chain.sh")],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({"pid":proc.pid,"training_started":False}))'
 rc,out,err=c.execute('python3 -c '+shlex.quote(launch),timeout=10)
 result={'remote':REMOTE,'launch_exit':rc,'stdout':out,'stderr':err,'plan_sha256':hashlib.sha256((HERE/'plan.json').read_bytes()).hexdigest(),'training_started':False,'gpu_exposed':False}
 (HERE/'launch.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
 if rc:raise RuntimeError('owned replay audit detached launch failed')
finally:c.close()
