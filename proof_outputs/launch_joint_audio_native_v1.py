"""Fresh joint proof composition and CE, after actual speech201 seal."""
from pathlib import Path
import hashlib,json,shlex,shutil,subprocess,sys,tarfile
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
HERE=Path(__file__).resolve().parent/'joint_audio_native_prepare_v1'
BASE=Path('C:/Users/cereal/Documents/ai training')
ROOT=BASE/'.agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-training-recovery-oct13-eae1b72e2c3f'
REMOTE='/home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1'
SPEECH='/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1'
PROCEDURAL='/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013'
CPU='sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164'
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1<<24),b''):digest.update(chunk)
    return digest.hexdigest()
transfer=json.loads((HERE/'transfer.json').read_text())
if sha(HERE/'new_inputs.tar.gz')!=transfer['sha256']:raise ValueError('joint input transfer changed')
source=HERE/'source';source.mkdir(exist_ok=False)
commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
files=[]
for name in ('training.py','train_media.py','media_cache.py','stage_inheritance.py','native_state_inheritance.py',
             'train_audio.py','audio_cache.py','audio_paths.py','audio_dialogue.py','h3_audio_probe.py','h3_audio_chart.py','audio_sources.py'):
    original=ROOT/'experiments/chimera_recovery'/name
    target=source/'experiments/chimera_recovery'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    before=sha(original);shutil.copyfile(original,target)
    if sha(original)!=before or sha(target)!=before:raise ValueError('promoted joint source changed: '+name)
    files.append({'path':str(target.relative_to(source)).replace('\\','/'),'sha256':before})
if subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()!=commit:raise ValueError('root revision changed while sealing source')
bundle=HERE/'source.tar.gz'
with tarfile.open(bundle,'w:gz',compresslevel=1) as handle:
    for item in files:handle.add(source/item['path'],arcname=item['path'],recursive=False)
admission=json.loads((HERE/'inputs/admission/manifest.json').read_text())
source_path=admission['sources']['source']['path'];admission_sha=sha(HERE/'inputs/admission/manifest.json')
common=f'''--network none --cpus 8 --memory 32g --no-healthcheck --entrypoint /usr/bin/timeout \
 -e CUDA_VISIBLE_DEVICES= -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
 -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e PYTHONDONTWRITEBYTECODE=1 \
 -e PYTHONPATH=/opt/teacher-runtime/lib/python3.12/site-packages:/joint/source \
 -v /home/cereal/model-surgery-lab/venv:/opt/teacher-runtime:ro \
 -v /home/cereal/models/Qwen3.8-27B-Uncensored:/model:ro \
 -v "{REMOTE}:/joint:rw" -v "{REMOTE}/source:/joint/source:ro" -v "{REMOTE}/inputs:/joint/inputs:ro" \
 -v "{SPEECH}:/work:ro" \
 -v "{PROCEDURAL}/clean-audio-cache-v1:/work/clean-audio-cache-v1:ro" \
 -v "{PROCEDURAL}/audio-cache-inputs-v1:/work/audio-cache-inputs-v1:ro"'''
chain=f'''set -euo pipefail
cd "{REMOTE}"
trap 'rc=$?; printf "%s\\n" "$rc" > logs/chain.exitcode' EXIT
deadline=$((SECONDS+10800))
while [ ! -f "{SPEECH}/clean-speech-cache-v1/manifest.json" ]; do
 if [ "$SECONDS" -ge "$deadline" ]; then printf 'speech seal deadline exceeded\\n' >&2; exit 2; fi
 running=$(docker inspect --format '{{{{.State.Running}}}}' codex-speech-clean-modality-v1)
 if [ "$running" != true ]; then printf 'speech encoder stopped without a seal\\n' >&2; exit 3; fi
 sleep 10
done
# Serialize with the original native CE/speaking chain: one CPU8/32GB
# preparation phase at a time. A failed original chain grants no joint seal.
while [ ! -f "{SPEECH}/logs/chain.exitcode" ]; do
 if [ "$SECONDS" -ge "$deadline" ]; then printf 'speech native chain deadline exceeded\\n' >&2; exit 4; fi
 sleep 10
done
test "$(cat '{SPEECH}/logs/chain.exitcode')" = 0
python3 - <<'PY'
from pathlib import Path
import hashlib,json
p=Path('{REMOTE}');speech=Path('{SPEECH}/clean-speech-cache-v1/manifest.json')
proc=Path('{PROCEDURAL}/clean-audio-cache-v1/manifest.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(proc)=='ae08c35f02458b9e36a19431e2d9ff9ffedaf29d4d3ed9118b2d00a9b0572b47'
pins=[{{'path':'/work/clean-audio-cache-v1/manifest.json','sha256':sha(proc)}},
      {{'path':'/work/clean-speech-cache-v1/manifest.json','sha256':sha(speech)}}]
(p/'clean_pins.json').write_text(json.dumps(pins)+'\\n')
PY
docker run --name codex-joint-audio-compose-modality-v1 {common} \
 {CPU} 3600 /usr/bin/python3.12 /joint/source/experiments/chimera_recovery/audio_sources.py compose-cache \
 --source {shlex.quote(source_path)} --admission /joint/inputs/admission/manifest.json --admission-sha256 {admission_sha} \
 --clean-manifests /joint/clean_pins.json --out /joint/joint-clean-cache-v1 > logs/compose.log 2>&1
printf '0\\n' > logs/compose.exitcode
clean_sha=$(sha256sum joint-clean-cache-v1/manifest.json | cut -d' ' -f1)
docker run --name codex-joint-audio-native-ce-modality-v1 {common} \
 {CPU} 3600 /usr/bin/python3.12 /joint/source/experiments/chimera_recovery/audio_cache.py prepare \
 --source {shlex.quote(source_path)} --clean-manifest /joint/joint-clean-cache-v1/manifest.json --clean-manifest-sha256 "$clean_sha" \
 --admission /joint/inputs/admission/manifest.json --admission-sha256 {admission_sha} \
 --tokenizer-dir /model --charted-attention /joint/inputs/charted_attention.py \
 --amendment-sha256 312706e85efa8fbb996b67f220856adc76236bc516af1ba30f03a365d5b64055 \
 --out /joint/native-audio-ce-v1 > logs/native-ce.log 2>&1
printf '0\\n' > logs/native-ce.exitcode
'''
(HERE/'run_chain.sh').write_bytes(chain.encode())
plan={'schema':'chimera_recovery.joint_audio_native_launch.v1','root_revision':commit,'files':files,'input_transfer':transfer,
    'derived_admission_sha256':admission_sha,'original_joint_admission_sha256':'6d4d2eac4644179c615f60b8288b11b582096f6080aac225a29fad42946eed43',
    'cpu_image':CPU,'source_path':source_path,'source_rows':225,'full_source_inventory':352,'train_admitted':194,'held_excluded':31,
    'source_and_inputs_readonly':True,'original_encoder_proofs_readonly':True,'thread_budget':8,'memory_bytes':32*1024**3,
    'chain_sha256':sha(HERE/'run_chain.sh'),'training_started':False,'gpu_visible':False,'native_parent_xyz_parity':False}
(HERE/'launch_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
client=SSHClient.from_config(ServerConfig().get('ai'))
if not client.connect():raise RuntimeError('server-manager connect failed')
try:
    make='from pathlib import Path;import json;p=Path('+repr(REMOTE)+');assert not p.exists();p.mkdir();(p/"logs").mkdir();(p/"owner.json").write_text('+repr(json.dumps({'subagent':'modality_architecture_review','training_started':False}))+')'
    rc,out,err=client.execute('python3 -c '+shlex.quote(make),timeout=10)
    if rc:raise RuntimeError('fresh owned joint namespace failed: '+err)
    for path,target in ((HERE/'new_inputs.tar.gz','new_inputs.tar.gz'),(bundle,'source.tar.gz'),(HERE/'run_chain.sh','run_chain.sh'),
                        (HERE/'launch_plan.json','launch_plan.json'),(HERE/'inputs_manifest.json','inputs_manifest.json')):
        if not client.upload_file(str(path),REMOTE+'/'+target):raise RuntimeError('joint capsule upload failed: '+target)
    setup='''from pathlib import Path
import hashlib,json,shutil,tarfile
p=Path(REMOTE)
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as h:
  for chunk in iter(lambda:h.read(1<<24),b''):d.update(chunk)
 return d.hexdigest()
assert sha(p/'new_inputs.tar.gz')==INPUT_SHA and sha(p/'source.tar.gz')==SOURCE_SHA
for name,folder in [('new_inputs.tar.gz','inputs'),('source.tar.gz','source')]:
 (p/folder).mkdir()
 with tarfile.open(p/name,'r:gz') as h:
  for entry in h.getmembers():
   assert (p/folder/entry.name).resolve().is_relative_to((p/folder).resolve()) and entry.isfile()
  h.extractall(p/folder)
manifest=json.loads((p/'inputs_manifest.json').read_text())
for item in manifest['verified_byte_reuse']:
 target=(p/'inputs'/item['path']).resolve();donor=Path(item['donor_path']).resolve()
 assert target.is_relative_to((p/'inputs').resolve()) and donor.is_relative_to(Path(DONOR).resolve())
 assert not target.exists() and sha(donor)==item['sha256']
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(donor,target)
for item in manifest['files']:
 target=(p/'inputs'/item['path']).resolve()
 assert target.is_relative_to((p/'inputs').resolve()) and sha(target)==item['sha256'] and target.stat().st_size==item['bytes']
(p/'inputs_verified.json').write_text(json.dumps({'complete':True,'files':len(manifest['files']),'training_started':False})+'\\n')
'''.replace('REMOTE',repr(REMOTE)).replace('INPUT_SHA',repr(transfer['sha256'])).replace('SOURCE_SHA',repr(sha(bundle))).replace('DONOR',repr(SPEECH+'/inputs'))
    rc,out,err=client.execute('python3 -c '+shlex.quote(setup),timeout=20)
    if rc:raise RuntimeError('joint capsule verification/copy failed: '+err)
    start='import subprocess,json;from pathlib import Path;p=Path('+repr(REMOTE)+');assert not (p/"started").exists();(p/"started").touch();log=open(p/"logs/chain.log","wb");proc=subprocess.Popen(["timeout","18000","bash",str(p/"run_chain.sh")],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({"pid":proc.pid}))'
    rc,out,err=client.execute('python3 -c '+shlex.quote(start),timeout=10)
    receipt={'remote':REMOTE,'launch_exit':rc,'stdout':out,'stderr':err,'launch_plan_sha256':sha(HERE/'launch_plan.json'),
        'source_archive_sha256':sha(bundle),'training_started':False,'gpu_visible':False}
    (HERE/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
    if rc:raise RuntimeError('joint preparation chain launch failed')
finally:client.close()

