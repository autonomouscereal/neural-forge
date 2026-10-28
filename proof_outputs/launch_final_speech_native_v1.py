"""Source-isolated real H3 CPU201 proof and native prep. No optimizer/GPU."""
from pathlib import Path
import hashlib,json,shlex,shutil,subprocess,sys,tarfile
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
HERE=Path(__file__).resolve().parent/'speech_native_prepare_v1'
BASE=Path('C:/Users/cereal/Documents/ai training')
PROMOTED=BASE/'.agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-training-recovery-oct13-eae1b72e2c3f'
REMOTE='/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1'
COMFY='sha256:5070872ac7aef716eadc60ec92b0f944521c7899d90c6899a7b1ed1e883942e1'
CPU='sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164'
OWNER={'native_task':'01a1326f-4f44-7ef0-a2fa-2a64a706fedc','subagent':'modality_architecture_review','training_started':False}
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1<<24),b''):digest.update(chunk)
    return digest.hexdigest()
transfer=json.loads((HERE/'transfer.json').read_text())
if sha(HERE/'inputs.tar.gz')!=transfer['sha256']:raise ValueError('portable full source capsule changed')
source=HERE/'source';source.mkdir(exist_ok=False)
commit=subprocess.run(['git','rev-parse','HEAD'],cwd=PROMOTED,text=True,capture_output=True,check=True).stdout.strip()
files=[]
for name in ('training.py','train_media.py','media_cache.py','stage_inheritance.py','native_state_inheritance.py',
             'train_audio.py','audio_cache.py','audio_paths.py','audio_dialogue.py','h3_audio_probe.py','h3_audio_chart.py',
             'speech_flow.py','train_speech_flow.py'):
    original=PROMOTED/'experiments/chimera_recovery'/name
    target=source/'experiments/chimera_recovery'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    before=sha(original);shutil.copyfile(original,target)
    if sha(original)!=before or sha(target)!=before:raise ValueError('promoted source changed while sealing: '+name)
    files.append({'path':str(target.relative_to(source)).replace('\\','/'),'sha256':before})
if subprocess.run(['git','rev-parse','HEAD'],cwd=PROMOTED,text=True,capture_output=True,check=True).stdout.strip()!=commit:raise ValueError('root source revision changed while sealing')
inventory=json.loads((HERE/'inputs/admission/manifest.json').read_text())
source_path=inventory['sources']['source']['path']
admission_sha=sha(HERE/'inputs/admission/manifest.json')
chain='''set -euo pipefail
cd "{remote}"
test ! -e clean-speech-cache-v1
test ! -e native-speech-ce-v1
test ! -e native-speech-flow-v1
docker run --name codex-speech-clean-modality-v1 --network none --cpus 8 --memory 32g --no-healthcheck \
 --entrypoint /usr/bin/timeout \
 -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=8 \
 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work/source \
 -v "{remote}:/work:rw" -v "{remote}/source:/work/source:ro" -v "{remote}/inputs:/work/inputs:ro" \
 -v /home/cereal/comfyui/models:/opt/ComfyUI/models:ro \
 {comfy} 10800 /opt/environments/python/comfyui/bin/python \
 /work/source/experiments/chimera_recovery/audio_cache.py encode-admitted-source \
 --source {source_path} --admission /work/inputs/admission/manifest.json --admission-sha256 {admission_sha} \
 --stage-c /work/inputs/stage_c_media_a.py --comfy-root /opt/ComfyUI \
 --audio-vae /opt/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors \
 --reuse-runtime --out /work/clean-speech-cache-v1 > logs/encoder.log 2>&1
printf '0\\n' > logs/encoder.exitcode
clean_sha=$(sha256sum clean-speech-cache-v1/manifest.json | cut -d' ' -f1)
for phase in ce flow; do
 if [ "$phase" = ce ]; then entry='audio_cache.py prepare'; else entry='speech_flow.py'; fi
 docker run --name "codex-speech-native-$phase-modality-v1" --network none --cpus 8 --memory 32g --no-healthcheck \
  --entrypoint /usr/bin/timeout \
  -e CUDA_VISIBLE_DEVICES= -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
  -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/opt/teacher-runtime/lib/python3.12/site-packages:/work/source \
  -v /home/cereal/model-surgery-lab/venv:/opt/teacher-runtime:ro \
  -v /home/cereal/models/Qwen3.8-27B-Uncensored:/model:ro \
  -v "{remote}:/work:rw" -v "{remote}/source:/work/source:ro" -v "{remote}/inputs:/work/inputs:ro" \
  -v "{remote}/clean-speech-cache-v1:/work/clean-speech-cache-v1:ro" \
  {cpu} 3600 /usr/bin/python3.12 /work/source/experiments/chimera_recovery/$entry \
  --source {source_path} --clean-manifest /work/clean-speech-cache-v1/manifest.json --clean-manifest-sha256 "$clean_sha" \
  --admission /work/inputs/admission/manifest.json --admission-sha256 {admission_sha} \
  --tokenizer-dir /model --charted-attention /work/inputs/charted_attention.py \
  --amendment-sha256 312706e85efa8fbb996b67f220856adc76236bc516af1ba30f03a365d5b64055 \
  --out "/work/native-speech-$phase-v1" > "logs/native-$phase.log" 2>&1
 printf '0\\n' > "logs/native-$phase.exitcode"
done
printf '0\\n' > logs/chain.exitcode
'''.format(remote=REMOTE,comfy=COMFY,cpu=CPU,source_path=shlex.quote(source_path),admission_sha=admission_sha)
(HERE/'run_chain.sh').write_bytes(chain.encode())
plan={'schema':'chimera_recovery.final_speech_native_launch.v1','owner':OWNER,'root_revision':commit,'source_files':files,
      'inputs':transfer,'images':{'comfy':COMFY,'cpu':CPU},'thread_budget':8,'memory_bytes':32*1024**3,
      'source_manifest_sha256':sha(HERE/'inputs_manifest.json'),'derived_admission_sha256':admission_sha,
      'source_ready_sha256':inventory['sources']['source']['sha256'],'source_rows':201,'train_admitted':186,'held_rows':15,
      'original_inventory_rows':208,'quarantine':7,'native_ce_batches_expected':186,'native_speaking_batches_expected':744,
      'native_source_xyz_parent_parity':False,'selected_preregistration_sha256':'cf676e8d0b1c63052972921059fed97628122fb2257c6c4236ee421a923d9a91',
      'amendment_sha256':'312706e85efa8fbb996b67f220856adc76236bc516af1ba30f03a365d5b64055','training_started':False,'gpu_exposed':False,
      'chain_sha256':sha(HERE/'run_chain.sh'),'source_mount_readonly':True,'inputs_mount_readonly':True}
(HERE/'launch_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
bundle=HERE/'source.tar.gz'
with tarfile.open(bundle,'w:gz',compresslevel=1) as handle:
    for item in files:handle.add(source/item['path'],arcname=item['path'],recursive=False)
client=SSHClient.from_config(ServerConfig().get('ai'))
if not client.connect():raise RuntimeError('server-manager connect failed')
try:
    make='from pathlib import Path;import json;p=Path('+repr(REMOTE)+');assert not p.exists();p.mkdir();(p/"owner.json").write_text('+repr(json.dumps(OWNER))+');(p/"logs").mkdir()'
    rc,out,err=client.execute('python3 -c '+shlex.quote(make),timeout=10)
    if rc:raise RuntimeError('fresh owned namespace failed: '+err)
    for path,target in [(HERE/'inputs.tar.gz','inputs.tar.gz'),(bundle,'source.tar.gz'),(HERE/'run_chain.sh','run_chain.sh'),
                        (HERE/'launch_plan.json','launch_plan.json'),(HERE/'inputs_manifest.json','inputs_manifest.json')]:
        if not client.upload_file(str(path),REMOTE+'/'+target):raise RuntimeError('immutable capsule upload failed: '+target)
    prepare='''set -euo pipefail
cd "{remote}"
printf '%s  %s\\n' '{inputs_sha}' inputs.tar.gz '{source_sha}' source.tar.gz | sha256sum --check -
mkdir inputs source
tar --extract --gzip --file inputs.tar.gz --directory inputs
tar --extract --gzip --file source.tar.gz --directory source
'''.format(remote=REMOTE,inputs_sha=transfer['sha256'],source_sha=sha(bundle))
    rc,out,err=client.execute_script(prepare)
    if rc:raise RuntimeError('source/input capsule verification failed: '+err)
    code='import subprocess,json;from pathlib import Path;p=Path('+repr(REMOTE)+');assert not (p/"started").exists();(p/"started").touch();log=open(p/"logs/chain.log","wb");proc=subprocess.Popen(["bash",str(p/"run_chain.sh")],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({"pid":proc.pid}))'
    rc,out,err=client.execute('python3 -c '+shlex.quote(code),timeout=10)
    receipt={'remote':REMOTE,'exit_code':rc,'stdout':out,'stderr':err,'launch_plan_sha256':sha(HERE/'launch_plan.json'),
             'source_archive_sha256':sha(bundle),'training_started':False,'gpu_exposed':False}
    (HERE/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    if rc:raise RuntimeError('native preparation chain launch failed')
finally:client.close()
