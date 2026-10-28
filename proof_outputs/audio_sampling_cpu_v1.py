"""Owned actual tiny CPU free-answer proof; source copies are immutable."""
import hashlib,json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
local=Path(__file__).resolve().parents[1]
version=sys.argv[1] if len(sys.argv)>1 else 'v1'
if version not in {'v1','v2'}:raise ValueError('select explicit private proof namespace')
remote='/home/cereal/omni-campaign/distill/codex-audio-sampling-tests-modality-01a1326f-'+version
files=['experiments/chimera_recovery/'+name for name in ('training.py','train_media.py','media_cache.py','stage_inheritance.py',
    'native_state_inheritance.py','train_audio.py','audio_cache.py','audio_paths.py','audio_dialogue.py','h3_audio_probe.py','h3_audio_chart.py','audio_sampling.py')]
files+=['tests/test_chimera_audio_sampling.py','tests/test_chimera_recovery_train_audio.py']
client=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not client.connect():raise RuntimeError('server-manager connect failed')
try:
    code='from pathlib import Path;p=Path('+repr(remote)+');assert not p.exists();p.mkdir();(p/"owner.json").write_text('+repr(json.dumps({'subagent':'modality_architecture_review','training_started':False}))+')'
    rc,out,err=client.execute('python3 -c '+shlex.quote(code),timeout=10)
    if rc:raise RuntimeError('fresh sampling namespace failed: '+err)
    pins=[]
    for name in files:
        path=local/name
        if not client.upload_file(str(path),remote+'/'+name):raise RuntimeError('sampling source upload failed: '+name)
        pins.append({'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    command=' '.join(['cd '+shlex.quote(remote)+' &&',"CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        'PYTHONPATH=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9/src',
        'CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9',
        'CHIMERA_D3_BASELINE=/home/cereal/omni-campaign/distill/tools/d3_hybrid_module.py',
        'CHIMERA_STREAMED_BASELINE=/home/cereal/omni-campaign/distill/tools/streamed_kd_trainer.py',
        'STREAMED_KD_TRAINER_PATH=/home/cereal/omni-campaign/distill/tools/streamed_kd_trainer.py',
        'nice -n15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -q -p no:cacheprovider',
        '--basetemp '+shlex.quote(remote+'/pytest-tmp'),'tests/test_chimera_audio_sampling.py'])
    run=local/('proof_outputs/audio_sampling_cpu_'+version+'.sh')
    run.write_bytes(('set -u\n'+command+'\nrc=$?\nprintf "%s\\n" "$rc" > '+shlex.quote(remote+'/exitcode')+'\nexit "$rc"\n').encode())
    if not client.upload_file(str(run),remote+'/run.sh'):raise RuntimeError('sampling script upload failed')
    launch='import subprocess,json;from pathlib import Path;p=Path('+repr(remote)+');assert not (p/"started").exists();(p/"started").touch();log=open(p/"output.log","wb");proc=subprocess.Popen(["timeout","180","bash",str(p/"run.sh")],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({"pid":proc.pid}))'
    rc,out,err=client.execute('python3 -c '+shlex.quote(launch),timeout=10)
    result={'schema':'chimera_recovery.audio_sampling_actual_cpu.v1','remote':remote,'files':pins,'command':command,
        'launch_exit':rc,'stdout':out,'stderr':err,'training_started':False,'gpu_visible':False,'actual_current_model_loaded':False}
    (local/('proof_outputs/audio_sampling_cpu_'+version+'_launch.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'remote':remote,'launch_exit':rc,'stdout':out,'stderr':err}))
    if rc:raise RuntimeError('sampling CPU launch failed')
finally:client.close()
