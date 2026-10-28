"""Own source-isolated genuine CPU fixture, via server-manager only."""
import hashlib
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0, 'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig, SSHClient
local = Path(__file__).resolve().parents[1]
version = sys.argv[1] if len(sys.argv) > 1 else 'v4'
if version not in {'v4'}:
    raise ValueError('explicit owned fixture version required')
remote = '/home/cereal/omni-campaign/distill/codex-speech-tests-modality-01a1326f-' + version
sources = ['experiments/chimera_recovery/' + name for name in (
    'training.py', 'train_media.py', 'media_cache.py', 'stage_inheritance.py', 'native_state_inheritance.py',
    'train_audio.py', 'audio_cache.py', 'audio_paths.py', 'audio_dialogue.py', 'h3_audio_probe.py', 'h3_audio_chart.py',
    'speech_flow.py', 'train_speech_flow.py')]
sources += ['tests/' + name for name in ('test_chimera_speech_flow.py', 'test_chimera_audio_cache.py', 'test_chimera_recovery_audio_dialogue.py', 'test_chimera_recovery_h3_audio_probe.py')]
client = SSHClient.from_config(ServerConfig().get('ai'), control=True)
if not client.connect():
    raise RuntimeError('server-manager connect failed')
try:
    code, out, err = client.execute('python3 -c ' + shlex.quote('from pathlib import Path;p=Path(' + repr(remote) + ');assert not p.exists();p.mkdir();(p/"owner.json").write_text(' + repr(json.dumps({'native_task': '01a1326f-4f44-7ef0-a2fa-2a64a706fedc', 'subagent': 'modality_architecture_review', 'training_started': False})) + ')'))
    if code:
        raise RuntimeError('fresh source-isolated fixture directory failed: ' + err)
    files = []
    for name in sources:
        path = local / name
        if not client.upload_file(str(path), remote + '/' + name):
            raise RuntimeError('source upload failed: ' + name)
        files.append({'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    command = ' '.join([
        'cd ' + shlex.quote(remote) + ' &&',
        "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        'PYTHONPATH=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9/src',
        'CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9',
        'CHIMERA_D3_BASELINE=/home/cereal/omni-campaign/distill/tools/d3_hybrid_module.py',
        'CHIMERA_STREAMED_BASELINE=/home/cereal/omni-campaign/distill/tools/streamed_kd_trainer.py',
        'STREAMED_KD_TRAINER_PATH=/home/cereal/omni-campaign/distill/tools/streamed_kd_trainer.py',
        'nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -p no:cacheprovider',
        '--basetemp ' + shlex.quote(remote + '/pytest-tmp'),
        'tests/test_chimera_speech_flow.py -k speaking_sampler -q'])
    script = 'set -u\n' + command + '\nrc=$?\nprintf "%s\\n" "$rc" > ' + shlex.quote(remote + '/exitcode') + '\nexit "$rc"\n'
    shell = local / ('proof_outputs/speech_flow_cpu_' + version + '.sh')
    shell.write_bytes(script.encode())
    if not client.upload_file(str(shell), remote + '/run.sh'):
        raise RuntimeError('fixture command upload failed')
    launch = 'import subprocess,json;from pathlib import Path;p=Path(' + repr(remote) + ');assert not (p/"started").exists();(p/"started").touch();log=open(p/"output.log","wb");proc=subprocess.Popen(["timeout","120","bash",str(p/"run.sh")],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({"pid":proc.pid}))'
    code, out, err = client.execute('python3 -c ' + shlex.quote(launch), timeout=10)
    receipt = {'schema': 'chimera_recovery.speech_flow_actual_cpu.v1', 'remote': remote, 'files': files,
               'command': command, 'launch_exit': code, 'launch_stdout': out, 'launch_stderr': err,
               'training_started': False, 'gpu_visible': False, 'actual_current_model_loaded': False}
    (local / ('proof_outputs/speech_flow_cpu_' + version + '_launch.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'remote': remote, 'launch_exit': code, 'stdout': out, 'stderr': err}))
    if code:
        raise RuntimeError('genuine CPU fixture launch failed')
finally:
    client.close()
