import hashlib,json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
local=Path(__file__).resolve().parents[1]
remote='/home/cereal/omni-campaign/distill/codex-audio-replay-index-modality-tests-v1'
files=['experiments/chimera_recovery/audio_replay_index.py','tests/test_chimera_audio_replay_index.py']
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('server-manager connect failed')
try:
 rc,out,err=c.execute('python3 -c '+shlex.quote('from pathlib import Path;p=Path('+repr(remote)+');assert not p.exists();p.mkdir()'),timeout=10)
 if rc:raise RuntimeError('fresh replay proof namespace failed: '+err)
 pins=[]
 for name in files:
  path=local/name
  if not c.upload_file(str(path),remote+'/'+name):raise RuntimeError('source upload failed')
  pins.append({'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 command='cd '+shlex.quote(remote)+" && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -q -p no:cacheprovider tests/test_chimera_audio_replay_index.py"
 rc,out,err=c.execute(command,timeout=20)
 result={'remote':remote,'files':pins,'command':command,'exit_code':rc,'output':out,'stderr':err,'gpu_exposed':False,'training_started':False}
 (local/'proof_outputs/audio_replay_index_cpu_v1_result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
 if rc:raise RuntimeError('actual CPU replay index tests failed')
finally:c.close()
