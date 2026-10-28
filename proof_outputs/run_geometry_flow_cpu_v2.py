import hashlib,json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import SSHClient,ServerConfig
OWN=Path(__file__).resolve().parents[1];MAIN=Path('C:/Users/cereal/Documents/ai training')
LOCAL=MAIN/'reviews/2026-10-13-folding/recovery-preparation/geometry-flow-cpu-v2'
OLD=MAIN/'reviews/2026-10-13-folding/recovery-preparation/geometry-flow-cpu-v1'
ROOT='/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013'
REMOTE=ROOT+'/geometry-flow-cpu-v2'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,v):path.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
LOCAL.mkdir(parents=True,exist_ok=True)
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise SystemExit('connection failed')
try:
 if sys.argv[1]=='launch':
  previous=json.loads((OLD/'launch.json').read_text())
  test=OWN/'tests/test_chimera_geometry_flow.py'
  runner=LOCAL/'runner.py';text=(OLD/'runner.py').read_text()
  text=text.replace("archive=root/'source_capsule.zip'","archive=Path('/work/base/source_capsule.zip')")
  needle="capsule=output/'capsule';aliases=json.loads((capsule/'aliases.json').read_text())"
  new=needle+"\noverride=root/'test_chimera_geometry_flow.py'\nif hashlib.sha256(override.read_bytes()).hexdigest()!=TEST_OVERRIDE_SHA:raise ValueError('fixture override seal differs')\n(capsule/'code/tests/test_chimera_geometry_flow.py').write_bytes(override.read_bytes())"
  if text.count(needle)!=1:raise ValueError('runner splice ambiguity')
  text=text.replace(needle,new)
  text=text.replace("'source_capsule_sha256':CAPSULE_SHA,","'source_capsule_sha256':CAPSULE_SHA,'test_fixture_sha256':TEST_OVERRIDE_SHA,")
  text='TEST_OVERRIDE_SHA='+repr(sha(test))+'\n'+text
  runner.write_text(text,encoding='utf-8')
  setup="from pathlib import Path;import json;p=Path("+repr(REMOTE)+");p.mkdir(exist_ok=False);[(p/x).mkdir() for x in ['input','output']];print(json.dumps({'mem_available_kib':int([x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0].split()[1])}))"
  rc,out,err=c.execute('python3 -c '+shlex.quote(setup),timeout=10)
  if rc:raise SystemExit(err)
  memory=json.loads(out)
  if memory['mem_available_kib']<40*1024**2:raise ValueError('insufficient bounded CPU headroom')
  for file in (runner,test):
   if not c.upload_file(str(file),REMOTE+'/input/'+file.name):raise ValueError('upload failed')
  argv=[v.replace('codex-geometry-flow-cpu-oct13-v1','codex-geometry-flow-cpu-oct13-v2').replace(ROOT+'/geometry-flow-cpu-v1/input',REMOTE+'/input').replace(ROOT+'/geometry-flow-cpu-v1/output',REMOTE+'/output') for v in previous['argv']]
  index=argv.index('--entrypoint')
  argv[index:index]=['--mount','type=bind,src='+ROOT+'/geometry-flow-cpu-v1/input,dst=/work/base,readonly']
  rc,out,err=c.execute(shlex.join(argv),timeout=15)
  receipt={'schema':'recovery_geometry_flow_cpu_launch.v2','argv':argv,'source_capsule_sha256':previous['source_capsule_sha256'],'test_fixture_sha256':sha(test),'runner_sha256':sha(runner),'container':out.strip(),'exit_code':rc,'stderr':err,'memory_before':memory,'launcher_sha256':sha(__file__),'training_started':False,'cpu_only':True}
  dump(LOCAL/'launch.json',receipt);print(json.dumps({k:receipt[k] for k in ('container','exit_code','memory_before')}))
 elif sys.argv[1]=='status':
  if c.download_file(REMOTE+'/output/status.json',str(LOCAL)):print((LOCAL/'status.json').read_text())
 elif sys.argv[1]=='collect':
  for name in ('tests.json','proof.json'):
   if c.download_file(REMOTE+'/output/'+name,str(LOCAL)):print((LOCAL/name).read_text())
  if c.download_file(REMOTE+'/output/native-batches/manifest.json',str(LOCAL)):print('actual batch manifest SHA',sha(LOCAL/'manifest.json'))
 elif sys.argv[1]=='logs':
  receipt=json.loads((LOCAL/'launch.json').read_text());rc,out,err=c.execute('docker logs --tail 40 '+shlex.quote(receipt['container']),timeout=10);print(out);print(err)
 else:raise SystemExit('launch/status/collect/logs only')
finally:c.close()
