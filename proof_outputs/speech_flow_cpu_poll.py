import hashlib
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0, 'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig, SSHClient
local = Path(__file__).resolve().parent
version = sys.argv[1] if len(sys.argv) > 1 else 'v1'
if version not in {'v1', 'v2', 'v3'}:
    raise ValueError('explicit owned fixture version required')
receipt = json.loads((local / ('speech_flow_cpu_' + version + '_launch.json')).read_text())
remote = receipt['remote']
client = SSHClient.from_config(ServerConfig().get('ai'), control=True)
if not client.connect():
    raise RuntimeError('server-manager connect failed')
try:
    code = '''from pathlib import Path
import hashlib,json
p=Path({remote!r});f={files!r}
print(json.dumps({{'done':(p/'exitcode').exists(),'exitcode':(p/'exitcode').read_text().strip() if (p/'exitcode').exists() else None,
'stdout':(p/'output.log').read_text() if (p/'output.log').exists() else '',
'files':{{x['path']:hashlib.sha256((p/x['path']).read_bytes()).hexdigest() for x in f}}}}))
'''.format(remote=remote, files=receipt['files'])
    rc, out, err = client.execute('python3 -c ' + shlex.quote(code), timeout=10)
    if rc:
        raise RuntimeError('fixture polling failed: ' + err)
    observed = json.loads(out)
    observed['source_snapshot_unchanged'] = observed['files'] == {x['path']: x['sha256'] for x in receipt['files']}
    result = {**receipt, **observed, 'test_exitcode': observed['exitcode']}
    (local / ('speech_flow_cpu_' + version + '_result.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:observed[k] for k in ('done','exitcode','stdout','source_snapshot_unchanged')}))
finally:
    client.close()
