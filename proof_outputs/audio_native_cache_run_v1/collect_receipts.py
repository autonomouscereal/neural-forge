from pathlib import Path
import sys
sys.path.insert(0, 'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import SSHClient, ServerConfig
remote = '/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013'
out = Path(__file__).resolve().parent / 'actual_receipts'
client = SSHClient.from_config(ServerConfig().get('ai'))
if not client.connect():
    raise RuntimeError('server-manager connection failed')
try:
    for name, path in [('audit', 'native-audio-audit-v1/result.json'), ('clean', 'clean-audio-cache-v1/manifest.json'), ('native', 'native-audio-batches-v1/manifest.json')]:
        destination = out / name
        destination.mkdir(parents=True, exist_ok=True)
        if not client.download_file(remote + '/' + path, str(destination)):
            if name != 'native':
                raise RuntimeError('receipt download failed: ' + name)
            script = '''set -euo pipefail
docker run --rm --network none --cpus 1 --memory 256m --entrypoint /usr/bin/python3.12 \\
  -v "{remote}/native-audio-batches-v1:/receipt:ro" \\
  sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164 \\
  -c 'import sys;sys.stdout.buffer.write(open("/receipt/manifest.json","rb").read())'
'''.format(remote=remote)
            code, stdout, stderr = client.execute_script(script)
            if code:
                raise RuntimeError('read-only receipt copy failed: ' + stderr)
            (destination / 'manifest.json').write_bytes(stdout.encode('utf-8'))
finally:
    client.close()
