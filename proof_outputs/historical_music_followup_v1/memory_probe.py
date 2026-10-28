import hashlib
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
CLI = Path('C:/Users/cereal/.claude/memory_backend/agent_memory.py')
TARGET = 'c4f73350b7d516d3e7448221f12cdae621a72de3097046c7e40db3b8655aaa16'

def save(name, args):
    p = subprocess.run([sys.executable, str(CLI), '--json', *args], capture_output=True, text=True, encoding='utf-8', timeout=35)
    (OUT / (name + '.json')).write_text(p.stdout, encoding='utf-8')
    (OUT / (name + '.stderr.txt')).write_text(p.stderr, encoding='utf-8')
    try:
        data = json.loads(p.stdout)
    except ValueError:
        data = {'parse_failed': True, 'stdout_bytes': len(p.stdout.encode())}
    print(json.dumps({'name': name, 'exit': p.returncode, 'type': type(data).__name__, 'keys': list(data) if isinstance(data, dict) else None, 'sha256': hashlib.sha256(p.stdout.encode()).hexdigest()}))
    return data

if __name__ == '__main__':
    data = save('pg_exact_writer_hash', ['grep', TARGET, '--contains', '--agent', 'ClaudeCode', '--all-spaces', '--limit', '20', '--timeout-ms', '12000'])
    print(json.dumps(data, ensure_ascii=False)[:28000])
    data = save('pg_event_884355_trace', ['trace', '--event-id', '884355', '--all-spaces', '--limit', '1'])
    print(json.dumps(data, ensure_ascii=False)[:10000])

