"""ROOT-invoked readonly normal-PG witness; no connection occurs on import.

Uses the existing installed desktop_bridge.memory_query readonly transaction.
No allocator connect(), schema creation, reconciliation or identity operation.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

BRIDGE = Path('C:/Users/cereal/Documents/Codex/2026-10-01/task/desktop_bridge.py')
BRIDGE_SHA = '32f498e8487eba53b845e4714a8696ec71c35d91fc514bfa34283f79699f3e10'
OWNER = 'codex-native-01a111e4-2e50-7eb0-916a-74e1058b3a9c'
ALLOCATION = 'f8a611e8-6c1b-4ff2-8a4a-5149bf9bb077'
CID = '7fb0e676b730d2dded9294d5318a18ba347e67c7d096a759a65dbc5946fd610f'
SQL = ("SELECT * FROM gpu_runtime_allocations WHERE server=$1 AND "
       "(allocation_id=$2 OR (owner=$3 AND state IN ('starting','running'))) "
       "ORDER BY allocation_id")


def normalize(value):
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError('timezone-aware PG timestamp required')
        return value.isoformat()
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError('unsupported raw PG value')


async def capture(attempt):
    assert BRIDGE.is_file() and not BRIDGE.is_symlink()
    assert hashlib.sha256(BRIDGE.read_bytes()).hexdigest() == BRIDGE_SHA
    spec = importlib.util.spec_from_file_location('k3_existing_readonly_bridge', BRIDGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = datetime.now(timezone.utc).isoformat()
    rows = normalize(await module.memory_query(SQL, 'ai', uuid.UUID(ALLOCATION), OWNER))
    # asyncpg JSONB may be returned as a JSON string; preserve its semantics.
    for row in rows:
        if isinstance(row.get('details'), str):
            row['details'] = json.loads(row['details'])
    matching = [row for row in rows if row['allocation_id'] == ALLOCATION]
    assert len(matching) == 1
    row = matching[0]
    assert row['owner'] == OWNER and row['container_id'] == CID and row['state'] == 'released'
    assert row['group_id'] == ALLOCATION and row['server'] == 'ai'
    assert not any(item['state'] in ('starting', 'running') for item in rows)
    return {'schema': 'neural_forge.k3_QIMG_current_normal_released_row_witness.v1',
            'at': datetime.now(timezone.utc).isoformat(), 'query_started_at': started,
            'attempt_uuid': attempt, 'read_only': True,
            'producer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'bridge_source_sha256': BRIDGE_SHA, 'query': SQL,
            'query_parameters': ['ai', ALLOCATION, OWNER], 'rows': rows,
            'provider_or_allocator_or_identity_mutations': False,
            'dated_release_row_restamped': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--attempt-uuid', required=True)
    args = parser.parse_args()
    assert str(uuid.UUID(args.attempt_uuid)) == args.attempt_uuid
    root = args.output_directory
    assert root.is_absolute() and root.resolve() == root and root.is_dir() and not root.is_symlink()
    assert root.is_relative_to(Path('D:/codex-owned-source-receipts'))
    doc = asyncio.run(capture(args.attempt_uuid))
    raw = (json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    assert len(raw) <= 16 << 20
    path = root / ('QIMG-current-released-row-' + args.attempt_uuid + '.json')
    with path.open('xb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    assert path.read_bytes() == raw
    print(json.dumps({'path': str(path), 'name': path.name,
                      'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                      'request_field': 'QIMG_current_released_row_witness', 'read_only': True}))


if __name__ == '__main__':
    main()
