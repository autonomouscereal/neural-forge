"""Package existing public QIMG metadata only; never query or act on a runtime."""
import base64
import hashlib
import json
import os
from pathlib import Path
import zlib

HERE = Path(__file__).resolve().parent
RECOVERY = HERE.parents[1]
WORKSPACE = RECOVERY.parents[3]
Q = WORKSPACE / 'native_executor/qimg_first32/01a111e4-2e50-7eb0-916a-74e1058b3a9c'


def read(path, expected, size=None):
    path = Path(path).resolve()
    assert path.is_file() and not path.is_symlink() and path.stat().st_size <= 16 << 20
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    assert hashlib.sha256(raw).hexdigest() == expected and (size is None or len(raw) == size)
    return raw


def member(path, expected, size=None):
    raw = read(path, expected, size)
    return {'path': str(Path(path).resolve()), 'sha256': expected, 'bytes': len(raw),
            'encoding': 'zlib-base64',
            'raw_zlib_base64': base64.b64encode(zlib.compress(raw, 9)).decode('ascii')}


def main():
    refs = {
        'terminal': (Q / 'ACTUAL_NATIVE_TERMINAL_METADATA_FULL_READBACK_V1-20261008T011131147377Z.json',
                     '5289b9b8869f8bdf86316b5721293e773c9303d907e8895b1c22df0a442650a6', 62825),
        'release': (Q / 'actual_native_preserving_release-ddd71487-ac39-4d09-ab2b-c25c438382cb/ACTUAL_EXACT_OWN_PRESERVING_RELEASE_COMPLETED_V1.json',
                    '6ddcd4815040a345b12e979df272633e09593ad9561dec5a90df95c4036e0b63', 11302),
        'owner_closure': (Q / 'ACTUAL_QIMG_SAME_NATIVE_TERMINAL_CLOSURE_CHECKPOINT_V37-20261008T011504064114Z.json',
                          '54c19ac7a14e0849578cf85afe4dc7798c6cf72b9fc3eb1d6169a67e1897af63', 3594),
        'owner_closure_source': (Q / 'persist_actual_native_complete_terminal_and_release_v37.py',
                                 'f2d040fb042765bac1636ab1dcb685e26f45c884b9b521263e3b712e1b8f2a3c', None),
    }
    outer = {role: member(*ref) for role, ref in refs.items()}
    terminal = json.loads(read(*refs['terminal']))
    release = json.loads(read(*refs['release']))
    raws = []
    for row in terminal['full_readbacks']:
        raws.append({'role': row['role'], **member(row['path'], row['sha256'], row['bytes'])})
    assert len(raws) == 62 and len({row['role'] for row in raws}) == 62
    physical = [{'role': row['label'], **member(row['path'], row['sha256'], row['bytes'])}
                for row in release['fresh_physical_readbacks']]
    total = sum(row['bytes'] for row in outer.values()) + sum(row['bytes'] for row in raws + physical)
    assert total <= 64 << 20
    proof = {'schema': 'neural_forge.k3_QIMG_exact_native_compute_closure_public_proof.v1',
             'source_only': True, 'fresh_current_row_proven': False,
             'outer': outer, 'terminal_raw_members': raws, 'release_physical_members': physical,
             'raw_terminal_member_count': 62, 'raw_terminal_bytes': sum(row['bytes'] for row in raws),
             'all_decoded_metadata_and_source_bytes': total,
             'per_raw_member_limit_bytes': 16 << 20, 'aggregate_decoded_limit_bytes': 64 << 20,
             'retained_disk_credit_bytes': 0, 'foreign_provider_scope_discharged': False}
    raw = (json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    assert len(raw) <= 16 << 20
    path = HERE / 'QIMG_EXACT_NATIVE_COMPUTE_CLOSURE_PUBLIC_PROOF_V1.json'
    with path.open('xb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    assert path.read_bytes() == raw
    print(json.dumps({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                      'bytes': len(raw), 'raw_terminal_bytes': proof['raw_terminal_bytes'],
                      'all_decoded_metadata_and_source_bytes': total, 'runtime_actions': False}))


if __name__ == '__main__':
    main()
