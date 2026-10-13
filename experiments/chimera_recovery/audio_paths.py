"""Byte-verified, explicit derived audio admission paths for private execution.

Original manifests and row files are retained unchanged. Only file locations
change; source IDs, admission, grouping and all original file digests remain.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

spec = importlib.util.spec_from_file_location('chimera_audio_path_common', Path(__file__).with_name('audio_cache.py'))
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)
R, Error = C.R, C.Error
POLICY = 'verified-byte-identical-audio-path-binding-v1'

def projection(value):
    if isinstance(value, list):
        return [projection(item) for item in value]
    if isinstance(value, dict):
        return {key: None if key == 'asset_path' or key == 'path' and 'sha256' in value else projection(item)
                for key, item in value.items()}
    return value

def rebind(manifest_path, expected_sha, out, execution_root):
    if out.exists() or not str(execution_root).startswith('/'):
        raise Error('path binding needs a fresh private output and absolute execution root')
    original = C.T.read_json(C.T.pinned_file({'path': str(manifest_path), 'sha256': expected_sha}, Path('.')))
    source = C.T.pinned_file(original['sources']['source'], manifest_path.parent)
    C.admission(C.pin(manifest_path), manifest_path.parent, source)
    rows_path = C.T.pinned_file(original['outputs']['admission'], manifest_path.parent)
    rows = [json.loads(line) for line in rows_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    out.mkdir(parents=True)
    files, mappings = {}, []
    def copied(path, digest):
        path = Path(path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        if R.sha(path) != digest:
            raise Error('path binding input changed: ' + str(path))
        if digest not in files:
            name = digest + '__' + path.name
            target = out / 'files' / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            if R.sha(target) != digest:
                raise Error('path binding copy differs')
            files[digest] = str(execution_root).rstrip('/') + '/files/' + name
        mappings.append({'original_path': str(path.resolve()), 'execution_path': files[digest], 'sha256': digest})
        return files[digest]
    def rewrite(value):
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: rewrite(item) for key, item in value.items()}
        if 'path' in value and 'sha256' in value:
            result['path'] = copied(value['path'], value['sha256'])
        if 'asset_path' in value and 'asset_sha256' in value:
            result['asset_path'] = copied(value['asset_path'], value['asset_sha256'])
        return result
    target_rows = rewrite(rows)
    if projection(target_rows) != projection(rows):
        raise Error('audio path binding changed non-path source eligibility or identities')
    sources = rewrite(original['sources'])
    (out / 'original').mkdir()
    shutil.copyfile(manifest_path, out / 'original' / 'manifest.json')
    shutil.copyfile(rows_path, out / 'original' / 'admission.jsonl')
    result = copy.deepcopy(original)
    result['sources'] = sources
    result['outputs'] = {}
    for name, subset in (('admission', target_rows), ('train_admitted', [row for row in target_rows if row['train_admitted']])):
        path = out / (name + '.jsonl')
        path.write_bytes(b''.join(R.canonical(row) + b'\n' for row in subset))
        result['outputs'][name] = {'path': str(execution_root).rstrip('/') + '/' + path.name, 'sha256': R.sha(path), 'rows': len(subset)}
    result['source_path_binding'] = {'policy': POLICY, 'dated': '2026-10-13', 'adapter_sha256': R.sha(Path(__file__)),
        'original_manifest': {'path': str(execution_root).rstrip('/') + '/original/manifest.json', 'sha256': expected_sha},
        'original_admission': {'path': str(execution_root).rstrip('/') + '/original/admission.jsonl', 'sha256': R.sha(rows_path)},
        'semantic_rows_sha256': R.digest(projection(rows)), 'sources_sha256': R.digest({name: ref['sha256'] for name, ref in original['sources'].items()}),
        'original_manifest_seal': original.get('manifest_sha256'), 'only_paths_rebound': True}
    result.pop('manifest_sha256', None)
    result['manifest_sha256'] = R.digest(result)
    (out / 'manifest.json').write_bytes(R.canonical(result) + b'\n')
    (out / 'transfer_receipt.json').write_bytes(R.canonical({'schema': 'chimera_recovery.audio_paths.v1', 'policy': POLICY,
        'original_manifest_sha256': expected_sha, 'derived_manifest_sha256': R.sha(out / 'manifest.json'),
        'mappings': mappings, 'unique_files': len(files), 'source_bytes': sum(path.stat().st_size for path in (out / 'files').iterdir()),
        'training_started': False}) + b'\n')
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--execution-root', required=True)
    args = parser.parse_args(argv)
    result = rebind(args.manifest, args.manifest_sha256, args.out, args.execution_root)
    print(json.dumps({'rows': result['rows'], 'train_admitted': result['train_admitted'], 'training_started': False}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
