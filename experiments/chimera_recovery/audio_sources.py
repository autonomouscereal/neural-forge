"""Source-preserving global procedural/speech CE composition, never training.

Original admissions and native proofs remain immutable. A new joint admission
re-traverses the full registry/evaluation/prior/source inventories; old exclusions
cannot be lifted by composition. Only its exact accepted rows reach native CE.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('chimera_audio_sources_cache', HERE / 'audio_cache.py')
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)
R, T, Error = C.R, C.T, C.Error
COMMON = ('normalized', 'corpus_manifest', 'eval_rows', 'frozen_exclusions', 'prior_admission')


def helper(path, expected_sha):
    path = T.pinned_file({'path': str(path), 'sha256': expected_sha}, Path('.'))
    module = T.load_module(path, 'chimera_audio_sources_global_admission')
    if not callable(getattr(module, 'seal_admission', None)):
        raise Error('source-pinned actual global admission helper is required')
    return module


def check_speech(A, record, admitted):
    """Reverify full original/derived PCM prefix and prior exact teacher joins."""
    p, media = record['provenance'], record['media'][0]
    if (record['route'] != 'speech_understand' or p.get('independent_asr_exact_match') is not True or
            p.get('synthetic_official_voice') is not True or p.get('user_reference_voice_used') is not False or
            p.get('speaker') not in {'Ryan', 'Aiden', 'Vivian', 'Serena'} or
            [m.get('role') for m in record['messages']] != ['user', 'assistant'] or
            record.get('supervision', {}).get('policy') != 'assistant_only' or
            record['supervision'].get('message_mask') != [False, True] or
            not isinstance(record['messages'][-1].get('content'), str) or not record['messages'][-1]['content'].strip()):
        raise Error('joint speech source requires prior exact ASR and official voice provenance')
    original, derived = Path(p['original_audio_path']), Path(media['path'])
    a, raw = A.pcm(original)
    b, pcm = A.pcm(derived)
    extra = (-a['samples']) % (a['sample_rate'] // 40)
    if (A.file_sha(original) != p['original_audio_sha256'] or A.file_sha(derived) != media['sha256'] or
            media.get('source_media_sha256') != p['original_audio_sha256'] or
            a['sample_rate'] != b['sample_rate'] or a['channels'] != b['channels'] or
            b['samples'] != a['samples'] + extra or pcm[:len(raw)] != raw or any(pcm[len(raw):])):
        raise Error('joint source full original PCM/minimal zero-tail provenance differs')
    derivation = p['minimal_tail_derivation']
    for key, value in {'original': a, 'derived': b, 'tail_padding_samples': extra,
                       'original_asset_sha256': p['original_audio_sha256'], 'derived_asset_sha256': media['sha256']}.items():
        if derivation.get(key) != value:
            raise Error('joint source actual PCM measure differs from sealed source producer')
    evidence = admitted.get('teacher_evidence', {})
    if evidence.get('independent_asr_exact_match') is not True:
        raise Error('joint speech source prior independent ASR agreement is absent')
    for name in ('plan_row_sha256', 'tts_row_sha256', 'asr_row_sha256'):
        if evidence.get(name) != p.get(name):
            raise Error('joint speech source prior plan/TTS/ASR exact row join differs')


def prepare(procedural_admission, speech_admission, *, procedural_sha, speech_sha, admission_helper, helper_sha,
            procedural_asset_root, out):
    if out.exists():
        raise Error('preserve prior source admissions; select a fresh joint output')
    A = helper(admission_helper, helper_sha)
    parents, candidate_sets, inventories = [], [], []
    for label, path in (('procedural', procedural_admission), ('speech', speech_admission)):
        path = T.pinned_file({'path': str(path), 'sha256': procedural_sha if label == 'procedural' else speech_sha}, Path('.'))
        data = T.read_json(path)
        source = T.pinned_file(data['sources']['source'], path.parent)
        _, admitted = C.admission(C.pin(path), path.parent, source)
        rows = C.source_rows(source)
        inventory_path = T.pinned_file(data['sources']['source_inventory'], path.parent)
        inventory = list(A.rows(inventory_path))
        if len({row.get('id') for row in inventory}) != len(inventory):
            raise Error('joint source requires each original inventory row exactly once')
        all_rows = {row['id']: row for row in inventory}
        if any(ident not in all_rows or R.digest(record) != R.digest(all_rows[ident]) for ident, record in rows.items()):
            raise Error('joint source candidate differs from the full original inventory')
        if label == 'procedural':
            paths = {name: T.pinned_file(pin, path.parent) for name, pin in data['sources'].items()}
            verified, full = A.procedural_candidates(paths, data['sources'], procedural_asset_root)
            if {R.digest(v['record']) for v in verified} != {R.digest(v) for v in rows.values()} or full != inventory:
                raise Error('full procedural source/assets/groundtruth reproof differs')
        else:
            verified = []
            for ident, record in rows.items():
                check_speech(A, record, admitted[ident])
                verified.append({'record': record, 'asset_path': record['media'][0]['path'],
                    'asset_sha256': record['media'][0]['sha256'], 'original_asset_sha256': record['provenance']['original_audio_sha256'],
                    'minimal_tail_derivation': record['provenance']['minimal_tail_derivation'],
                    'measure': admitted[ident]['measure'], 'teacher_evidence': admitted[ident]['teacher_evidence']})
        for candidate in verified:
            old = admitted[candidate['record']['id']]
            candidate.update(blockers=list(old['blockers']),
                original_source_admission=C.pin(path), original_recovery_group_id=old['recovery_group_id'])
            if old['train_admitted'] is not True and not candidate['blockers']:
                raise Error('prior excluded source cannot become eligible through composition')
        parents.append((label, path, data, source))
        candidate_sets.append(verified)
        inventories.append(inventory)
    for name in COMMON:
        if parents[0][2]['sources'][name]['sha256'] != parents[1][2]['sources'][name]['sha256']:
            raise Error('joint source global normalized/eval/prior/frozen identities differ: ' + name)
    candidates = [row for values in candidate_sets for row in values]
    inventory = [row for values in inventories for row in values]
    if len({row['record']['id'] for row in candidates}) != len(candidates) or len({row['id'] for row in inventory}) != len(inventory):
        raise Error('source composition refuses duplicate IDs across source campaigns')
    out.mkdir(parents=True)
    files = {}
    for name, records in (('source_ready', [c['record'] for c in candidates]), ('source_inventory', inventory)):
        path = out / (name + '.jsonl')
        path.write_bytes(b''.join(R.canonical(row) + b'\n' for row in records))
        files[name] = C.pin(path)
    specs = {name: parents[0][2]['sources'][name] for name in COMMON}
    specs.update(source=files['source_ready'], source_inventory=files['source_inventory'],
                 source_admission_helper={'path': str(Path(admission_helper).resolve()), 'sha256': helper_sha})
    for label, path, data, source in parents:
        specs[label + '_source_admission'] = C.pin(path)
        specs[label + '_source'] = C.pin(source)
        for name, pin in data['sources'].items():
            specs[label + '_' + name] = pin
    manifest = A.seal_admission(specs=specs, candidates=candidates, inventory=inventory, output=out / 'global_admission')
    counts = {route: sum(row['route'] == route for row in A.rows(out / 'global_admission/train_admitted.jsonl'))
              for route in ('audio_understand', 'speech_understand')}
    result = {'schema': 'chimera_recovery.joint_audio_source_preparation.v1', 'sealed': True,
        'sources': {label: C.pin(path) for label, path, _, _ in parents}, 'outputs': files,
        'source_candidates': len(candidates), 'full_original_inventory': len(inventory),
        'global_admission': C.pin(out / 'global_admission/manifest.json'), 'admission_seal': manifest['manifest_sha256'],
        'train_admitted': manifest['train_admitted'], 'admitted_by_route': counts,
        'prior_exclusions_preserved': True, 'cross_source_groups_recomputed': True, 'source_rows_unchanged': True,
        'adapter_sha256': R.sha(Path(__file__)), 'admission_helper_sha256': helper_sha,
        'native_train_ready': False, 'training_started': False}
    (out / 'preparation.json').write_bytes(R.canonical(result) + b'\n')
    return result


def compose_cache(source, admission_path, admission_sha, clean_manifests, out):
    """Carry immutable genuine proofs into a new source scope, never rebind them."""
    if out.exists():
        raise Error('clean proof composition needs a fresh output')
    _, admitted = C.admission({'path': str(admission_path), 'sha256': admission_sha}, Path('.'), source)
    rows, entries, origins = C.source_rows(source), {}, []
    for pin in clean_manifests:
        path = T.pinned_file(pin, Path('.'))
        data = T.read_json(path)
        if (data.get('schema') != 'chimera_recovery.clean_audio_cache.v1' or data.get('sealed') is not True or
                data.get('complete_source_traversal') is not True):
            raise Error('joint cache accepts only complete original genuine proof manifests')
        original_rows = C.source_rows(T.pinned_file(data['source'], path.parent))
        if len(data['entries']) != len(original_rows) or {v['id'] for v in data['entries']} != set(original_rows):
            raise Error('original clean proof inventory is not complete')
        for entry in data['entries']:
            ident = entry['id']
            if ident in entries or ident not in rows or R.digest(original_rows[ident]) != R.digest(rows[ident]) or entry['source_row_sha256'] != R.digest(rows[ident]):
                raise Error('joint cache duplicated/changed source record or original proof identity')
            proof_path, proof, tensors = C.verified_probe(entry['encoder_proof'], path.parent, rows[ident])
            C.load_latent(proof, tensors)
            entries[ident] = {**entry, 'encoder_proof': C.pin(proof_path), 'original_clean_manifest': C.pin(path),
                'original_encoder_adapter_sha256': data['encoder_adapter_sha256'],
                'original_runtime_adapter_sha256': proof['adapter_sha256']}
        origins.append(C.pin(path))
    if set(entries) != set(rows):
        raise Error('joint clean cache must independently cover every retained source row')
    out.mkdir(parents=True)
    result = {'schema': 'chimera_recovery.clean_audio_cache.v1', 'sealed': True, 'complete_source_traversal': True,
        'source': C.pin(source), 'source_admission': {'path': str(admission_path), 'sha256': admission_sha},
        'entries': [entries[ident] for ident in rows], 'original_clean_manifests': origins,
        'adapter_sha256': R.sha(HERE / 'audio_cache.py'), 'encoder_adapter_sha256': R.sha(HERE / 'h3_audio_probe.py'),
        'composition_adapter_sha256': R.sha(Path(__file__)), 'original_proofs_unchanged': True,
        'train_source_rows': sum(v['train_admitted'] for v in admitted.values()),
        'native_admission_granted': False, 'training_started': False}
    (out / 'manifest.json').write_bytes(R.canonical(result) + b'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare-source')
    for name in ('procedural-admission', 'speech-admission', 'admission-helper', 'procedural-asset-root', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--helper-sha256', required=True)
    p.add_argument('--procedural-sha256', required=True)
    p.add_argument('--speech-sha256', required=True)
    p = sub.add_parser('compose-cache')
    for name in ('source', 'admission', 'clean-manifests', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--admission-sha256', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare-source':
        result = prepare(args.procedural_admission, args.speech_admission, procedural_sha=args.procedural_sha256,
            speech_sha=args.speech_sha256, admission_helper=args.admission_helper,
            helper_sha=args.helper_sha256, procedural_asset_root=args.procedural_asset_root, out=args.out)
    else:
        result = compose_cache(args.source, args.admission, args.admission_sha256, T.read_json(args.clean_manifests), args.out)
    print(json.dumps({'sealed': result['sealed'], 'source_rows': result.get('source_candidates', len(result.get('entries', []))),
                      'train_admitted': result.get('train_admitted', result.get('train_source_rows')), 'training_started': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
