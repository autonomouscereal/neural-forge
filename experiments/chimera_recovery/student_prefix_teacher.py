"""Pinned singleton parent rescoring; preparation never supplies student outputs.

The original offline cache writer is unchanged. Its sessions/shards, rather than
only cache_config (which omits batching), prove the selected execution shape.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import uuid

from . import corpus as C, train_text as T

Error = T.ReadinessError
SCHEMA = 'chimera_recovery.student_prefix_teacher_plan.v1'
CAPTURE = 'chimera_recovery.student_prefix_rescoring.v1'
PROFILE = {'micro_batch': 1, 'max_batch_tokens': 0, 'batch_order': 'corpus',
           'head_chunk_rows': 256, 'head_precision': 'fp32', 'dtype': 'bfloat16',
           'attention_implementation': 'sdpa', 'top_k': 64, 'vocab_rows': 248320,
           'rows_per_shard': 256, 'host_resident': 'pageable',
           'sequence_policy': 'complete_real_length_no_crop_no_padding'}
sha, digest, canonical = C.file_sha256, C.digest, C.canonical


def pin(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def checked(record, aliases=None):
    path = Path(record['path'])
    if not path.is_file() and aliases and record.get('sha256') in aliases:
        path = Path(aliases[record['sha256']])
    if not path.is_file() or sha(path) != record.get('sha256'):
        raise Error('student-prefix teacher evidence changed: ' + str(path))
    return path


def read(record, aliases=None):
    return json.loads(checked(record, aliases).read_text())


def alias_data(path):
    if path is None:
        return None
    values = json.loads(Path(path).read_text())
    if (not isinstance(values, dict) or any(not isinstance(k, str) or len(k) != 64
            or any(c not in '0123456789abcdef' for c in k) or not isinstance(v, str)
            or not Path(v).is_absolute() or not Path(v).is_file() or sha(Path(v)) != k for k,v in values.items())):
        raise Error('source aliases must name existing unchanged SHA-pinned mounted files')
    return values


def publish(path, value):
    path = Path(path)
    if path.exists():
        raise Error('retain earlier immutable teacher evidence')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(canonical(value) + b'\n')
    return value


def prepare_plan(prompt_manifest, execution_policy, shape_proof, distribution_review,
                 paired_proof, native_parity, source_plan, portable_proof, writer, engine, output):
    paths = {key: pin(value) for key, value in locals().copy().items() if key != 'output'}
    return publish(output, derive_plan(paths))


def derive_plan(paths, aliases=None):
    required = {'prompt_manifest','execution_policy','shape_proof','distribution_review','paired_proof',
                'native_parity','source_plan','portable_proof','writer','engine'}
    if set(paths) != required:
        raise Error('teacher plan is missing original source/proof pins')
    for value in paths.values():
        checked(value, aliases)
    docs = {key: read(value, aliases) for key, value in paths.items() if key not in {'writer', 'engine'}}
    policy, shape, parity, source, portable = (docs[k] for k in
        ('execution_policy', 'shape_proof', 'native_parity', 'source_plan', 'portable_proof'))
    if (policy.get('schema') != 'chimera_recovery.text_teacher_execution_policy.v1'
            or policy.get('selected_for_preparation') is not True
            or policy.get('preserve_original_paired_gate') is not True
            or policy.get('claim_batch_shape_invariance') is not False
            or any(policy.get(key + '_sha256') != paths[name]['sha256'] for key, name in
                (('shape_proof','shape_proof'), ('distribution_review','distribution_review'),
                 ('paired_parity_proof','paired_proof'), ('source_plan','source_plan')))):
        raise Error('selected teacher policy and its actual proof pins differ')
    parent = shape.get('parent_integrity', {})
    runtime = source.get('runtime', {})
    versions = parity.get('runtime', {}).get('versions', {})
    if (shape.get('complete') is not True or parent.get('unchanged') is not True
            or parent.get('before', {}).get('mode') != 'full'
            or parent['before']['tree_sha256'] != parent.get('after', {}).get('tree_sha256')
            or parity.get('passed') is not True or docs['paired_proof'].get('passed') is not True
            or parity.get('top_k') != 64 or parity.get('vocab_rows') != 248320
            or parity.get('tool', {}).get('sha256') != paths['writer']['sha256']
            or parity.get('engine', {}).get('sha256') != paths['engine']['sha256']
            or portable.get('plan_sha256') != paths['source_plan']['sha256']
            or portable.get('actual_writer_engine_imported') is not True
            or portable.get('python') != runtime.get('python')
            or parity.get('runtime', {}).get('python') != runtime.get('python')
            or any(versions.get(k) != v for k, v in runtime.get('packages', {}).items())
            or runtime.get('teacher_sha256') != paths['writer']['sha256']):
        raise Error('original parent/writer/K64 or reference runtime proof differs')
    for item in (runtime.get('compiler_image_proof'), runtime.get('actual_fla_kernel_proof')):
        checked(item, aliases)
    files = {f['path']: f['sha256'] for f in parent['before']['files']}
    manifest = docs['prompt_manifest']
    if (manifest.get('schema') != 'chimera_recovery.onpolicy_prompts.v1'
            or manifest.get('train_only') is not True or manifest.get('actual_sampled_tokens') != 0):
        raise Error('pinned preparatory train-only prompt plan required')
    image = runtime.get('image_id', '')
    if (not image.startswith('sha256:') or len(image) != 71
            or any(c not in '0123456789abcdef' for c in image[7:]) or not versions
            or runtime.get('parent_config_sha256') != files['config.json']):
        raise Error('immutable reference image and package profile required')
    result = {'schema': SCHEMA, 'date': '2026-10-13', 'sources': paths, 'profile': PROFILE,
        'helper_sha256': sha(Path(__file__)),
        'teacher_sha256': parent['before']['tree_sha256'], 'native_tokenizer_sha256': files['tokenizer.json'],
        'native_template_sha256': files['chat_template.jinja'], 'native_config_sha256': files['config.json'],
        'runtime_profile': {'image_id': image, 'python': runtime['python'], 'versions': versions,
                           'linear_attention_kernel': parity['linear_attention_kernel'],
                           'read_only_runtime_mount': runtime['venv_container']},
        'writer_argv_template': [runtime['python'], runtime['teacher_path'], 'build-text',
            '--model-dir','/models/parent','--teacher','q38-student-prefix-oct13-singleton-v1',
            '--corpus','PINNED_PACKED_STUDENT_CORPUS','--out-root','NEW_ALLOCATED_CACHE_ROOT',
            '--device','cuda:0','--dtype','bfloat16','--attn-implementation','sdpa',
            '--linear-attn-kernel',parity['linear_attention_kernel']['requested'],
            '--top-k','64','--vocab-rows','248320','--micro-batch','1',
            '--max-batch-tokens','0','--batch-order','corpus','--rows-per-shard','256',
            '--head-precision','fp32','--head-chunk-rows','256','--teacher-digest','full',
            '--digest-threads','4','--host-resident','pageable','--ids-key','input_ids',
            '--lengths-key','real_lengths','--verify-existing'],
        'argv_is_preparation_template_not_launch': True,
        'prompt_rows': manifest['rows'], 'maximum_sampled_tokens': manifest['maximum_sampled_tokens'],
        'actual_sampled_tokens': 0, 'offline_paired_profile_changed': False,
        'production_runtime_receipt': None, 'student_generation_completed': False,
        'parent_rescoring_completed': False, 'production_training_ready': False,
        'remaining': ['Accepted complete D2 student and native captured traces',
                      'Actual source-bound singleton forward/capacity and allocated runtime proof',
                      'Complete fresh rescoring cache and per-session/shard singleton checks',
                      'Selected round amendment and matched full-answer evaluation']}
    return result


def validate_plan(path, aliases=None):
    plan = json.loads(Path(path).read_text())
    if plan.get('schema') != SCHEMA or plan.get('profile') != PROFILE:
        raise Error('singleton complete-sequence teacher profile differs')
    if derive_plan(plan['sources'], aliases) != plan:
        raise Error('teacher plan was not derived from its exact selected source/proofs')
    return plan


def verify_records(plan, manifest, records, sessions, row_lengths, row_counts):
    """Pure metadata checks, repeated against actual tensor lengths before use."""
    p, runtime = plan['profile'], plan['runtime_profile']
    config = manifest.get('cache_config', {})
    teacher = manifest.get('teacher', {})
    if (manifest.get('schema') != 'omni-distill-teacher-cache-manifest-v1'
            or manifest.get('all_finite') is not True or manifest.get('all_structural_ok') is not True
            or teacher.get('equal_before_and_after_every_session') is not True
            or teacher.get('tree_sha256_before') != plan['teacher_sha256']
            or teacher.get('tree_sha256_after') != plan['teacher_sha256']
            or manifest.get('tool', {}).get('sha256') != plan['sources']['writer']['sha256']
            or manifest.get('engine', {}).get('sha256') != plan['sources']['engine']['sha256']
            or any(config.get(k) != p[k] for k in ('dtype','head_precision','attention_implementation','top_k','vocab_rows','rows_per_shard'))
            or config.get('linear_attention_kernel') != runtime['linear_attention_kernel']
            or config.get('corpus_read') != {'ids_key': 'input_ids', 'lengths_key': 'real_lengths',
                'mask_source': 'target_mask shifted by one inside the tool (stored t iff target t+1)'}) :
        raise Error('original singleton teacher/cache precision, source or mask contract differs')
    if not records or len(row_lengths) != len(row_counts) or len(row_lengths) != config.get('corpus_rows'):
        raise Error('complete native student inventory is missing')
    seen = set()
    for record in records:
        start, stop = record['rows']
        table = record.get('row_table', [])
        sid = record.get('session')
        if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(row_lengths)
                or set(range(start, stop)) & seen or len(table) != stop-start
                or record.get('batches') != stop-start
                or any(record.get(k) != p[k] for k in ('micro_batch','max_batch_tokens','batch_order','head_chunk_rows'))
                or record.get('forward_tokens') != sum(row_lengths[start:stop])
                or record.get('padded_tokens') != sum(row_lengths[start:stop])):
            raise Error('teacher shard grouped, padded, cropped or repeated a source sequence')
        seen.update(range(start, stop))
        if sid not in sessions:
            raise Error('teacher shard has no sealed actual session')
        session = sessions[sid]
        if (session.get('batching') != {'micro_batch': 1, 'max_batch_tokens': 0, 'order': 'corpus'}
                or session.get('runtime', {}).get('python') != runtime['python']
                or session.get('runtime', {}).get('versions') != runtime['versions']
                or session.get('tool', {}).get('sha256') != plan['sources']['writer']['sha256']
                or session.get('engine', {}).get('sha256') != plan['sources']['engine']['sha256']
                or session.get('head_matmul') != {'cuda_matmul_allow_tf32': False,
                    'float32_matmul_precision': 'highest', 'head_precision': 'fp32'}
                or session.get('host_resident', {}).get('mode') != p['host_resident']
                or session.get('teacher_unchanged_in_session') is not True
                or any(session.get(k, {}).get('tree_sha256') != plan['teacher_sha256'] for k in
                       ('teacher_digest_before','teacher_digest_after'))):
            raise Error('actual singleton teacher session/runtime/source differs')
        expected = [(i, row_lengths[i], row_counts[i]) for i in range(start, stop)]
        if [(r.get('row'),r.get('length'),r.get('count')) for r in table] != expected:
            raise Error('teacher row table differs from complete source trace lengths/mask')
    if seen != set(range(len(row_lengths))):
        raise Error('teacher cache does not cover every student sequence exactly once')
    if (manifest.get('forward_tokens_total') != sum(row_lengths)
            or manifest.get('positions_total') != sum(row_counts)):
        raise Error('teacher aggregate denominator/real tokens differ')
    return {'rows': len(seen), 'real_tokens': sum(row_lengths), 'supervised_positions': sum(row_counts)}


def verify_capture(teacher_plan, corpus_manifest, cache, runtime_receipt, source_aliases=None):
    """Authenticate real recorded metadata, original raw verifier and source tensors.

    Runtime receipt is produced from an actual allocated run, not by this helper.
    It binds each actual session to its external container/runtime proof.
    """
    aliases = alias_data(source_aliases)
    plan = validate_plan(teacher_plan, aliases)
    cache, corpus_manifest = Path(cache).resolve(), Path(corpus_manifest).resolve()
    corpus = json.loads(corpus_manifest.read_text())
    manifest = json.loads((cache/'manifest.json').read_text())
    identity = corpus.get('identity', {})
    if (corpus.get('schema') != 'chimera_recovery.student_prefix_corpus.v1'
            or corpus.get('source_prompt_manifest', {}).get('sha256') != plan['sources']['prompt_manifest']['sha256']
            or identity.get('tokenizer_sha256') != plan['native_tokenizer_sha256']
            or identity.get('template_sha256') != plan['native_template_sha256']
            or manifest.get('corpus', {}).get('sha256') != corpus['corpus']['sha256']):
        raise Error('rescoring changed native student traces/template/corpus')
    for key in ('corpus','generation_receipt','traces','source_prompts','source_prompt_manifest'):
        checked(corpus[key], aliases)
    runtime_path = Path(runtime_receipt)
    observed = json.loads(runtime_path.read_text())
    if (observed.get('schema') != 'chimera_recovery.student_prefix_teacher_runtime.v1'
            or observed.get('passed') is not True or observed.get('teacher_plan_sha256') != sha(Path(teacher_plan))
            or observed.get('cache_manifest_sha256') != sha(cache/'manifest.json')
            or observed.get('source_aliases') != (pin(source_aliases) if source_aliases else None)
            or observed.get('runtime_profile') != plan['runtime_profile']):
        raise Error('actual allocated singleton runtime/profile evidence is missing or different')
    records, sessions, pins = [], {}, []
    for entry in manifest.get('shards', []):
        name = entry['shard']
        if Path(name).name != name:
            raise Error('unsafe teacher shard name')
        rpin = {'path': str(cache/name/'shard.json'), 'sha256': entry['shard_json_sha256']}
        record = read(rpin); records.append(record); pins.append(rpin)
        sid = record['session']
        if sid not in sessions:
            matches = [s for s in manifest.get('sessions', []) if s['session'] == sid]
            if len(matches) != 1 or Path(sid).name != sid:
                raise Error('complete actual teacher session is missing/ambiguous')
            spin = {'path': str(cache/'sessions'/('session-'+sid+'.json')), 'sha256': matches[0]['record_sha256']}
            sessions[sid] = read(spin); pins.append(spin)
            external = observed.get('sessions', {}).get(sid, {})
            if external.get('cache_session_sha256') != spin['sha256']:
                raise Error('actual runtime proof references another teacher session')
            proof = read(external['allocation_forward_proof'], aliases)
            try:
                allocation = str(uuid.UUID(proof.get('allocation_id', '')))
                gpu = 'GPU-' + str(uuid.UUID(proof.get('gpu_uuid', '').removeprefix('GPU-')))
            except (ValueError, AttributeError, TypeError):
                raise Error('actual allocated singleton forward identity is malformed') from None
            container = proof.get('container_id', '')
            if (proof.get('passed') is not True or proof.get('native_singleton_forward_verified') is not True
                    or proof.get('cache_session_sha256') != spin['sha256']
                    or proof.get('runtime_profile') != plan['runtime_profile']
                    or allocation != proof['allocation_id'] or gpu != proof['gpu_uuid']
                    or len(container) != 64 or any(c not in '0123456789abcdef' for c in container)
                    or gpu not in {d.get('uuid') for d in sessions[sid].get('runtime', {}).get('devices', [])}):
                raise Error('actual allocated singleton forward proof is incomplete')
            pins.append(external['allocation_forward_proof'])
    import torch
    from safetensors.torch import load_file
    native = load_file(str(checked(corpus['corpus'], aliases)))
    ids, lengths, mask = (native[k] for k in ('input_ids','real_lengths','assistant_mask'))
    if (ids.ndim != 2 or ids.dtype not in (torch.int32,torch.int64)
            or lengths.shape != (ids.shape[0],) or lengths.dtype not in (torch.int32,torch.int64) or mask.dtype != torch.bool
            or mask.shape != ids.shape or 'target_mask' not in native or not torch.equal(native['target_mask'], mask)):
        raise Error('student corpus needs identical target/assistant masks and exact real lengths')
    traces = list(C.iter_jsonl(checked(corpus['traces'], aliases)))
    if len(traces) != len(lengths) or corpus['row_ids'] != [r['id'] for r in traces]:
        raise Error('complete original source traces are missing')
    for i, trace in enumerate(traces):
        tokens = trace['prompt_ids'] + trace['generated_ids']
        n = int(lengths[i])
        expected_mask = [False]*len(trace['prompt_ids']) + [True]*len(trace['generated_ids'])
        if (n != len(tokens) or not 1 <= n <= ids.shape[1] or ids[i,:n].tolist() != tokens
                or mask[i,:n].tolist() != expected_mask or bool(mask[i,n:].any())):
            raise Error('native corpus cropped/changed actual student sequence or assistant ownership')
    counts = [int(mask[i,1:int(lengths[i])].sum()) for i in range(len(lengths))]
    result = verify_records(plan, manifest, records, sessions, lengths.tolist(), counts)
    spec = importlib.util.spec_from_file_location('student_prefix_original_qtext_verifier', checked(plan['sources']['writer'], aliases))
    writer = importlib.util.module_from_spec(spec); spec.loader.exec_module(writer)
    for entry in manifest['shards']:
        if writer.verify_shard(cache/entry['shard']).get('ok') is not True:
            raise Error('original raw Q64 shard verifier rejected actual capture')
        raw = writer.read_shard(cache/entry['shard'])
        for row in raw['record']['row_table']:
            i, offset, count = row['row'], row['offset'], row['count']
            selected = mask[i,1:int(lengths[i])].nonzero().flatten()
            if (raw['pos'][offset:offset+count].tolist() != selected.tolist()
                    or raw['ids'][offset:offset+count].tolist() != ids[i,selected].tolist()):
                raise Error('raw teacher stored IDs/positions differ from the actual student prefix')
    return {'schema': CAPTURE, 'passed': True, 'teacher_plan': pin(teacher_plan),
            'corpus_manifest': pin(corpus_manifest), 'cache_manifest': pin(cache/'manifest.json'),
            'runtime_receipt': pin(runtime_receipt), 'execution_profile': plan['profile'],
            'source_aliases': pin(source_aliases) if source_aliases else None,
            'metadata_pins': pins, **result, 'production_training_ready': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest='mode', required=True)
    plan = sub.add_parser('prepare-plan')
    for key in ('prompt-manifest','execution-policy','shape-proof','distribution-review','paired-proof',
                'native-parity','source-plan','portable-proof','writer','engine','output'):
        plan.add_argument('--'+key, type=Path, required=True)
    capture = sub.add_parser('seal-capture')
    for key in ('teacher-plan','corpus-manifest','cache','runtime-receipt','output'):
        capture.add_argument('--'+key, type=Path, required=True)
    capture.add_argument('--source-aliases', type=Path)
    args = vars(parser.parse_args()); mode = args.pop('mode')
    if mode == 'prepare-plan':
        result = prepare_plan(**args)
    else:
        output = args.pop('output'); result = publish(output, verify_capture(**args))
    print(json.dumps({key: result[key] for key in ('schema','production_training_ready')}))


if __name__ == '__main__':
    main()
