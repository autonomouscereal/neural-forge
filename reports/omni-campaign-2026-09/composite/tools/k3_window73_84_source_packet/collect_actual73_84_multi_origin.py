"""Host metadata-only catalog/stat collector for the separate-output73--84 run.

Adapted from the reviewed collect_ro_nas_observation_v2 route. No connection,
Docker, allocation, tensor reader, payload copy/hash, or current-model mutation.
Authentic60 seal/metadata must exist. Payload SHAs are authenticated producer/
scanner expectations; each production resolver performs its own first-use hash.
Only exclusive metadata in the new own production_control directory is written.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import ast
from types import SimpleNamespace

RUN = '__UNRESOLVED_NEW73_84_UUID__'
RUN49 = '03603908-570d-4c90-96cb-ee3a63a9a139'
PREVIOUS_RUN = '__UNRESOLVED_CLOSED61_72_UUID__'
OLDER_RUN = '764524b1-96a9-4636-be74-942f9c54754c'
LEGACY_CATALOG_SHA = '7a6b254fd7822bb2b868be149f2246b892868a7b2bbb0daa5f37db5ae1fe5cd2'
PARENT_ROWS_SHA = 'c5d06a6f17377abdb50e948a830861f0b1539f83f4ca404c64d4a1166f3c7eb8'
CLOSED36_SHA = 'dd0ab8e821307e352096cdded457934dfd6df3434cfbe40d74e24b09a215870b'
COMPLETED_NATIVE_SHA = '720465874722e1548c08f82a019be7dcb477efcc5828d6edd10610c1ea076243'
PREVIOUS_NATIVE_SHA = '9086d7c0a5ec2bc7cfd1b823aecdc12e2973b3c81b064776ad3670df9d6b7b69'
PREVIOUS_PACKET_SHA = '5b9559b8ac20428de7e4ac09201f71839f4d03113a985715ca81d3aedd01246c'
SOURCES = {'k3_dense_pass.py': '21e322039fa65334a49e222997f92a76e640cf68d75e5388244b1e7719baae56',
           'ro_resume_paths.py': '146665bbce576e52462d73384d96fafd7ec07a07a2f7a9f95bd52c9655f84ada',
           'ro_resume_runtime.py': '288bf5f3cde564b70932097d28a049fc1c4be500b83d7a43e33cb1273fa5e260',
           'k3_recovery_ancestry.py': '88e118847efb589d9cc48fc578996c3ca48cd0a0b9a4b491aab573fed54a9d41'}
VERIFIER_SHA = '51a178d37c142f646aacd6c726cfd2c8868d2ea19a2b632d578fd1a504257901'
ALIASES = {'legacy_closed36': '/resume-prefix', 'closed37_48': '/resume-window37-48', 'closed49_60': '/resume-window49-60', 'closed61_72': '/resume-window61-72'}
HOST_ORIGINS = {'legacy_closed36': '/mnt/lexar/omni-campaign/k3-diagnostic-recovery-oct07-048cff57-935c-4e2b-a31b-96a3c7fa6318',
                'closed37_48': '/mnt/ai_data/omni-campaign/k3-nas-window37-48-oct07-'+OLDER_RUN+'/model',
                'closed49_60': '/mnt/ai_data/omni-campaign/k3-nas-window49-60-oct07-'+RUN49+'/model',
                'closed61_72': '__UNRESOLVED_CLOSED61_72_ROOT__'}
SEALED60_REL = 'ancestry/window-49-60-' + RUN49
SEAL_REL = 'ancestry/window-61-72-' + PREVIOUS_RUN
SEALED48_REL = 'ancestry/window-37-48-' + OLDER_RUN
SEALED36_REL = 'ancestry/window-25-36-b873d51f-b251-4944-aa84-5d91df96e87b'
STAT_FIELDS = ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns', 'mode')


def require(value, message):
    if value is not True: raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def identity(path):
    s = Path(path).lstat()
    return dict(dev=s.st_dev, ino=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, mode=s.st_mode)


def no_symlinks(path, *, directory=False):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts, 'canonical absolute physical path required')
    for current in [*reversed(p.parents), p]:
        s = current.lstat()
        require(not stat.S_ISLNK(s.st_mode), 'symlink path refused: '+str(current))
        require(stat.S_ISDIR(s.st_mode) if current != p or directory else stat.S_ISREG(s.st_mode),
                'wrong physical path type: '+str(current))
    return p


def whole_sha(path):
    p = no_symlinks(path); before = identity(p)
    with p.open('rb') as handle: digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    require(identity(p) == before, 'source changed during whole metadata/code hash')
    return digest


def pinned_doc(path, expected):
    require(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected) is not None,
            'unresolved full metadata source SHA')
    p = no_symlinks(path); before = identity(p)
    require(before['size'] <= 256 << 20, 'bounded native metadata document required')
    raw = p.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected and identity(p) == before,
            'actual full pinned metadata bytes changed: '+str(p))
    return json.loads(raw, object_pairs_hook=unique), raw


def logical_row(row):
    logical = row.get('recorded_logical_path') or '/run-output/'+row['relative_path']
    p = PurePosixPath(logical)
    require(p.is_relative_to('/run-output') and str(p) == logical and '..' not in p.parts,
            'canonical actual logical source path required')
    return logical, p.relative_to('/run-output')


def add_row(rows, logical, role, expected_sha, expected_bytes, *, origin='closed61_72'):
    rel = PurePosixPath(logical).relative_to('/run-output')
    require(isinstance(expected_sha, str) and re.fullmatch(r'[0-9a-f]{64}', expected_sha) is not None
            and '..' not in rel.parts and type(expected_bytes) is int and expected_bytes >= 0,
            'actual complete source SHA/byte extent required')
    physical = no_symlinks(Path(HOST_ORIGINS[origin])/rel)
    require(identity(physical)['size'] == expected_bytes, 'authenticated actual source extent changed')
    require(not any(logical_row(row)[0] == logical for row in rows), 'duplicate actual logical source row')
    rows.append({'recorded_logical_path': logical, 'role': role, 'sha256': expected_sha,
                 'bytes': expected_bytes, 'source_origin': origin,
                 'prospective_RO_input_path': ALIASES[origin]+'/'+str(rel),
                 'unchanged_host_input_path': str(physical)})


def exact_write_paths():
    native = []
    for layer in range(73, 85):
        native.extend((f'/run-output/fold/dense/L{layer:02d}/L{layer:02d}_K17408_formE.safetensors',
                       f'/run-output/fold/dense/L{layer:02d}/ffn.safetensors',
                       f'/run-output/fold/dense/L{layer:02d}/done.json',
                       f'/run-output/body/L{layer:02d}/body.safetensors',
                       f'/run-output/fold/resid/L{layer:02d}/held_rows.safetensors'))
    native.extend(f'/run-output/fold/snapshots/before_L{layer:02d}{suffix}.safetensors'
                  for layer in (84, 85) for suffix in ('', '_folded'))
    leaves = ['fold/snapshots/snapshots_manifest.json', 'receipts/k3_dense_pass_receipt.json',
              'receipts/k3_dense_pass_receipt.partial.json', 'receipts/k3_dense_pass_receipt.sha256']
    native.extend('/run-output/'+leaf for leaf in leaves)
    seal = '/run-output/ancestry/window-73-84-'+RUN
    copied = leaves + ['fold/snapshots/before_L85.safetensors', 'fold/snapshots/before_L85_folded.safetensors']
    verifier = [seal+'/'+leaf for leaf in copied]
    verifier.extend(seal+'/inputs/'+label+'.json' for label in ('original_record_plan', 'original_accepted_receipt',
                      'native_runtime_observation', 'separate_copy_receipt', 'full_ffn_validation', 'previous_window_continuity'))
    verifier.extend((seal+'/ffn_request.json', seal+'/window_continuity.json'))
    result = sorted(native+verifier)
    require(len(result) == len(set(result)) == 82, 'exact new73-84 finite write roster differs')
    return result


def exclusive_json(path, doc):
    p = Path(path); no_symlinks(p.parent, directory=True)
    require(not p.exists() and not p.is_symlink(), 'existing metadata must remain unchanged')
    raw = (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    with p.open('xb') as handle: handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    require(p.read_bytes() == raw, 'exclusive metadata publication full byte readback differs')
    return {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def collect(args):
    require(re.fullmatch(r'[0-9a-f-]{36}',RUN) is not None,'actual new73-84 UUID must be bound before collection')
    require(os.name == 'posix', 'actual Linux host observation required')
    i, _ = pinned_doc(args.identity, args.identity_sha256)
    parent, parent_raw = pinned_doc(args.parent_catalog, i['parent_catalog_sha256'])
    require(i['run_uuid'] == RUN and i['new_layers'] == [73,84] and i['only_this_window_scope'] is True
            and i['readonly_resume_source_roots'] == HOST_ORIGINS and i['resume_source_mounted_read_only'] is True
            and i['native_working_copy_mounted_read_only'] is False and i['actual_runtime_admitted'] is False
            and i['native_binding']['harness_id'] == '01a11108-009c-72f3-b2ba-e0d48fca1435'
            and type(i['native_binding']['session_generation']) is int and i['native_binding']['session_generation'] >= 6,
            'genuine source-bound separate61-72 identity required')
    prefix = 'k3-nas-window73-84-oct07-'+RUN
    storage = {'NAS': '/mnt/ai_data', 'Lexar': '/mnt/lexar'}[i['new_output_filesystem']]
    require(i['runtime_root'] == '/home/cereal/omni-campaign/runtime/'+prefix
            and i['native_working_output_root'] == storage+'/omni-campaign/'+prefix+'/model'
            and i['controller_output_root'] == i['output_root'] == '/home/cereal/omni-campaign/outputs/'+prefix+'/control-output',
            'canonical exact new runtime/new-output/controller roots required')
    runtime = no_symlinks(i['runtime_root'], directory=True)
    control = no_symlinks(runtime/'production_control', directory=True)
    code = no_symlinks(runtime/'production_code', directory=True)
    native_dir = code/'reports/omni-campaign-2026-09/composite/tools'
    for name, expected in SOURCES.items():
        require(whole_sha(native_dir/name) == expected, 'actual selected executing source differs: '+name)
        if name != 'k3_dense_pass.py': require(whole_sha(control/name) == expected, 'actual imported control source differs')
    require(whole_sha(control/'verify_later_window.py') == VERIFIER_SHA, 'actual verifier source differs')
    for name, expected in [('recovery_ancestry_capacity_wait.json', PREVIOUS_PACKET_SHA),
                           ('k3_dense_pass_9086.py', PREVIOUS_NATIVE_SHA), ('k3_dense_pass_720.py', COMPLETED_NATIVE_SHA),
                           ('k3_dense_pass_5e.py', '5e852b63d28ba41599ca3e994851b3e72103f213384de77b9c172c7823036007'),
                           ('k3_dense_pass_61.py', '078187a4cfa2f5f7810f0c0a8fa85f7fcc8304085684d45dfda5b895854881d0')]:
        require(whole_sha(control/'source_revisions'/name) == expected, 'actual preserved producer/source ancestry differs')
    mounts, _ = pinned_doc(args.mounts, args.mounts_sha256)
    expected_mounts = {'/code/neural-forge': (str(code), True), '/control': (str(control), True),
        '/inputs/corpus': ('/home/cereal/omni-campaign/composite/corpus', True),
        '/inputs/digest-receipts': ('/home/cereal/omni-campaign/runtime/k3-fold-sep07-e2e615be/inputs/receipts', True),
        '/parent': ('/mnt/lexar/models/moonshotai/Kimi-K3', True), '/recovery-source': (i['historical_output_root'], True),
        '/resume-prefix': (HOST_ORIGINS['legacy_closed36'], True), '/resume-window37-48': (HOST_ORIGINS['closed37_48'], True),
        '/resume-window49-60': (HOST_ORIGINS['closed49_60'], True),
        '/resume-window61-72': (HOST_ORIGINS['closed61_72'], True),
        '/run-output': (i['native_working_output_root'], False), '/run-control-output': (i['controller_output_root'], False)}
    require(len(mounts) == 12 and [r['target'] for r in mounts] == sorted(expected_mounts)
            and {r['target']: (r['source'], r['read_only']) for r in mounts} == expected_mounts
            and all(set(r) == {'source', 'target', 'read_only'} for r in mounts), 'exact sorted eleven planned mount rows required')
    targets = [PurePosixPath(row['target']) for row in mounts]
    require(all(a == b or not a.is_relative_to(b) for a in targets for b in targets), 'nested target mounts refused')
    roots = {role: no_symlinks(path, directory=True) for role, path in HOST_ORIGINS.items()}
    output = no_symlinks(i['native_working_output_root'], directory=True)
    meta = no_symlinks(i['controller_output_root'], directory=True)
    root_stats = {role: identity(path) for role, path in roots.items()}; output_stat = identity(output)
    all_roots = list(roots.values())+[output]
    for index, left in enumerate(all_roots):
        for right in all_roots[index+1:]:
            require(left != right and not left.is_relative_to(right) and not right.is_relative_to(left)
                    and (identity(left)['dev'],identity(left)['ino']) != (identity(right)['dev'],identity(right)['ino']),
                    'source/output roots must be exact disjoint directories; sharing NASdevice is allowed')
    require(not any((output/leaf).exists() or (output/leaf).is_symlink() for leaf in ('fold','body','receipts','ancestry')),
            'unique fresh new73-84 model namespace required')
    require(args.output_directory == control and args.identity == control/'identity.json', 'exact new own control publication leaf required')
    parent_rows = parent['input_catalog']
    require(len(parent_rows) > 127 and parent['legacy_catalog_sha256'] == LEGACY_CATALOG_SHA,
            'authentic actual49 source catalog through48 required')
    legacy = [{key:value for key,value in row.items() if key != 'source_origin'} for row in parent_rows[:127]]
    require(hashlib.sha256(json.dumps(legacy,sort_keys=True,separators=(',',':')).encode()).hexdigest() == PARENT_ROWS_SHA
            and all(row['source_origin'] == 'legacy_closed36' for row in parent_rows[:127])
            and all(row['source_origin'] in ('closed37_48','closed49_60') for row in parent_rows[127:]),
            'unchanged original127 plus authentic37-48 parent origins required')
    rows = copy.deepcopy(parent_rows)
    parent_rows_sha = hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    source = roots['closed61_72']; seal = source/SEAL_REL
    closed, closed_raw = pinned_doc(seal/'window_continuity.json', i['closed72_continuity_sha256'])
    require(closed['schema'] == 'neural_forge.k3_actual_window_continuity.v1'
            and closed['completed_window'] == '61-72' and closed['full_prefix_layers'] == 72
            and closed['ffn_full_scan_complete'] is True, 'authentic full sealed48 required; no marker/partial substitute')
    terminal, _ = pinned_doc(seal/'receipts/k3_dense_pass_receipt.json', closed['receipt_sha256'])
    scanner, _ = pinned_doc(seal/'inputs/full_ffn_validation.json', closed['ffn_validation_sha256'])
    manifest, _ = pinned_doc(seal/'fold/snapshots/snapshots_manifest.json', closed['snapshot_manifest_sha256'])
    for layer in range(61,73):
        prefix_row = closed['prefix'][layer-1]
        require(prefix_row['layer'] == layer, 'ordered complete48 prefix required')
        for name, key, role in [('ffn.safetensors','ffn_sha256','closed_prefix_ffn.safetensors'),
                                 ('done.json','done_sha256','closed_prefix_done.json')]:
            logical = f'/run-output/fold/dense/L{layer:02d}/'+name
            add_row(rows, logical, role, prefix_row[key], identity(source/PurePosixPath(logical).relative_to('/run-output'))['size'])
        entry = terminal['layers'][str(layer)]; body = entry['body']; held = entry['held_rows']
        b = next(row for row in closed['body'] if row['layer'] == layer)
        require(b['byte_exact'] is True and body['byte_exact'] is True and body['sha256'] == b['sha256'], 'actual body producer/continuity byte proof differs')
        require(body['path'] == f'/run-output/body/L{layer:02d}/body.safetensors'
                and held['path'] == f'/run-output/fold/resid/L{layer:02d}/held_rows.safetensors', 'exact actual native body/held path required')
        add_row(rows, body['path'], 'closed_prefix_verbatim_body', body['sha256'], body['bytes'])
        add_row(rows, held['path'], 'closed_prefix_retained_held_rows', held['sha256'], held['bytes'])
    for suffix in ('', '_folded'):
        logical = '/run-output/fold/snapshots/before_L73'+suffix+'.safetensors'
        add_row(rows, logical, 'native_before73_dual_boundary', manifest['files'][logical]['sha256'],
                identity(source/PurePosixPath(logical).relative_to('/run-output'))['size'])
    for row in closed['retained_ancestry']:
        require(PurePosixPath(row['retained']).is_relative_to('/run-output/'+SEAL_REL), 'exact actual sealed37-48 retained ancestry required')
        add_row(rows, row['retained'], 'sealed61_72_ancestry', row['sha256'], row['bytes'])
    add_row(rows, '/run-output/'+SEAL_REL+'/ffn_request.json', 'sealed61_72_ancestry', closed['ffn_request_sha256'], identity(seal/'ffn_request.json')['size'])
    add_row(rows, '/run-output/'+SEAL_REL+'/window_continuity.json', 'sealed61_72_ancestry', i['closed72_continuity_sha256'], len(closed_raw))
    old = {logical_row(row)[0]:row for row in rows}
    runtime_raw = (control/'ro_resume_runtime.py').read_bytes(); tree = ast.parse(runtime_raw)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('context_equal', 'validate_closed48_metadata', 'validate_closed60_metadata', 'validate_closed72_metadata')]
    env = {'json':json,'require':require,'CLOSED36':CLOSED36_SHA}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<selected exact pure prior window validators>','exec'),env)
    closed36,_ = pinned_doc(roots['legacy_closed36']/SEALED36_REL/'window_continuity.json',CLOSED36_SHA)
    expected_context = copy.deepcopy(closed36['resume_context'])
    sealed48 = roots['closed37_48']/SEALED48_REL
    closed48,_ = pinned_doc(sealed48/'window_continuity.json',i['closed48_continuity_sha256'])
    scanner48,_ = pinned_doc(sealed48/'inputs/full_ffn_validation.json',closed48['ffn_validation_sha256'])
    terminal48,_ = pinned_doc(sealed48/'receipts/k3_dense_pass_receipt.json',closed48['receipt_sha256'])
    manifest48,_ = pinned_doc(sealed48/'fold/snapshots/snapshots_manifest.json',closed48['snapshot_manifest_sha256'])
    expected_context['software']['tool_sha256'] = '720465874722e1548c08f82a019be7dcb477efcc5828d6edd10610c1ea076243'
    env['validate_closed48_metadata'](closed48,scanner48,terminal48,manifest48,SimpleNamespace(old=old),expected_context)
    sealed60 = roots['closed49_60']/SEALED60_REL
    closed60,_ = pinned_doc(sealed60/'window_continuity.json',i['closed60_continuity_sha256'])
    scanner60,_ = pinned_doc(sealed60/'inputs/full_ffn_validation.json',closed60['ffn_validation_sha256'])
    terminal60,_ = pinned_doc(sealed60/'receipts/k3_dense_pass_receipt.json',closed60['receipt_sha256'])
    manifest60,_ = pinned_doc(sealed60/'fold/snapshots/snapshots_manifest.json',closed60['snapshot_manifest_sha256'])
    expected_context['software']['tool_sha256'] = '5e852b63d28ba41599ca3e994851b3e72103f213384de77b9c172c7823036007'
    env['validate_closed60_metadata'](closed60,scanner60,terminal60,manifest60,SimpleNamespace(old=old),expected_context,i['closed48_continuity_sha256'])
    sealed72 = roots['closed61_72']/SEAL_REL
    closed72,_ = pinned_doc(sealed72/'window_continuity.json',i['closed72_continuity_sha256'])
    scanner72,_ = pinned_doc(sealed72/'inputs/full_ffn_validation.json',closed72['ffn_validation_sha256'])
    terminal72,_ = pinned_doc(sealed72/'receipts/k3_dense_pass_receipt.json',closed72['receipt_sha256'])
    manifest72,_ = pinned_doc(sealed72/'fold/snapshots/snapshots_manifest.json',closed72['snapshot_manifest_sha256'])
    expected_context['software']['tool_sha256'] = '078187a4cfa2f5f7810f0c0a8fa85f7fcc8304085684d45dfda5b895854881d0'
    env['validate_closed72_metadata'](closed72,scanner72,terminal72,manifest72,SimpleNamespace(old=old),expected_context,i['closed60_continuity_sha256'])
    at = datetime.now(timezone.utc).isoformat(); source_files = {}; fresh_metadata = {}
    for row in rows:
        logical, rel = logical_row(row); origin = row['source_origin']; physical = no_symlinks(roots[origin]/rel)
        got = identity(physical)
        require(got['dev'] == root_stats[origin]['dev'] and row['unchanged_host_input_path'] == str(physical)
                and row['prospective_RO_input_path'] == ALIASES[origin]+'/'+str(rel)
                and ('bytes' not in row or row['bytes'] == got['size']), 'actual origin/path/extent differs')
        fact = {'path':ALIASES[origin]+'/'+str(rel),'stat':got,'source_origin':origin,'source_sha256':row['sha256'],
                'source_sha256_scope':'authenticated sealed native/scanner/catalog expectation; production resolver hashes payload on first consumption'}
        if physical.suffix in ('.json','.sha256'):
            require(got['size'] <= 256 << 20, 'bounded authentic native metadata required')
            digest = whole_sha(physical)
            require(digest == row['sha256'] and identity(physical) == got, 'fresh actual metadata hash differs')
            fact['fresh_metadata_full_sha256'] = digest; fresh_metadata[logical] = {'sha256':digest,'bytes':got['size']}
        source_files[logical] = fact
    for logical, fact in source_files.items():
        require(identity(roots[fact['source_origin']]/PurePosixPath(logical).relative_to('/run-output')) == fact['stat'], 'actual input changed during metadata roster collection')
    require(all(identity(path) == root_stats[role] for role,path in roots.items()) and identity(output) == output_stat,
            'actual root identity changed during metadata-only collection')
    catalog = {'schema':'chimera.k3_RO_multi_origin_prefix_fresh_NAS_output_source_route_review.v1',
               'parent_catalog_sha256':i['parent_catalog_sha256'],'legacy_catalog_sha256':LEGACY_CATALOG_SHA,'input_catalog':rows,'run_uuid':RUN,
               'source_provenance':'unchanged entire actual49 source catalog through48 plus authentic49-60 sealed fullraw receipt/continuity expectations',
               'actual_runtime_admission_claimed':False,'payloads_copied_or_rehashed':False}
    catalog_raw = (json.dumps(catalog,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    catalog_sha = hashlib.sha256(catalog_raw).hexdigest()
    observation = {'schema':'chimera.k3_RO_multi_origin_prefix_separate_output_observation.v1','observed_utc':at,
        'source_catalog_sha256':catalog_sha,'acting_identity':{'owner':i['owner'],'native_context':i['native_thread_id'],
            'run_uuid':RUN,'image':i['image'],'control_binding':i['native_binding']},'first_new_layer':73,'last_new_layer':84,
        'source_roots':{role:{'path':ALIASES[role],'alias':ALIASES[role],'host_path':str(path),'read_only':True,'stat':root_stats[role]} for role,path in roots.items()},
        'output_root':{'path':'/run-output','alias':'/run-output','host_path':str(output),'read_only':False,'stat':output_stat},
        'source_files':source_files,'new_write_paths':exact_write_paths(),'source_pins':SOURCES,
        'closed48_continuity':{'path':'/run-output/'+SEALED48_REL+'/window_continuity.json','sha256':i['closed48_continuity_sha256']},
        'closed60_continuity':{'path':'/run-output/'+SEALED60_REL+'/window_continuity.json','sha256':i['closed60_continuity_sha256']},
        'closed72_continuity':{'path':'/run-output/'+SEAL_REL+'/window_continuity.json','sha256':i['closed72_continuity_sha256']},
        'parent_catalog_sha256':i['parent_catalog_sha256'],'parent_catalog_rows_sha256':parent_rows_sha,'parent_catalog_row_count':len(parent_rows),
        'previous_recovery_packet':{'path':'/control/source_revisions/recovery_ancestry_capacity_wait.json','sha256':PREVIOUS_PACKET_SHA},
        'previous_native_source':{'path':'/control/source_revisions/k3_dense_pass_9086.py','sha256':PREVIOUS_NATIVE_SHA},
        'completed_native_source':{'path':'/control/source_revisions/k3_dense_pass_720.py','sha256':COMPLETED_NATIVE_SHA},
        'completed49_native_source':{'path':'/control/source_revisions/k3_dense_pass_5e.py','sha256':'5e852b63d28ba41599ca3e994851b3e72103f213384de77b9c172c7823036007'},
        'completed61_native_source':{'path':'/control/source_revisions/k3_dense_pass_61.py','sha256':'078187a4cfa2f5f7810f0c0a8fa85f7fcc8304085684d45dfda5b895854881d0'},
        'current_mounts':mounts,'mount_evidence_scope':'exact staged planned mapping; genuine raw allocation and actual normal mounted/cgroup attestation separately required before compute',
        'payload_hash_scope':'collector bulk payload bytes read/hashed0; each production resolver independently hashes consumed old files once and rechecks all six stat fields',
        'fresh_small_metadata_hashes':fresh_metadata,'runtime_admission_claimed':False,'catalog_payloads_copied':False,'old_prefix_modified':False}
    # Publication is exclusively in the NEW controller directory. No existing
    # current37-48 source or payload is opened for write, copied, or renamed.
    for name in ('ro_multi_origin_catalog.json','ro_multi_origin_observation.json','ro_multi_origin_collection_receipt.json'):
        require(not(control/name).exists() and not(control/name).is_symlink(), 'exclusive future metadata leaf already exists')
    catalog_result = exclusive_json(control/'ro_multi_origin_catalog.json',catalog)
    require(catalog_result['sha256'] == catalog_sha, 'exact generated catalog raw SHA differs')
    observation_result = exclusive_json(control/'ro_multi_origin_observation.json',observation)
    receipt = {'schema':'neural_forge.k3_actual73_84_multi_origin_metadata_collection.v1','at_utc':datetime.now(timezone.utc).isoformat(),
        'run_uuid':RUN,'identity_sha256':args.identity_sha256,'collector_sha256':whole_sha(Path(__file__).absolute()),
        'parent_catalog_sha256':i['parent_catalog_sha256'],'legacy_catalog_sha256':LEGACY_CATALOG_SHA,'closed72_sha256':i['closed72_continuity_sha256'],'closed48_sha256':i['closed48_continuity_sha256'],
        'mounts_sha256':args.mounts_sha256,'selected_source_pins':SOURCES,'verifier_sha256':VERIFIER_SHA,
        'source_current_stat_rows':len(source_files),'retained_held_rows':36,'fresh_metadata_rows':len(fresh_metadata),
        'fresh_metadata_bytes':sum(row['bytes'] for row in fresh_metadata.values()),'fresh_payload_bytes_hashed':0,
        'payload_copy_bytes':0,'committed_new_write_paths':82,'actual_model_or_runtime_admission_claimed':False,
        'output_root_stat':output_stat,'controller_root_stat':identity(meta),'catalog':catalog_result,'observation':observation_result}
    receipt_result = exclusive_json(control/'ro_multi_origin_collection_receipt.json',receipt)
    return {'catalog':catalog_result,'observation':observation_result,'collection_receipt':receipt_result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('identity','parent-catalog','mounts','output-directory'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--identity-sha256',required=True); parser.add_argument('--mounts-sha256',required=True)
    args = parser.parse_args()
    try: print(json.dumps(collect(args),sort_keys=True))
    except (OSError,ValueError,KeyError,TypeError) as error: parser.exit(2,'actual three-origin metadata observation refused: '+str(error)+'\n')


if __name__ == '__main__': main()
