"""Local two-step binder for one separately admitted K3 NAS49--60 window.

Current-peer V4 selects the normal-tested550-member4e8c packet and one
authentic public peer role; the prior0865/529-member binder stays unchanged.
This writes an exclusive local source bundle or allocation-bound configs only.
It never connects, uploads, allocates, changes a cgroup, starts a caller, or arms
a barrier. Authentic sealed48 input pins are mandatory before source binding.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from types import SimpleNamespace
import uuid
import zipfile

HERE = Path(__file__).resolve().parent
RECOVERY = HERE.parents[1]
PROPOSAL = RECOVERY / 'recovery_resource_review/nas49_60_continuity_feasibility_v1'
PACKET_DIR = RECOVERY / 'nas49_60_proposal/immutable_packet_v3'
RUN = '03603908-570d-4c90-96cb-ee3a63a9a139'
PREVIOUS_RUN = '764524b1-96a9-4636-be74-942f9c54754c'
PACKET_SHA = 'f28b4ef8acf58d89eca2956baf27b1a7bbb1390c54fa7a68d6a22f435485cc43'
MANIFEST_SHA = 'd593e851c5552fef362e8172a8dfa3e6a2bce2ee97ac59672057d8575c856a0d'
COMPOSITION_SHA = '8bd0ebb3678220b9b8ab6375dd4b2ed1efec53ac634ccc3ab89c4cdb94adb023'
DRAFT_SHA = '86aa5957e75f21356e552952f268a1b5ddfc7e9fb442ac2d1f817ab9a14cd805'
OLD_I_SHA = 'd8bf5e9270930e0bdb4b470a07402791d16529c0c23546a31c9ec79e500b4997'
WRITER_SHA = '16102e1523cbfd72c89df6e801554748e2601bae4a1d5ada081cfe22e85484a2'
NATIVE_SHA = '5e852b63d28ba41599ca3e994851b3e72103f213384de77b9c172c7823036007'
RESOLVER_SHA = 'be4f3aede6f94d242cad9c4037656c3de3eff8f2268f5f2b17d23cf69025a804'
RUNTIME_SHA = '4e58891969a6e83b6cf2247b8837edeb53d7a37d03de2aba07a4a14e7b150e6b'
GUARD_SHA = '88e118847efb589d9cc48fc578996c3ca48cd0a0b9a4b491aab573fed54a9d41'
GIB = 1 << 30
CAP_BYTES, CAP_CPUS = 96 * GIB, 22
WRITER_BYTES, COMMON_BYTES = 152170243072, 844196848640
RESIDUAL_DELTA = 8220835840
TOOL_REL = 'reports/omni-campaign-2026-09/composite/tools/'
SEALED36 = '/resume-prefix/ancestry/window-25-36-b873d51f-b251-4944-aa84-5d91df96e87b'
SEALED48_LOGICAL = '/run-output/ancestry/window-37-48-' + PREVIOUS_RUN
SEALED48_MOUNTED = '/resume-window37-48/ancestry/window-37-48-' + PREVIOUS_RUN

# Each selected blob has an explicit production destination. The 469-member
# frozen baseline contains720/9230/7a0; those three baseline blobs are overlaid.
DESTINATIONS = {
    'k3_dense_pass.py': ['production_code/' + TOOL_REL + 'k3_dense_pass.py'],
    'ro_resume_paths.py': ['production_code/' + TOOL_REL + 'ro_resume_paths.py', 'production_control/ro_resume_paths.py'],
    'ro_resume_runtime.py': ['production_code/' + TOOL_REL + 'ro_resume_runtime.py', 'production_control/ro_resume_runtime.py'],
    'k3_recovery_ancestry.py': ['production_code/' + TOOL_REL + 'k3_recovery_ancestry.py', 'production_control/k3_recovery_ancestry.py'],
    'verify_later_window.py': ['production_control/verify_later_window.py'],
    'continuation_worker49_60_template_v4.py': ['production_control/continuation_worker.py'],
    'k3_owned_readonly_facts49_60_template_v2.py': ['production_control/k3_owned_readonly_facts_NAS49_60_v1.py'],
    'k3_owned_capacity_preflight49_60_template_v2.py': ['production_control/k3_owned_capacity_preflight_NAS49_60_v1.py'],
    'k3_capacity_host_loop49_60_template_v2.py': ['production_control/k3_capacity_host_loop_v1.py'],
    'native_nas_io_qualification49_60_template_v2.py': ['production_control/native_nas_io_qualification.py'],
}
SOURCE_INPUT_ROLES = {'closed48_continuity', 'terminal_receipt', 'full_ffn_validation',
                      'snapshot_manifest', 'fitter_execution', 'controller_terminal_result',
                      'resources_jsonl', 'controller_events_jsonl', 'binding_receipt'}


def require(ok, message):
    if ok is not True:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def full_sha(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
            'unresolved or malformed full SHA256')
    return value


def encode(body):
    return (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def read_raw(path, expected, limit=64 << 20):
    p = Path(path)
    full_sha(expected)
    require(p.is_file() and not p.is_symlink() and p.stat().st_size <= limit,
            'regular bounded local input required: ' + str(p))
    before = p.stat(); raw = p.read_bytes(); after = p.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'local input changed during read')
    require(sha(raw) == expected, 'raw fullSHA differs: ' + str(p))
    return raw


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, 'nonfinite JSON'))


def read_ref(ref, json_value=True):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'} and
            isinstance(ref['path'], str) and bool(ref['path']), 'unresolved exact local input reference')
    raw = read_raw(ref['path'], ref['sha256'])
    return (decode(raw) if json_value else raw), raw


def pin(path, raw, name=None):
    return {'path': str(Path(path).resolve()), 'name': name or Path(path).name,
            'sha256': sha(raw), 'bytes': len(raw)}


def leaf(row):
    require(PurePosixPath(row['name']).name == row['name'], 'bounded control leaf required')
    return {'name': row['name'], 'sha256': full_sha(row['sha256'])}


def write_raw(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    require(path.read_bytes() == raw, 'exclusive output byte readback differs')
    return pin(path, raw)


def bind_assignments(raw, values):
    """Change only one exact top-level literal assignment for each named slot."""
    text = raw.decode(); tree = ast.parse(text); spans = {}
    lines = text.splitlines(keepends=True); offsets = [0]
    for line in lines: offsets.append(offsets[-1] + len(line))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in values:
                require(name not in spans, 'duplicate literal binding slot')
                ast.literal_eval(node.value)
                spans[name] = (offsets[node.lineno-1] + node.col_offset,
                               offsets[node.end_lineno-1] + node.end_col_offset)
    require(set(spans) == set(values), 'missing exact top-level binding slot')
    for name, (start, end) in sorted(spans.items(), key=lambda row: row[1][0], reverse=True):
        text = text[:start] + name + ' = ' + repr(values[name]) + text[end:]
    new = ast.parse(text)
    funcs = lambda tr: [ast.dump(n, include_attributes=False) for n in tr.body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    require(funcs(tree) == funcs(new), 'binding changed reviewed function/class AST')
    compile(text, '<bound source>', 'exec')
    return text.encode()


def packet_members():
    manifest = decode(read_raw(PACKET_DIR / 'K3_NAS49_60_CURRENT_PEER_IMMUTABLE_SOURCE_PACKET_MANIFEST_V3.json', MANIFEST_SHA))
    packet = read_raw(PACKET_DIR / 'K3_NAS49_60_CURRENT_PEER_IMMUTABLE_SOURCE_PACKET_V3.zip', PACKET_SHA)
    require(manifest['normal_promoted_commit'] == '4e8c18ec39b2b9540422dea05d68b47c635aec41'
            and manifest['member_count'] == 550 and manifest['selected_module_count'] == 10,
            'corrected promoted immutable source packet required')
    import io
    result = {}
    with zipfile.ZipFile(io.BytesIO(packet)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) == 550 and set(names) == set(manifest['members']),
                'whole source archive member roster differs')
        for name in names:
            p = PurePosixPath(name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in name,
                    'canonical relative archive member required')
            raw = archive.read(name); expected = manifest['members'][name]
            require(len(raw) == expected['bytes'] and sha(raw) == expected['sha256'],
                    'immutable packet member fullSHA/bytes differs: ' + name)
            result[name] = raw
    require(set(manifest['selected_source_template_modules']) == set(DESTINATIONS),
            'all ten selected module destinations must be explicit')
    require(manifest['schema'] == 'neural_forge.k3_NAS49_60_current_peer_immutable_source_packet.v3'
            and manifest['canonical_other9_modules_unchanged'] is True
            and manifest['normal_promotion_total_source_cases'] == 12
            and manifest['preserved_parent_packet_sha256'] == 'd5c0c96d3e294268f01b3bfc1f47ce76d2c19ee6cba7268c007fbe7e162e1d43'
            and sha(result['committed_candidate_v4/k3_owned_capacity_preflight49_60_template_v2.py'])
                == 'c3c6359b79342824af5b77d5073e4dabde720501de8648de587defbecb2f102e',
            'exact tested-promoted current-peer derivative and preserved source parent required')
    return manifest, result


def validate_closed48(inputs, runtime_raw):
    require(set(inputs) == SOURCE_INPUT_ROLES, 'complete exact sealed48 source input roster required')
    values, raws = {}, {}
    for role, ref in inputs.items():
        values[role], raws[role] = read_ref(ref, not role.endswith('_jsonl'))
    closed, terminal = values['closed48_continuity'], values['terminal_receipt']
    scan, snapshots = values['full_ffn_validation'], values['snapshot_manifest']
    require(closed.get('schema') == 'neural_forge.k3_actual_window_continuity.v1'
            and closed.get('completed_window') == '37-48' and closed.get('full_prefix_layers') == 48,
            'actual closed48 seal has not been emitted')
    require(closed['receipt_sha256'] == sha(raws['terminal_receipt'])
            and closed['snapshot_manifest_sha256'] == sha(raws['snapshot_manifest'])
            and closed['ffn_validation_sha256'] == sha(raws['full_ffn_validation']),
            'sealed48 exact terminal/manifest/full-scanner raw joins differ')
    require(closed['resume_context']['software']['tool_sha256'] ==
            '720465874722e1548c08f82a019be7dcb477efcc5828d6edd10610c1ea076243',
            'authentic completed720 producer context required')
    # Use the promoted runtime's pure metadata validator. The map is raw
    # metadata pins only; no claimed local tensor read/hash or fabricated FS.
    old = {}
    for row in closed['prefix']:
        root = f"/run-output/fold/dense/L{row['layer']:02d}/"
        old[root+'ffn.safetensors'] = {'sha256': full_sha(row['ffn_sha256'])}
        old[root+'done.json'] = {'sha256': full_sha(row['done_sha256'])}
    for stream in ('teacher', 'folded'):
        suffix = '' if stream == 'teacher' else '_folded'
        logical = '/run-output/fold/snapshots/before_L49' + suffix + '.safetensors'
        old[logical] = {'sha256': full_sha(snapshots['files'][logical]['sha256'])}
    tree = ast.parse(runtime_raw)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in ('context_equal', 'validate_closed48_metadata')]
    require(len(selected) == 2, 'exact promoted pure closed48 metadata functions required')
    env = {'json': json, 'require': require,
           'CLOSED36': 'dd0ab8e821307e352096cdded457934dfd6df3434cfbe40d74e24b09a215870b'}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<promoted pure closed48 join>', 'exec'), env)
    prior_closed = decode(read_raw(
        RECOVERY/'generation5_capacity_wait/actual_closed36_metadata_20261007T101629064667Z/window_continuity.json',
        'dd0ab8e821307e352096cdded457934dfd6df3434cfbe40d74e24b09a215870b'))
    expected_context = copy.deepcopy(prior_closed['resume_context'])
    expected_context['software']['tool_sha256'] = '720465874722e1548c08f82a019be7dcb477efcc5828d6edd10610c1ea076243'
    env['validate_closed48_metadata'](closed, scan, terminal, snapshots,
                                     SimpleNamespace(old=old), expected_context)
    status = values['controller_terminal_result']
    require(status['schema'] == 'neural_forge.k3_actual_continuation_worker_status.v1'
            and status['identity']['run_uuid'] == PREVIOUS_RUN
            and status['config_sha256'] == '7c351f93bac052a4da33c638133d9922c728dee5fcaa2e659f87039cbf6ef640'
            and status['allocation']['allocation_id'] == '572e149e-57d7-41db-ac73-4e3deb262287'
            and status['allocation']['container_id'] == '0adb0c5b803caa5ff22858f636b4df3fce7ff51366bc5159748e7c016b825231'
            and len(status['completed_windows']) == 1
            and status['completed_windows'][0]['layers'] == '37-48'
            and status['completed_windows'][0]['continuity_sha256'] == sha(raws['closed48_continuity'])
            and status['completed_windows'][0]['terminal_receipt_sha256'] == sha(raws['terminal_receipt']),
            'actual one-window controller terminal/result source join differs')
    execution = values['fitter_execution']
    events = [decode(line) for line in raws['controller_events_jsonl'].splitlines() if line]
    started = [row for row in events if row['stage'] == 'native_fitter_started']
    running = [row for row in events if row['stage'] == 'native_fitter_running']
    ended = [row for row in events if row['stage'] == 'native_fitter_terminal']
    require(execution['phase'] == 'native_fitter' and execution['exit_code'] == 0
            and execution['pid'] == 203 and execution['ended_at']
            and len(started) == len(ended) == 1 and running
            and all(row['child_pid'] == execution['pid'] for row in running)
            and started[0]['command_argv'] == execution['argv']
            and ended[0]['actual_command_result'] == execution,
            'authentic natural successful37-48 native terminal execution required')
    resources = [decode(line) for line in raws['resources_jsonl'].splitlines() if line]
    terminal_rows = [row for row in resources if row['stage'] == 'window_validated']
    require(len(terminal_rows) == 1, 'actual post-reader sealed48 resource sample required')
    frames = [execution['resources_after'], terminal_rows[0]]
    peaks = []
    for frame in frames:
        cg = frame['cgroup']
        require(cg['memory.max'] == 88*GIB and cg['memory.swap.max'] == 0
                and cg['memory.swap.current'] == 0 and cg['memory.events'].get('oom_kill', 0) == 0,
                'actual old88GiB/noSwap/non-OOM completed phase evidence required')
        peaks.append(cg['memory.peak'])
    peak = max(peaks)
    require(type(peak) is int and 0 < peak <= 88*GIB and peak + RESIDUAL_DELTA <= CAP_BYTES,
            'actual final48 peak plus source array delta exceeds proposed next96GiB cap')
    receipt = values['binding_receipt']
    require(receipt['exit_code'] == 0 and receipt['response']['ok'] is True
            and receipt['argv'][-1] == 'connect', 'authentic normal current-session connect receipt required')
    data = receipt['response']['data']
    binding = {key: data[key] for key in ('harness_id', 'thread_id', 'session_id', 'session_generation', 'context_sha256')}
    require(binding['harness_id'] == '01a11108-009c-72f3-b2ba-e0d48fca1435'
            and binding['thread_id'] == '95f6c2bd-7139-46f6-b5d6-228d1b9753f7'
            and type(binding['session_generation']) is int and binding['session_generation'] >= 6
            and str(uuid.UUID(binding['session_id'])) == binding['session_id'],
            'genuine own current native binding required')
    full_sha(binding['context_sha256'])
    return values, raws, binding, peak


def source_bind(args):
    spec, _ = read_ref({'path': str(args.inputs), 'sha256': args.inputs_sha256})
    require(spec['schema'] == 'neural_forge.k3_actual49_60_local_source_binding_inputs.v1'
            and spec['run_uuid'] == RUN, 'exact reserved one-window binding input schema required')
    # A null actual48 input refuses before creating any output directory.
    require(set(spec['inputs']) == SOURCE_INPUT_ROLES, 'complete actual48 metadata references required')
    for ref in spec['inputs'].values():
        require(isinstance(ref, dict) and isinstance(ref.get('path'), str)
                and isinstance(ref.get('sha256'), str), 'actual48/binding source reference remains unresolved')
    manifest, members = packet_members()
    values, raw_inputs, binding, peak = validate_closed48(spec['inputs'], members['committed_candidate_v4/ro_resume_runtime.py'])
    composition = decode(read_raw(PROPOSAL/'admission_caller_proposal_v4/SOURCE_ONLY_49_60_EXACT_ADMISSION_CALLER_COMPOSITION_V4.json', COMPOSITION_SHA))
    draft = decode(read_raw(HERE/'working_identity_draft.json', DRAFT_SHA))
    draft['native_binding'] = binding; draft['binding_receipt'] = spec['inputs']['binding_receipt']['path']
    draft['closed48_continuity_sha256'] = sha(raw_inputs['closed48_continuity'])
    draft['identity_role'] = 'separate49-60 source-bound identity; allocation and actual runtime admission are not claimed'
    identity_raw, binding_raw = encode(draft), encode(binding)
    common = {'EXPECTED_RUN': RUN, 'IDENTITY_SHA': sha(identity_raw), 'BINDING_SHA': sha(binding_raw),
              'CLOSED48_SHA': sha(raw_inputs['closed48_continuity']), 'FUTURE_SESSION_ID': binding['session_id'],
              'FUTURE_GENERATION': binding['session_generation'], 'FUTURE_CONTEXT_SHA': binding['context_sha256'],
              'OWN_RAM_BYTES': CAP_BYTES, 'OWN_CPU_THREADS': CAP_CPUS}
    generated = {}
    for name in DESTINATIONS:
        generated[name] = members['committed_candidate_v4/'+name]
    generated['k3_owned_readonly_facts49_60_template_v2.py'] = bind_assignments(generated['k3_owned_readonly_facts49_60_template_v2.py'], common)
    generated['k3_capacity_host_loop49_60_template_v2.py'] = bind_assignments(generated['k3_capacity_host_loop49_60_template_v2.py'], common)
    generated['continuation_worker49_60_template_v4.py'] = bind_assignments(generated['continuation_worker49_60_template_v4.py'],
        {'RUN': RUN, 'IDENTITY_SHA': sha(identity_raw), 'CAP_BYTES': CAP_BYTES, 'CAP_CPUS': CAP_CPUS,
         'SESSION_GENERATION': binding['session_generation'], 'SESSION_ID': binding['session_id']})
    generated['native_nas_io_qualification49_60_template_v2.py'] = bind_assignments(generated['native_nas_io_qualification49_60_template_v2.py'], {'EXPECTED_RUN': RUN})
    output = args.output_directory.resolve()
    require(output.parent == HERE and not output.exists() and not output.is_symlink(),
            'new exclusive source bundle must be a direct child of the reserved local leaf')
    roster = copy.deepcopy(composition['whole_capacity_metadata_role_roster'])
    additions = manifest['metadata_roster_additions']
    require(set(additions) == {'current_peer_holder_source'} and 'current_peer_holder_source' not in roster,
            'only the exact authenticated current-peer source role may be added')
    added = additions['current_peer_holder_source']
    require(added == {'member': 'committed_candidate_v4/current_peer_evidence/CURRENT_QIMG_NATIVE_AND_ROOT1075_PUBLIC_PROOF_V3.json',
                     'name': 'CURRENT_QIMG_NATIVE_AND_ROOT1075_PUBLIC_PROOF_V3.json',
                     'sha256': '1a49ab6d7ae2c5cff10a93df3e6ef63a0f239feb9646ec246c3a9b5cc5d242b9', 'bytes': 1797866},
            'complete current-peer raw proof role differs')
    peer_raw = members[added['member']]
    require(sha(peer_raw) == added['sha256'] and len(peer_raw) == added['bytes'],
            'promoted current-peer proof fullSHA/bytes differs')
    roster['current_peer_holder_source'] = {key: added[key] for key in ('name', 'sha256', 'bytes')}
    raw_roles = {}
    for role, row in roster.items():
        if role in ('identity', 'binding', 'closed48_continuity', 'NAS_current_scope', 'current_peer_holder_source'): continue
        raw = read_raw(row['path'], row['sha256'])
        require(len(raw) == row['bytes'], 'complete capacity metadata bytes differ')
        raw_roles[role] = raw
    raw_roles.update(identity=identity_raw, binding=binding_raw, closed48_continuity=raw_inputs['closed48_continuity'], current_peer_holder_source=peer_raw)
    roster['identity']['name'] = 'working_identity.json'; roster['binding']['name'] = 'selected_native_binding49_60.json'
    scope = decode(read_raw(composition['unbound_current_scope']['path'], composition['unbound_current_scope']['sha256']))
    scope['run_uuid'] = RUN; scope['identity_sha256'] = sha(identity_raw)
    scope['source_proposal_only'] = False
    for role in ('identity', 'closed48_continuity'):
        scope['source_pins'][role] = {'name': roster[role]['name'], 'sha256': sha(raw_roles[role])}
    require(scope['NAS_current_common_guard_bytes_once'] == COMMON_BYTES
            and scope['NAS_common_guard_current_K3_writer_bytes_once'] == WRITER_BYTES,
            'whole new writer must replace old130GiB once')
    raw_roles['NAS_current_scope'] = encode(scope)
    roster['NAS_current_scope']['name'] = 'NAS49_60_BOUND_RESOURCE_SCOPE_V1.json'
    generated['k3_owned_capacity_preflight49_60_template_v2.py'] = bind_assignments(
        generated['k3_owned_capacity_preflight49_60_template_v2.py'],
        dict(common, FACTS_SHA=sha(generated['k3_owned_readonly_facts49_60_template_v2.py']),
             CURRENT_SCOPE_SHA=sha(raw_roles['NAS_current_scope'])))
    # These three dated observer pins are exact native import/read guards.
    observer = read_raw(RECOVERY/'nas_window37_48_generation6/observe_native.py',
                        'f8a5a53da85dfd2b83cbe8ab521e1cb2544cfefbd1641ea0f6de6dba921f4906')
    observer_parent_sha = sha(observer)
    for old, new in [('720465874722e1548c08f82a019be7dcb477efcc5828d6edd10610c1ea076243', NATIVE_SHA),
                     ('9230b81428d6ddf35748a471443fb7ffc7cd4a0d9ac2cec6a5b61a170cd0742d', RESOLVER_SHA),
                     ('7a0d25b0314029cd8177a7e292f377466ab24d3c0cfa44cf9135ae754ae35233', RUNTIME_SHA)]:
        require(observer.count(old.encode()) == 1, 'exact dated observer source pin slot required')
        observer = observer.replace(old.encode(), new.encode())
    boot = read_raw(RECOVERY/'nas_window37_48_generation6/recovery_boot.py',
                    'd273f39e937a59f28519975617b7b0102d8bf3b178bcbe0edde684516c11a9c8')
    boot_parent_sha = sha(boot)
    require(boot.count(PREVIOUS_RUN.encode()) == 1, 'exact dated boot RUN slot required')
    boot = boot.replace(PREVIOUS_RUN.encode(), RUN.encode())
    staged = {name[len('frozen_gen6_base/'):]: raw for name, raw in members.items()
              if name.startswith('frozen_gen6_base/')}
    require(sum(name.startswith('production_code/') for name in staged) == 469,
            'whole authenticated469-code baseline required')
    for name, paths in DESTINATIONS.items():
        for dest in paths: staged[dest] = generated[name]
    staged['production_control/observe_native.py'] = observer
    staged['production_control/recovery_boot.py'] = boot
    completed_native = members['frozen_gen6_base/production_code/'+TOOL_REL+'k3_dense_pass.py']
    require(sha(completed_native) == '720465874722e1548c08f82a019be7dcb477efcc5828d6edd10610c1ea076243',
            'completed720 source must be retained before canonical5e overlay')
    staged['production_control/source_revisions/k3_dense_pass_720.py'] = completed_native
    staged['production_control/closed36_next_resource_admission.json'] = read_raw(
        RECOVERY/'generation5_capacity_wait/actual_closed36_metadata_20261007T101629064667Z/next_resource_admission.json',
        '1e9004bc11daca688ab836e460517759d8b7d36c7f174d27748f1a38b987443a')
    staged['production_control/identity.json'] = identity_raw
    for role, raw in raw_roles.items():
        dest = 'production_control/'+roster[role]['name']
        require(dest not in staged or staged[dest] == raw, 'different bytes share a production control leaf')
        staged[dest] = raw
        roster[role] = pin(output/dest, raw, roster[role]['name'])
    scripts = {}
    role_name = {'facts': 'k3_owned_readonly_facts_NAS49_60_v1.py', 'preflight': 'k3_owned_capacity_preflight_NAS49_60_v1.py',
                 'host_caller': 'k3_capacity_host_loop_v1.py'}
    for role in ('facts', 'preflight', 'host_caller', 'service', 'math'):
        name = role_name.get(role, composition['base_scripts'][role]['name'])
        if role in ('service', 'math'):
            original = composition['base_scripts'][role]
            staged['production_control/'+name] = read_raw(original['path'], original['sha256'])
        scripts[role] = pin(output/'production_control'/name, staged['production_control/'+name])
    for name, raw in staged.items():
        if name.endswith('.py'): compile(raw, name, 'exec')
    source_proof = {'schema': 'neural_forge.k3_actual49_60_local_source_binding.v1', 'run_uuid': RUN,
                    'identity_sha256': sha(identity_raw), 'closed48_sha256': sha(raw_inputs['closed48_continuity']),
                    'immutable_packet_sha256': PACKET_SHA, 'immutable_manifest_sha256': MANIFEST_SHA,
                    'normal_promoted_commit': manifest['normal_promoted_commit'], 'production_destinations': DESTINATIONS,
                    'actual48_input_raw_pins': spec['inputs'], 'actual_final48_cgroup_peak_bytes': peak,
                    'execution_PID_namespace': 'container Popen child203; actual event/argv/terminal joined',
                    'prior_observed_physical_native_host_identity': {'pid': 3541371, 'birth_ticks': 300296582,
                        'scope': 'separate historical physical identity; not execution.pid and not a new live attestation'},
                    'array_coexistence_delta_bytes_not_RSS': RESIDUAL_DELTA,
                    'operating_cap_proposal_bytes': CAP_BYTES, 'operating_cap_increase_bytes': 8*GIB,
                    'whole_RSS_upper_bound_claimed': False, 'fresh_whole_capacity_admission_required': True,
                    'writer_bytes_once_including_new_qualification': WRITER_BYTES,
                    'common_NAS_floor_compare_bytes': COMMON_BYTES,
                    'inherited_common_padding_is_not_new_phase_jobs': True,
                    'observer_parent_sha256': observer_parent_sha, 'boot_parent_sha256': boot_parent_sha,
                    'production_file_pins': {name: {'sha256': sha(raw), 'bytes': len(raw)} for name, raw in staged.items()},
                    'current_peer_roles_retained_as_pinned_conservative_history': True,
                    'fresh_actual_peer_source_adoption_or_unknown_refusal_required': True,
                    'current_peer_source_role_added': 'current_peer_holder_source',
                    'current_peer_public_proof_sha256': sha(peer_raw),
                    'current_peer_source_composition_sha256': manifest['current_peer_source_composition_sha256'],
                    'capacity_metadata_role_count': len(roster),
                    'current_peer_actual_cap_or_heavy_start_claimed_by_binding': False,
                    'preserved_prior_binder_sha256': '0865a0e8810de72ac2599eaa5fc363ce20ed1e4493b3b61084242e226fd3c5d0',
                    'model_payload_reads': 0, 'remote_or_allocation_or_runtime_actions': False}
    closure = {'schema': 'neural_forge.k3_owned_capacity_source_closure.v15', 'selected_run_uuid': RUN,
               'sources': roster, 'scripts': scripts, 'source_binding': source_proof,
               'actual_runtime_admission_claimed': False}
    old_a = decode(read_raw(RECOVERY/'nas_window37_48_generation6/actual_allocation.json',
                             '8eb0a6e0744cbf075f0be71c103665b3b0772cb5e18ef653759bc86d40e713dc'))
    mounts = copy.deepcopy(old_a['details']['mounts'])
    replacements = {'/code/neural-forge': draft['runtime_root']+'/production_code',
                    '/control': draft['runtime_root']+'/production_control', '/run-control-output': draft['output_root'],
                    '/run-output': draft['native_working_output_root']}
    for row in mounts:
        if row['target'] in replacements: row['source'] = replacements[row['target']]
    mounts.append({'source': draft['readonly_resume_source_roots']['closed37_48'],
                   'target': '/resume-window37-48', 'read_only': True})
    mounts.sort(key=lambda row: row['target'])
    require(len(mounts) == 10, 'exact ten two-origin/fresh-output mount rows required')
    staged['mounts.json'] = encode(mounts)
    source_proof['mounts_sha256'] = sha(staged['mounts.json'])
    staged['SOURCE_CLOSURE49_60.json'] = encode(closure)
    staged['SOURCE_BINDING_RESULT.json'] = encode(source_proof)
    staged['CAP_SELECTOR_PROPOSAL.json'] = encode({'memory_bytes': CAP_BYTES, 'memory_swap_limit_bytes': CAP_BYTES,
        'swap_bytes': 0, 'cpu_threads': CAP_CPUS, 'NAS_whole_writer_bytes': WRITER_BYTES,
        'NAS_qualification_inside_writer_bytes': 536870912, 'ROOT_metadata_bytes': 2*GIB,
        'host_headroom_bytes': 16*GIB, 'actual_cgroup_limits_not_yet_applied': True})
    # All metadata, selected member hashes and bound sources validate before
    # the first output-directory mutation. No payload or old-tree copies.
    output.mkdir(exist_ok=False)
    for name, raw in sorted(staged.items()): write_raw(output/name, raw)
    print(json.dumps({'path': str(output), 'source_binding_sha256': sha(staged['SOURCE_BINDING_RESULT.json']),
                      'closure_sha256': sha(staged['SOURCE_CLOSURE49_60.json']), 'allocation_started': False}))


def verify_bound_bundle(bundle, closure):
    require(closure['schema'] == 'neural_forge.k3_owned_capacity_source_closure.v15'
            and closure['selected_run_uuid'] == RUN and len(closure['sources']) == 63
            and len(closure['scripts']) == 5, 'complete source-bound63 metadata/five-script closure required')
    peer = closure['sources'].get('current_peer_holder_source')
    require(isinstance(peer, dict) and peer.get('name') == 'CURRENT_QIMG_NATIVE_AND_ROOT1075_PUBLIC_PROOF_V3.json'
            and peer.get('sha256') == '1a49ab6d7ae2c5cff10a93df3e6ef63a0f239feb9646ec246c3a9b5cc5d242b9'
            and peer.get('bytes') == 1797866, 'complete exact current-peer proof role required')
    read_raw(bundle/'mounts.json', closure['source_binding']['mounts_sha256'])
    expected = closure['source_binding']['production_file_pins']
    actual = {}
    for root in ('production_code', 'production_control'):
        for path in (bundle/root).rglob('*'):
            require(not path.is_symlink(), 'bound production source symlink refused')
            if path.is_file(): actual[path.relative_to(bundle).as_posix()] = path
    require(set(actual) == set(expected), 'whole source-bound production file roster changed')
    for name, path in actual.items():
        raw = read_raw(path, expected[name]['sha256'])
        require(len(raw) == expected[name]['bytes'], 'source-bound production file extent changed')


def observation_metadata(catalog, catalog_sha, observation, identity, mounts, closed):
    """Check source-recorded metadata only; selected runtime rechecks physical IO."""
    require(catalog['schema'] == 'chimera.k3_RO_multi_origin_prefix_fresh_NAS_output_source_route_review.v1'
            and observation['schema'] == 'chimera.k3_RO_multi_origin_prefix_separate_output_observation.v1'
            and observation['source_catalog_sha256'] == catalog_sha
            and observation['first_new_layer'] == 49 and observation['last_new_layer'] == 60
            and observation['current_mounts'] == mounts,
            'actual multi-origin metadata/source-catalog/mount joins differ')
    require(observation['acting_identity'] == {'owner': identity['owner'], 'native_context': identity['native_thread_id'],
            'run_uuid': RUN, 'image': identity['image'], 'control_binding': identity['native_binding']},
            'current own full acting identity differs')
    roots = observation['source_roots']
    require(set(roots) == {'legacy_closed36', 'closed37_48'}, 'exact two actual RO origins required')
    for role, alias in [('legacy_closed36', '/resume-prefix'), ('closed37_48', '/resume-window37-48')]:
        require(roots[role]['path'] == roots[role]['alias'] == alias and roots[role]['read_only'] is True
                and roots[role]['host_path'] == identity['readonly_resume_source_roots'][role],
                'exact actual source-origin alias/host/read-only role differs')
    out = observation['output_root']
    require(out['path'] == out['alias'] == '/run-output' and out['read_only'] is False
            and out['host_path'] == identity['native_working_output_root'], 'unique own fresh NAS RW root differs')
    all_roots = list(roots.values()) + [out]
    for i, left in enumerate(all_roots):
        for right in all_roots[i+1:]:
            for key in ('path', 'host_path'):
                a, b = PurePosixPath(left[key]), PurePosixPath(right[key])
                require(a != b and not a.is_relative_to(b) and not b.is_relative_to(a), 'RO/RW roots must be disjoint')
            require((left['stat']['dev'], left['stat']['ino']) != (right['stat']['dev'], right['stat']['ino']),
                    'distinct roots cannot be the same physical directory')
    require(observation['closed48_continuity'] == {'path': SEALED48_LOGICAL+'/window_continuity.json',
            'sha256': identity['closed48_continuity_sha256']}, 'latest actual sealed48 origin differs')
    rows = catalog['input_catalog']; observed = observation['source_files']
    require(len(rows) > 127 and catalog['parent_catalog_sha256'] ==
            '7a6b254fd7822bb2b868be149f2246b892868a7b2bbb0daa5f37db5ae1fe5cd2', 'actual old127 plus new37-48 catalog required')
    inherited = [{key: value for key, value in row.items() if key != 'source_origin'} for row in rows[:127]]
    require(sha(json.dumps(inherited, sort_keys=True, separators=(',', ':')).encode()) ==
            'c5d06a6f17377abdb50e948a830861f0b1539f83f4ca404c64d4a1166f3c7eb8', 'old127 catalog row values changed')
    old = {}
    for i, row in enumerate(rows):
        logical = row.get('recorded_logical_path') or '/run-output/'+row['relative_path']
        rel = PurePosixPath(logical).relative_to('/run-output')
        origin = row['source_origin']; root = roots[origin]
        require((origin == 'legacy_closed36') == (i < 127) and logical not in old
                and row['prospective_RO_input_path'] == root['alias']+'/'+str(rel)
                and row['unchanged_host_input_path'] == str(PurePosixPath(root['host_path'])/rel),
                'unique exact per-row actual source-origin mapping required')
        fact = observed[logical]
        require(fact['source_origin'] == origin and fact['source_sha256'] == full_sha(row['sha256'])
                and fact['path'] == root['alias']+'/'+str(rel) and fact['stat']['dev'] == root['stat']['dev']
                and (('bytes' not in row) or fact['stat']['size'] == row['bytes']), 'raw actual row/stat/source hash join differs')
        old[logical] = row
    require(set(old) == set(observed), 'all actual old source rows must be observed exactly')
    for row in closed['prefix']:
        root = f"/run-output/fold/dense/L{row['layer']:02d}/"
        require(old[root+'ffn.safetensors']['sha256'] == row['ffn_sha256']
                and old[root+'done.json']['sha256'] == row['done_sha256'], 'full actual48 prefix/catalog bytes differ')
    require(all(f'/run-output/fold/resid/L{i:02d}/held_rows.safetensors' in old for i in range(37,49)),
            'all12 actual retained held-row extent inputs required')
    require(observation['source_pins']['ro_resume_paths.py'] == RESOLVER_SHA
            and observation['source_pins']['ro_resume_runtime.py'] == RUNTIME_SHA
            and observation['source_pins']['k3_recovery_ancestry.py'] == GUARD_SHA
            and observation['source_pins']['k3_dense_pass.py'] == NATIVE_SHA, 'current executing/guard source pins differ')
    require(observation['previous_recovery_packet'] == {'path': '/control/source_revisions/recovery_ancestry_capacity_wait.json',
            'sha256': '5b9559b8ac20428de7e4ac09201f71839f4d03113a985715ca81d3aedd01246c'}
            and observation['previous_native_source'] == {'path': '/control/source_revisions/k3_dense_pass_9086.py',
            'sha256': '9086d7c0a5ec2bc7cfd1b823aecdc12e2973b3c81b064776ad3670df9d6b7b69'}
            and observation['completed_native_source'] == {'path': '/control/source_revisions/k3_dense_pass_720.py',
            'sha256': '720465874722e1548c08f82a019be7dcb477efcc5828d6edd10610c1ea076243'},
            'exact unchanged historical producer/context packet source roles differ')
    require(isinstance(observation['new_write_paths'], list) and observation['new_write_paths']
            and len(set(observation['new_write_paths'])) == len(observation['new_write_paths'])
            and not (set(observation['new_write_paths']) & set(old)), 'finite disjoint new49-60 writes required')


def plans_bind(args):
    spec, _ = read_ref({'path': str(args.inputs), 'sha256': args.inputs_sha256})
    require(spec['schema'] == 'neural_forge.k3_actual49_60_local_plan_binding_inputs.v1'
            and spec['run_uuid'] == RUN and set(spec['inputs']) == {'catalog', 'observation'},
            'actual metadata-only two-origin plan input schema required')
    bundle = Path(spec['source_bundle']).resolve()
    require(bundle.parent == HERE and bundle.is_dir() and not bundle.is_symlink(), 'own bound source bundle required')
    closure = decode(read_raw(bundle/'SOURCE_CLOSURE49_60.json', spec['source_closure_sha256']))
    verify_bound_bundle(bundle, closure)
    control = bundle/'production_control'
    identity = decode(read_raw(control/'identity.json', closure['sources']['identity']['sha256']))
    closed = decode(read_raw(control/closure['sources']['closed48_continuity']['name'], identity['closed48_continuity_sha256']))
    mounts = decode((bundle/'mounts.json').read_bytes())
    catalog, catalog_raw = read_ref(spec['inputs']['catalog'])
    observation, observation_raw = read_ref(spec['inputs']['observation'])
    observation_metadata(catalog, sha(catalog_raw), observation, identity, mounts, closed)
    prior = decode(read_raw(RECOVERY/'nas_window37_48_generation6/recovery_plan_v3.json',
                            '6de97afa62fc208d0eefdbf4e974e2a76dd233a48097bb091004c34ca63c3c5b'))
    # Use the reviewed Driver's exact pure original-argv transformer. Old
    # assertions remain inside dated_original_plan_provenance only.
    driver_raw = (control/'continuation_worker.py').read_bytes()
    tree = ast.parse(driver_raw)
    transform = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'operational_argv']
    require(len(transform) == 1, 'exact selected pure operational argv function required')
    env = {'require': require}
    exec(compile(ast.Module(body=transform, type_ignores=[]), '<reviewed original argv transform>', 'exec'), env)
    argv = env['operational_argv'](prior['dated_original_plan_provenance']['plan']['command_argv'], '49-60')
    require(argv[:4] == ['env', 'OMP_NUM_THREADS=22', 'MKL_NUM_THREADS=22', 'OPENBLAS_NUM_THREADS=22'],
            'source selected original CPU22 environment required')
    refs = {'manifest': {'path': '/control/ro_multi_origin_catalog.json', 'sha256': sha(catalog_raw)},
            'observation': {'path': '/control/ro_multi_origin_observation.json', 'sha256': sha(observation_raw)},
            'identity': {'path': '/control/identity.json', 'sha256': closure['sources']['identity']['sha256']}}
    flags = []
    for role in ('manifest', 'observation', 'identity'):
        flags += ['--resume-source-'+role, refs[role]['path'], '--resume-source-'+role+'-sha256', refs[role]['sha256']]
    flags += ['--resume-source-root', '/resume-prefix']
    plan = {'schema': 'neural_forge.k3_actual_NAS_closed48_record_plan.v1', 'identity': identity,
            'command_argv': argv+flags, 'next_layers': '49-60', 'mounts': mounts, 'image_id': identity['image'],
            'required_gpu_uuid': identity['gpu_uuid'], 'native_source_sha256': NATIVE_SHA,
            'closed48_continuity_sha256': identity['closed48_continuity_sha256'],
            'original_record_plan_sha256': prior['original_record_plan_sha256'],
            'dated_original_plan_provenance': prior['dated_original_plan_provenance'],
            'RO_resume_source_refs': refs, 'RO_resume_source_flags': flags,
            'current_native_software_and_inputs_require_actual_runtime_observation': True,
            'actual_runtime_admitted': False, 'actual_fitter_started': False, 'later_windows_admitted': False,
            'capability_accepted': False, 'training_eligible': False, 'numerical_targets_enforced': False,
            'artifact_scope': 'unique new49-60 NAS outputs; exact old0-36 Lexar and actual37-48 NAS inputsRO',
            'created_utc': datetime.now(timezone.utc).isoformat()}
    plan_raw = encode(plan)
    runtime = {'schema': 'neural_forge.k3_owned_NAS49_60_immutable_runtime_plan.v1', 'identity': identity,
               'plan_path': '/control/recovery_plan49_60.json', 'plan_sha256': sha(plan_raw), 'current_mounts': mounts,
               'source_pins': {p.relative_to(control).as_posix(): sha(p.read_bytes()) for p in control.rglob('*') if p.is_file()},
               'budget': {'memory_bytes': CAP_BYTES, 'cpus': CAP_CPUS, 'swap_bytes': 0, 'extra_GPUs': 0,
                          'selected_new_NAS_write_bytes': WRITER_BYTES, 'NAS_qualification_inside_writer_bytes': 536870912,
                          'ROOT_metadata_bound_bytes': 2*GIB, 'serial_NAS_model_writers': 1},
               'NAS_common_floor_comparison_bytes': COMMON_BYTES, 'inherited_common_padding_is_not_new_phase_jobs': True,
               'old_inputs_copied_bytes': 0, 'no_capability_claim': True, 'no_later_window_admission': True,
               'fresh_actual_native_runtime_observation_required_before_fit': True,
               'source_pinned_current_admission_required_before_allocation': True,
               'restore_original_exact_provider_configuration': True}
    out = args.output_directory.resolve()
    require(out.parent == HERE and not out.exists() and not out.is_symlink(), 'exclusive own plan output leaf required')
    raw_outputs = {'recovery_plan49_60.json': plan_raw, 'runtime_plan49_60.json': encode(runtime),
                   'ro_multi_origin_catalog.json': catalog_raw, 'ro_multi_origin_observation.json': observation_raw}
    out.mkdir(exist_ok=False)
    result = {name: write_raw(out/name, raw) for name, raw in raw_outputs.items()}
    print(json.dumps({'run_uuid': RUN, 'files': result, 'actual_runtime_admitted': False}, sort_keys=True))


def final_configs(args):
    spec, _ = read_ref({'path': str(args.inputs), 'sha256': args.inputs_sha256})
    require(spec['schema'] == 'neural_forge.k3_actual49_60_local_final_config_inputs.v1'
            and spec['run_uuid'] == RUN, 'exact post-allocation config input schema required')
    bundle = Path(spec['source_bundle']).resolve()
    require(bundle.parent == HERE and bundle.is_dir() and not bundle.is_symlink(), 'own source-bound local bundle required')
    closure = decode(read_raw(bundle/'SOURCE_CLOSURE49_60.json', spec['source_closure_sha256']))
    verify_bound_bundle(bundle, closure)
    require(closure['selected_run_uuid'] == RUN and len(closure['sources']) == 63 and len(closure['scripts']) == 5,
            'whole exact63 metadata/five-script source closure required')
    control = bundle/'production_control'; source_pins = {}
    for file in control.rglob('*'):
        require(not file.is_symlink(), 'source bundle symlink refused')
        if file.is_file(): source_pins[file.relative_to(control).as_posix()] = sha(file.read_bytes())
    for roster in (closure['sources'], closure['scripts']):
        for row in roster.values(): read_raw(control/row['name'], row['sha256'])
    identity = decode((control/'identity.json').read_bytes())
    require(sha((control/'identity.json').read_bytes()) == closure['sources']['identity']['sha256'],
            'byte-identical mounted and capacity identity required')
    required = {'allocation', 'recovery_plan', 'runtime_plan', 'catalog', 'observation'}
    require(set(spec['inputs']) == required, 'complete exact post-allocation current plan/observation inputs required')
    values, raws = {}, {}
    for role, ref in spec['inputs'].items(): values[role], raws[role] = read_ref(ref)
    allocation, plan, runtime = values['allocation'], values['recovery_plan'], values['runtime_plan']
    catalog, observation = values['catalog'], values['observation']
    mounts = decode((bundle/'mounts.json').read_bytes())
    closed = decode(read_raw(control/closure['sources']['closed48_continuity']['name'], identity['closed48_continuity_sha256']))
    observation_metadata(catalog, sha(raws['catalog']), observation, identity, mounts, closed)
    require(catalog['schema'] == 'chimera.k3_RO_multi_origin_prefix_fresh_NAS_output_source_route_review.v1'
            and observation['schema'] == 'chimera.k3_RO_multi_origin_prefix_separate_output_observation.v1'
            and plan['identity'] == runtime['identity'] == identity and plan['next_layers'] == '49-60'
            and plan['later_windows_admitted'] is False and plan['actual_fitter_started'] is False
            and plan['native_source_sha256'] == NATIVE_SHA and runtime['plan_sha256'] == sha(raws['recovery_plan'])
            and runtime['no_later_window_admission'] is True
            and plan['mounts'] == runtime['current_mounts'] == observation['current_mounts'] == mounts,
            'complete actual two-origin plan/catalog/observation/source joins differ')
    require(runtime['budget']['memory_bytes'] == CAP_BYTES and runtime['budget']['cpus'] == CAP_CPUS
            and runtime['budget']['swap_bytes'] == 0 and runtime['budget']['selected_new_NAS_write_bytes'] == WRITER_BYTES,
            '96GiB/22CPU/noSwap and whole152170243072 writer must be bound in current runtime plan')
    require(allocation['state'] == 'running' and allocation['owner'] == identity['owner']
            and allocation['name'] == identity['name'] and allocation['gpu_uuid'] == identity['gpu_uuid']
            and allocation['gpu_index'] == 6 and allocation['details']['image'] == identity['image']
            and allocation['details']['mounts'] == mounts and
            re.fullmatch(r'[0-9a-f]{64}', allocation['container_id']) is not None,
            'genuine normal new allocation/image/exact10mounts required')
    require(str(uuid.UUID(allocation['allocation_id'])) == allocation['allocation_id'], 'canonical actual allocation UUID required')
    flags = plan['RO_resume_source_flags']; refs = plan['RO_resume_source_refs']
    expected_names = {'catalog': 'ro_multi_origin_catalog.json', 'observation': 'ro_multi_origin_observation.json',
                      'recovery_plan': 'recovery_plan49_60.json', 'runtime_plan': 'runtime_plan49_60.json',
                      'allocation': 'actual_allocation.json'}
    require(len(flags) == 14 and plan['command_argv'][-14:] == flags and dict(zip(flags[::2], flags[1::2])) == {
        '--resume-source-manifest': '/control/'+expected_names['catalog'], '--resume-source-manifest-sha256': sha(raws['catalog']),
        '--resume-source-observation': '/control/'+expected_names['observation'], '--resume-source-observation-sha256': sha(raws['observation']),
        '--resume-source-identity': '/control/identity.json', '--resume-source-identity-sha256': source_pins['identity.json'],
        '--resume-source-root': '/resume-prefix'}, 'exact existing seven-flag/14-token caller ABI required')
    require(refs == {'manifest': {'path': '/control/'+expected_names['catalog'], 'sha256': sha(raws['catalog'])},
                     'observation': {'path': '/control/'+expected_names['observation'], 'sha256': sha(raws['observation'])},
                     'identity': {'path': '/control/identity.json', 'sha256': source_pins['identity.json']}},
            'complete current source references must match exact mounted byte pins')
    for role, name in expected_names.items(): source_pins[name] = sha(raws[role])
    sources = {role: leaf(row) for role, row in closure['sources'].items()}
    sources['allocation'] = {'name': expected_names['allocation'], 'sha256': sha(raws['allocation'])}
    scripts = {role: leaf(closure['scripts'][role]) for role in ('facts', 'preflight', 'service', 'math')}
    host_pin = leaf(closure['scripts']['host_caller'])
    host = {'schema': 'neural_forge.k3_capacity_host_loop_config.v1', 'run_uuid': RUN,
            'output_root': identity['output_root'], 'sources': sources, 'scripts': scripts,
            'host_caller_source_pin': host_pin, 'driver_config_name': 'continuation_config.json',
            'driver_sha256': source_pins['continuation_worker.py'], 'allocation_id': allocation['allocation_id'],
            'container_id': allocation['container_id']}
    host_raw = encode(host)
    driver = {'schema': 'neural_forge.k3_nas_closed48_worker_config.v1', 'identity': identity,
              'expected_allocator_name': allocation['name'], 'continuation_uuid': RUN,
              'window': [49,60], 'later_windows_admitted': False,
              'first_plan': {'path': SEALED36+'/inputs/original_record_plan.json', 'sha256': '17b4e172ac84c82dea494c7bef7e393f926a58e5fca67ece20358fb4e9a29a6e'},
              'baseline_receipt': {'path': '/recovery-source/ancestry/original_receipts/k3_dense_pass_receipt.json', 'sha256': 'c54e2fd0ade6fb8bbc9a111cb76fc5469abc1c197001b137e70f1c1156e976ff'},
              'prep_receipt': {'path': '/recovery-source/prep_result.json', 'sha256': 'bcc3876d5f79e84a5acaa90586ef9d03a84cee0f1781ac9f12cff17e902790ae'},
              'prior_continuity': {'path': SEALED48_MOUNTED+'/window_continuity.json', 'sha256': identity['closed48_continuity_sha256']},
              'closed36_disk_projection': {'path': '/control/closed36_next_resource_admission.json', 'sha256': '1e9004bc11daca688ab836e460517759d8b7d36c7f174d27748f1a38b987443a'},
              'source_pins': source_pins, 'native_tool_sha256': NATIVE_SHA,
              'budget': {'memory_limit_bytes': CAP_BYTES, 'cpus': CAP_CPUS, 'host_headroom_bytes': 16*GIB},
              'native_binding': identity['native_binding'], 'current_mounts': mounts, 'resume_argv': flags,
              'resume_source_manifest': refs['manifest'], 'resume_source_observation': refs['observation'],
              'resume_source_identity': refs['identity'], 'recovery_packet': refs['manifest'],
              'recovery_plan': {'path': '/control/'+expected_names['recovery_plan'], 'sha256': sha(raws['recovery_plan'])},
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'capacity_caller': {'host_caller_config_sha256': sha(host_raw), 'source_scripts': scripts, 'source_pins': sources}}
    out = args.output_directory.resolve()
    require(out.parent == HERE and not out.exists() and not out.is_symlink(), 'exclusive own final-config output leaf required')
    raw_outputs = {expected_names[role]: raw for role, raw in raws.items()}
    raw_outputs.update({'hostcapacity_caller_config.json': host_raw, 'continuation_config.json': encode(driver)})
    result = {'schema': 'neural_forge.k3_actual49_60_local_final_config_build.v1', 'run_uuid': RUN,
              'source_closure_sha256': spec['source_closure_sha256'], 'allocation_id': allocation['allocation_id'],
              'container_id': allocation['container_id'], 'files': {name: {'sha256': sha(raw), 'bytes': len(raw)} for name, raw in raw_outputs.items()},
              'actual_cgroup96GiB22CPU_noSwap_attestation_required_before_fit': True,
              'complete_source_freeze_and_fresh_owner_capacity_required': True, 'remote_or_runtime_actions': False}
    raw_outputs['FINAL_CONFIG_BUILD_RESULT.json'] = encode(result)
    out.mkdir(exist_ok=False)
    for name, raw in raw_outputs.items(): write_raw(out/name, raw)
    print(json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('sources', 'plans', 'configs'))
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    {'sources': source_bind, 'plans': plans_bind, 'configs': final_configs}[args.phase](args)


if __name__ == '__main__':
    main()
