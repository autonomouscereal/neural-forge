"""Owned host-side K3 recovery launch capacity; read-only host/model surfaces.

Called before the first normal own allocation and before every next window.
The caller stages exact metadata/source pins into one new own runtime control
leaf. This program calls the owned facts collector, observes host counters and
publishes a bounded exclusive packet. It neither allocates nor starts a job.
"""
import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

GIB=1<<30
NATIVE='01a11108-009c-72f3-b2ba-e0d48fca1435'

EXPECTED_RUN='__UNRESOLVED_NEXT49_60_RUN_UUID__'
IDENTITY_SHA='__UNRESOLVED_NEXT49_60_IDENTITY_SHA256__'
BINDING_SHA='__UNRESOLVED_NEXT49_60_BINDING_SHA256__'
CLOSED48_SHA='__UNRESOLVED_ACTUAL_CLOSED48_SHA256__'
FUTURE_SESSION_ID='__UNRESOLVED_GENUINE_SESSION_ID__'
FUTURE_GENERATION=None
FUTURE_CONTEXT_SHA='__UNRESOLVED_GENUINE_CONTEXT_SHA256__'
OWN_RAM_BYTES=None
OWN_CPU_THREADS=None
assert str(uuid.UUID(EXPECTED_RUN))==EXPECTED_RUN
assert EXPECTED_RUN!='764524b1-96a9-4636-be74-942f9c54754c'
assert str(uuid.UUID(FUTURE_SESSION_ID))==FUTURE_SESSION_ID
assert type(FUTURE_GENERATION) is int and FUTURE_GENERATION>=6
assert all(isinstance(x,str) and re.fullmatch(r'[0-9a-f]{64}',x) is not None for x in (IDENTITY_SHA,BINDING_SHA,CLOSED48_SHA,FUTURE_CONTEXT_SHA))
assert type(OWN_RAM_BYTES) is int and OWN_RAM_BYTES>0 and OWN_CPU_THREADS==22
WHOLE_WRITER_BYTES=152170243072
WRITER_SHA='16102e1523cbfd72c89df6e801554748e2601bae4a1d5ada081cfe22e85484a2'
CLOSED37_48_ROOT='/mnt/ai_data/omni-campaign/k3-nas-window37-48-oct07-764524b1-96a9-4636-be74-942f9c54754c/model'
RO_ROOTS={'legacy_closed36':'/mnt/lexar/omni-campaign/k3-diagnostic-recovery-oct07-048cff57-935c-4e2b-a31b-96a3c7fa6318','closed37_48':CLOSED37_48_ROOT}

RUN=EXPECTED_RUN
RUNTIME=Path('/home/cereal/omni-campaign/runtime/k3-nas-window49-60-oct07-'+RUN)
OUTPUT=Path('/home/cereal/omni-campaign/outputs/k3-nas-window49-60-oct07-'+RUN+'/control-output')
WORKING_OUTPUT=Path('/mnt/ai_data/omni-campaign/k3-nas-window49-60-oct07-'+RUN+'/model')
RESUME_SOURCE=Path('/mnt/lexar/omni-campaign/k3-diagnostic-recovery-oct07-048cff57-935c-4e2b-a31b-96a3c7fa6318')
CURRENT_SCOPE_SHA='__UNRESOLVED_NEXT49_60_SCOPE_SHA256__'
PROSPECTIVE_SUPERVISOR_SHA='e653c044aaaeaa01830fce60e0f57138e34a986ff1f3db970abecc779e944ad5'
FACTS_SHA='38e27e17c59bc1a6c5b1ed9dd55e3e7923330961534af54bed6ce739df01a0dd'
MATH_SHA='c448db00e09583f2f1e66e9ca32047fddb04a4f2f63d10a2fec61f1b0f7db2d2'
SERVICE_SHA='06ad5e26e88d66653e430055683309b6dbc83bf4d0d32b98fad4c6fcf7afe783'
POLICY_SHA='cd1593f9ef537ed9b9d0e56e96de7803f83f19e29e56c0afa9d04fc7d32daece'
ADOPTION_SHA='2032c2de59d841e233b5d0e4226114b31196cfb2153008045a53c350973f7d13'
PROPOSAL_SHA='131ae7ff05b7610bfeda9a6abb352599bbb0dcfcbe644d17374915f6d8c44238'
QVL_SHA='70a73a2140e9dd2eb6daa341492bc5bf2d21f31b8385830b838cdaa27e08d77b'
FUTURE_SHA='6e6f2e1f8f951125a6afb8d2a34fca85f47465d446c1095c5423503f967af593'
QIMG_SCOPE_SHA='6129837d4929380a5c59ffd51f2af0e5a54e54857247c9c4d620a0c716182a8b'
HY_SHA='5c039da8ba1d235fae74fd05109ab18196a7a821448578129696e36185e6f4e3'
H3_CPU_SHA='1bf2b94282ec58469149d76f5d5e2d720215183fc1371f493ce73c85ee5b0e06'
QIMG_OWNER='codex-native-01a111e4-2e50-7eb0-916a-74e1058b3a9c'
QIMG_ID='7c2af9e9114ed81a79aae4fc56aaeaa893e5ef4db73ca2a18e5d3dfeeb246336'
QIMG_ALLOCATION='1c1fed28-f501-4ea1-bbfb-b21c92ba2973'
OLD_K3_ID='cc405047d46114870c93b31ab732bbce8fe5f8c0b712cea8d12f38fa9cbfff4f'
PG_ID='fdad75f5106c8adc1da41036029c0f4e1a039b5120d46f2b7a40e79588e4cdb6'
PG_IMAGE='sha256:8ed3192326bb9d114cd5ef9acace453d5dae17425bd089d089330584c84c5a34'
FAMILY='ordinary_background_services_host_v1'
SCHEMA='neural_forge.k3_owned_capacity_preflight_request.v2'
ROOT_OWNER='codex-native-01a10f94-259b-7623-9679-ce4e68f82b8c'

SPEECH_SOURCE_PINS={'speech_completion': {'name': 'ACQUISITION_COMPLETE.json', 'sha256': '8936e30e0b76215169057b5935c120cf18f89121cd094122965b29a32537e1a3'}, 'speech_intent': {'name': 'OWN_INTENT_V1.json', 'sha256': 'fd933d4ee90a447b69946b680162252ba8c4bfa4ee542a5bcd3a44b7a1e3d929'}, 'speech_phase_sources': {'name': 'SPEECH_NATIVE_PHASE_SOURCE_READBACK_V1.json', 'sha256': 'e20c08d1efe95f2c5c25dc88aeade453bc9c59828570e459cee1bf15e8eda25a'}, 'speech_release': {'name': 'ACTUAL_NATURAL_CPU_RELEASE_LOCAL_AUDIT_V1.json', 'sha256': 'fb6ab1f1415c6a4dd94ac6497c4a0f19fb81122faa9b5f1d53d9551a379f99be'}, 'speech_release_proof_0': {'name': 'CONTROL_readback_ad59e033bd02461e9a3b3c3349676d06.json', 'sha256': '9fe3178aae2b0e1dd91bc2f8f80adb50fe31999685145d53319d4c6a84f0ddee'}, 'speech_release_proof_1': {'name': 'ACTUAL_IMMUTABLE_REQUEST_86396bd8.json', 'sha256': '8ce78472fb387bf004f32f0c18610e32c607a5c42df8130ada6f42383d55fc45'}, 'speech_release_proof_2': {'name': 'EXPORT_MANIFEST_V1.json', 'sha256': '2acca06986611c9115077aad102279d0117f76fb124e5b39b1545b098f200d7a'}, 'speech_release_proof_4': {'name': 'ACTUAL_SAME_OBJECT_START_LOCAL_AUDIT_V1.json', 'sha256': '90b1f6a47de46d078cec4912eecb87d3501f98fb53d695cace070767d2cd27e7'}, 'speech_release_proof_5': {'name': 'snapshot_contract_v1.json', 'sha256': '206b0e856fea25ba13e502cb7efdb4854345067eab7bd57ce96eea71a27d62fa'}, 'speech_runtime_contract': {'name': 'PHASE_CONTRACT.json', 'sha256': '2867bd1d9f6a0f969e1d360fc68c3bcfd9183680b1b5548ef3568a378a0ccf3f'}, 'speech_runtime_request': {'name': 'runtime-v1_REQUEST.json', 'sha256': '80d954b3c9642b182600b88b669b258570702a70800eb324a0c29bd673b78d32'}, 'speech_scope': {'name': 'SPEECH_NATIVE_FUTURE_SCOPE_V1.json', 'sha256': 'f79cb65a531e70517e846f44a3598d1a967d4a148426312fa91d3244b527c707'}}


def require(ok,message):
    if ok is not True:raise RuntimeError(message)


def integer(value,label):
    require(type(value) is int and value>=0,'nonnegative exact integer required: '+label)
    return value


def read(path,sha=None,limit=16<<20):
    require(path.is_absolute() and path.resolve()==path and path.is_file() and not path.is_symlink()
            and path.stat().st_size<=limit,'canonical bounded regular own metadata required')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) and len(raw)==before.st_size,
            'metadata changed during full-byte read')
    actual=hashlib.sha256(raw).hexdigest()
    if sha is not None:require(actual==sha,'exact metadata SHA differs: '+path.name)
    def unique(pairs):
        result={}
        for k,v in pairs:require(k not in result,'duplicate JSON key');result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda v:require(False,'nonfinite JSON')),actual


def code(path,sha,name):
    require(path.is_absolute() and path.resolve()==path and path.is_file() and not path.is_symlink()
            and path.is_relative_to(RUNTIME) and path.stat().st_size<=131072
            and hashlib.sha256(path.read_bytes()).hexdigest()==sha,'exact owned staged source required: '+name)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fresh(at):
    age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(at.replace('Z','+00:00'))).total_seconds()
    require(0<=age<=150,'current request/workload facts required within150seconds')


def host():
    roots={'controller':OUTPUT,'native_working':WORKING_OUTPUT,'readonly_resume':RESUME_SOURCE,'readonly_closed37_48':Path(CLOSED37_48_ROOT),'runtime':RUNTIME}
    require(all(path.is_dir() and path.resolve()==path and not path.is_symlink() for path in roots.values()),
        'exact canonical own NAS model/ROOT metadata/RO Lexar resume roots required')
    unescape=lambda value:re.sub(r'\\([0-7]{3})',lambda m:chr(int(m.group(1),8)),value)
    mounts=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left,right=line.split(' - ',1);fields,fs=left.split(),right.split()
        mounts.append({'mount_point':unescape(fields[4]),'major_minor':fields[2],
            'filesystem_type':fs[0],'source':unescape(fs[1]),'super_options':fs[2].split(',')})
    def covering(path):
        device=path.stat().st_dev
        major_minor=str(os.major(device))+':'+str(os.minor(device))
        rows=[r for r in mounts if path.is_relative_to(Path(r['mount_point'])) and r['major_minor']==major_minor]
        require(bool(rows),'actual covering mount required')
        return max(rows,key=lambda row:len(Path(row['mount_point']).parts))
    selected={key:covering(path) for key,path in roots.items()}
    nas=selected['native_working'];lexar=selected['readonly_resume'];root=selected['controller']
    require(nas['mount_point']=='/mnt/ai_data' and nas['filesystem_type']=='cifs'
        and nas['source']=='//192.168.50.250/ai' and WORKING_OUTPUT.stat().st_dev==162
        and 'nostrictsync' not in nas['super_options'],'exact supported existing NAS capacity domain required')
    require(lexar['mount_point']=='/mnt/lexar' and lexar['filesystem_type']=='exfat',
        'exact retained RO resume Lexar ExFAT capacity domain required')
    require(root==selected['runtime'] and root['mount_point']=='/'
        and OUTPUT.stat().st_dev==RUNTIME.stat().st_dev==Path('/').stat().st_dev,
        'own controller and staged source metadata must share existing ROOT capacity domain')
    require(selected['readonly_closed37_48']==nas and Path(CLOSED37_48_ROOT).stat().st_dev==WORKING_OUTPUT.stat().st_dev
        and WORKING_OUTPUT!=Path(CLOSED37_48_ROOT) and not WORKING_OUTPUT.is_relative_to(Path(CLOSED37_48_ROOT))
        and not Path(CLOSED37_48_ROOT).is_relative_to(WORKING_OUTPUT),'exact disjoint same-NAS RO source/current output capacity roots required')
    info={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1);parts=value.split();info[key]={'value':int(parts[0]),'unit':parts[1] if len(parts)==2 else ''}
    require(info['MemAvailable']['unit']=='kB','actual MemAvailable kB unit required')
    sizes={key:os.statvfs(path) for key,path in roots.items()}
    return {'at':dt.datetime.now(dt.timezone.utc).isoformat(),'MemAvailable_bytes':info['MemAvailable']['value']*1024,
        'disk_available_bytes':sizes['readonly_resume'].f_bavail*sizes['readonly_resume'].f_frsize,
        'NAS_available_bytes':sizes['native_working'].f_bavail*sizes['native_working'].f_frsize,
        'rootfs_available_bytes':sizes['controller'].f_bavail*sizes['controller'].f_frsize,
        'mount_identity':lexar,'native_working_mount_identity':nas,'controller_mount_identity':root,
        'all_output_mount_identities':selected,'full_meminfo':info,'host_logical_CPU_count':os.cpu_count(),
        'native_working_output_root':str(WORKING_OUTPUT),'controller_output_root':str(OUTPUT),
        'readonly_resume_source_roots':RO_ROOTS,
        'statvfs_by_role':{key:{'f_bavail':value.f_bavail,'f_frsize':value.f_frsize} for key,value in sizes.items()}}


"""Exact public owner deferral proof; no v3 adoption or reservation discharge."""
import hashlib
import json
import subprocess
from pathlib import Path


def backup_deferral_require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def backup_deferral_raw(row):
    raw = row['raw_utf8'].encode()
    backup_deferral_require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
                            'complete public backup deferral raw member differs')
    return raw


def backup_quiescent_deferral_source_proof(sources, request):
    expected = {'name': 'BACKUP_QUIESCENT_DEFERRAL_PUBLIC_SOURCE_COPY_V1.json',
                'sha256': 'ee13cd83893897974d72ad88b57e661cc8c34ebfbdac737a84f76c5bbdca13a5'}
    backup_deferral_require(request['sources']['backup_quiescent_deferral'] == expected,
                            'exact complete authentic backup deferral source packet required')
    packet = sources['backup_quiescent_deferral']
    backup_deferral_require(packet['schema'] == 'neural_forge.k3_exact_backup_quiescent_deferral_public_source_copy.v1'
                            and packet['read_only_public_metadata_copy'] is True
                            and packet['peer_adoption_or_runtime_actions'] is False,
                            'exact backup public source-copy schema required')
    receipt = json.loads(backup_deferral_raw(packet['receipt']))
    backup_deferral_require(packet['receipt']['sha256'] == '5fe231c4763baf6e066f1f1079140111ad4514f4b7dc6ba4672a395c08b1eda0'
                            and receipt['schema'] == 'exact2x.backup_deferred_quiescent.v1'
                            and receipt['state'] == 'DEFERRED_OWN_SOURCE_WORKER_STOPPED_CHECKPOINT_SEALED',
                            'authentic quiescent source deferral receipt required')
    members = packet['raw_members']
    backup_deferral_require(set(members) == set(receipt['artifacts']) and len(members) == 7,
                            'all seven actual owner deferral raw artifacts required')
    artifact_bytes = {}
    for name, row in members.items():
        backup_deferral_require(Path(row['path']).name == name
                                and row['sha256'] == receipt['artifacts'][name]['sha256']
                                and row['bytes'] == receipt['artifacts'][name]['bytes'],
                                'source-produced actual artifact pin differs')
        artifact_bytes[name] = backup_deferral_raw(row)
    tool = json.loads(backup_deferral_raw(packet['execution_tool_receipt']))
    backup_deferral_require(packet['execution_tool_receipt']['sha256'] == '4c87667bee7439e4cc5114e9cbe43f814f76433120a50164fbd1229290bf2f27'
                            and tool['exit_code'] == 0 and tool['stderr'] == '', 'actual owner deferral execution receipt required')
    recorded = [line[len('BACKUP_QUIESCENT_DEFERRAL_RECEIPT_JSON '):] for line in tool['stdout'].splitlines()
                if line.startswith('BACKUP_QUIESCENT_DEFERRAL_RECEIPT_JSON ')]
    backup_deferral_require(len(recorded) == 1 and json.loads(recorded[0]) == receipt,
                            'actual execution stdout and complete owner receipt differ')
    backup_deferral_raw(packet['execution_source'])
    backup_deferral_require(packet['execution_source']['sha256'] == '35155a329d2d845933059576d189ee303e7945b66d88c3a0479b1b5f183d90d2',
                            'exact source-owner deferral implementation required')
    authority = json.loads(artifact_bytes['deferral_authority.json'])
    foreign = sources['foreign_copy_worker']
    identity = receipt['producer_identity_before_stop']
    backup_deferral_require(authority['schema'] == 'exact2x.parent_authorized_backup_deferral.v1'
                            and authority['parent_thread'] == '01a0f883-9903-7346-ab6a-b73c780e3024'
                            and authority['producer_identity'] == identity
                            and identity['pid'] == foreign['pid'] == 2295182
                            and identity['start_ticks'] == foreign['start_ticks'] == 297731080
                            and identity['uid'] == 1000
                            and authority['execution_of_archive_or_handoff_retry_authorized'] is False
                            and authority['own_worker_checkpoint_stop_authorized_if_necessary'] is True
                            and authority['no_foreign_process_stop'] is True
                            and authority['no_original_or_diagnostic_deletion'] is True,
                            'exact genuine owner source-checkpoint deferral authority differs')
    backup_deferral_require(all(receipt[key] is True for key in ('producer_gone', 'hold_retained',
                                'source_metadata_consistent', 'archive_artifacts_absent'))
                            and all(receipt[key] is False for key in ('backup_execution_authorized', 'new_guard_activated',
                                'foreign_processes_changed', 'originals_removed', 'diagnostics_removed', 'unfinished_file_partial_state_reused'))
                            and receipt['backup_reservation_state'] == 'DEFERRED_UNAVAILABLE_FOR_EXECUTION'
                            and receipt['archive_bytes_written'] == receipt['reclaimed_bytes'] == receipt['other_backup_workers'] == 0
                            and receipt['future_backup_allowance_bytes'] == 132070244352,
                            'actual quiescent deferral preserves full backup reservation without reclamation')
    unit = receipt['old_unit']
    backup_deferral_require(unit['Id'] == 'codex-exact2x-backup-20261007-54c54ee2.service'
                            and unit['MainPID'] == '0' and unit['ActiveState'] == 'inactive' and unit['Restart'] == 'no',
                            'actual exact old backup unit must be inactive')
    checkpoint = json.loads(artifact_bytes['source_checkpoint_sealed.json'])
    module_raw = backup_deferral_raw(packet['checkpoint_module'])
    backup_deferral_require(packet['checkpoint_module']['sha256'] == '415726a236d6735c0569c035e3a4cac52f108000c1d02a9031177f25123a8d38',
                            'exact pure checkpoint validation source required')
    pure = {'__name__': 'k3_exact_public_checkpoint_integrity'}
    exec(compile(module_raw, '<full-pinned-checkpoint-source>', 'exec'), pure)
    # This checks the sealed historical metadata/journal integrity. It makes no
    # fresh source-payload or future restart admission claim.
    pure['validate_checkpoint'](checkpoint, artifact_bytes['source_metadata_before.json'],
                               artifact_bytes['source_events_sealed.jsonl'],
                               json.loads(artifact_bytes['source_metadata_before.json']), receipt['checkpoint_binding'])
    backup_deferral_require(checkpoint['completed_files'] == receipt['completed_files'] == 1019
                            and checkpoint['completed_bytes'] == receipt['completed_bytes'] == 129438938207
                            and checkpoint['binding']['worker_sha256'] == foreign['script_sha256']
                            and checkpoint['binding']['old_cmdline_sha256'] == identity['cmdline_sha256'],
                            'exact sealed source prefix or worker identity differs')
    own = sources['own_130GiB_scope']
    backup_deferral_require(request['sources']['own_130GiB_scope']['sha256'] == '913ba4b8dec778af7bd1e270c1f460f31a36ed97fb4c2caed7f8644c01d417f9'
                            and own['scope_accepted'] is True and own['run_uuid'] == sources['previous_phase_identity']['run_uuid']
                            and own['total_new_namespace_operational_budget_bytes'] == 139586437120
                            and own['common_backup_guard_prospective_bytes'] == 831613042688,
                            'genuine own single existing attempt130 scope required')
    return {'schema': 'exact2x.backup_deferred_quiescent.v1', 'receipt': packet['receipt']['sha256'],
            'actual_backup_dependency_reconciled': True, 'actual_backup_v3_transition_proved': False,
            'future_backup_allowance_bytes_preserved': 132070244352, 'new_guard_activated': False,
            'old_worker_pid': identity['pid'], 'old_worker_cgroup': foreign['cgroup_path'],
            'old_unit': unit['Id'], 'hold_identity': receipt['hold_identity'],
            'archive_attempt': receipt['checkpoint_binding']['attempt'], 'reclaimed_bytes': 0,
            'conservative_current_plus_future_NAS_common_guard_bytes_once': 831613042688,
            'future_backup_execution_authorized': False}


def backup_quiescent_deferral_current_absence(proof, facts):
    backup_deferral_require(not any(row['pid'] == proof['old_worker_pid']
                                   or row['cgroup_path'] == proof['old_worker_cgroup'] for row in facts['processes'])
                            and not Path('/proc', str(proof['old_worker_pid'])).exists(),
                            'actual deferred backup worker or unit member is current; fresh owner proof required')
    raw = subprocess.check_output(['systemctl', '--user', 'show', proof['old_unit'],
                                  '--property=Id,MainPID,ActiveState,Restart'], text=True, timeout=4)
    unit = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    backup_deferral_require(unit['Id'] == proof['old_unit'] and unit['MainPID'] == '0'
                            and unit['ActiveState'] == 'inactive' and unit['Restart'] == 'no',
                            'actual old backup unit resumed after quiescent deferral')
    audit = Path('/home/cereal/omni-campaign/runtime/disk-backup-exact2x-20261007T0418-approved-54c54ee2')
    hold = audit / 'OWNER_HOLD'
    st = hold.lstat()
    backup_deferral_require(not hold.is_symlink()
                            and (st.st_dev, st.st_ino) == (proof['hold_identity']['device'], proof['hold_identity']['inode']),
                            'genuine owner retained backup hold changed')
    backup_deferral_require(not any((Path(proof['archive_attempt']) / name).exists()
                                for name in ('cache.pax.tar.partial', 'cache.pax.tar', 'source_full_manifest.json')),
                            'deferred source-only backup archive artifacts appeared')
    return {'old_worker_absent': True, 'old_unit_inactive': True, 'owner_hold_identity_preserved': True,
            'archive_artifacts_absent': True, 'current_backup_RAM_bytes': 0, 'current_backup_CPU_threads': 0,
            'future_RAM_bytes_conservatively_preserved': 268435456, 'future_CPU_threads_conservatively_preserved': 1}


def current_NAS_scope(sources,request):
    require(request['sources']['NAS_current_scope']['sha256']==CURRENT_SCOPE_SHA,
        'exact selected current NAS resource source join required')
    scope=sources['NAS_current_scope']
    require(scope['schema']=='neural_forge.k3_NAS49_60.source_qualified_resource_scope.v1'
        and scope['run_uuid']==RUN and scope['layers']==[49,60], 'only selected generation6 window37-48 required')
    for role,expected in scope['source_pins'].items():
        require(request['sources'][role]==expected,'exact current source join differs: '+role)
    for row in scope['embedded_full_sources'].values():
        raw=row['source_utf8'].encode()
        require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],
            'complete current public producer/worker source bytes differ')
    require(request['sources']['previous_phase_identity']['sha256']=='d8bf5e9270930e0bdb4b470a07402791d16529c0c23546a31c9ec79e500b4997'
        and sources['previous_phase_identity']['run_uuid']=='764524b1-96a9-4636-be74-942f9c54754c','historical provider/backup producer identity must remain unchanged')
    require(request['sources']['closed48_continuity']['sha256']==CLOSED48_SHA
        and sources['closed48_continuity']['schema']=='neural_forge.k3_actual_window_continuity.v1'
        and sources['closed48_continuity']['completed_window']=='37-48'
        and sources['closed48_continuity']['full_prefix_layers']==48
        and sources['closed48_continuity']['ffn_full_scan_complete'] is True,'authentic full closed48 phase required before new49-60 admission')
    require(request['sources']['own49_60_writer_extent']['sha256']==WRITER_SHA
        and sources['own49_60_writer_extent']['full_single_attempt_NAS_writer_extent']['conservative_full_extent_bytes']==WHOLE_WRITER_BYTES,
        'complete source-derived new49-60 writer extent required')
    close=sources['QIMG_normal_fold_close'];terminal=sources['QIMG_fold_terminal']
    require(close['schema']=='chimera.qimg_first32.gpu_phase_terminal_validation_pending_closure.v1'
        and close['owner']==QIMG_OWNER and close['phase']=='fold' and close['container_id']==QIMG_ID
        and close['allocation_id']==QIMG_ALLOCATION and close['actual_original_release_group_verified'] is True
        and close['actual_original_PG_transaction_committed'] is True
        and close['fresh_committed_released_row']['state']=='released'
        and close['actual_physical_exited_object_preserved'] is True
        and close['CPU1933_validation_accepted'] is False and close['native_FIRST32_accepted'] is False
        and close['proof_pins']['gpu_terminal']['sha256']==request['sources']['QIMG_fold_terminal']['sha256'],
        'authentic original fold-only normal preserved close required')
    require(terminal['schema']=='chimera.qimg_first32.actual_fold_native_terminal_local_readback.v1'
        and terminal['owner']==QIMG_OWNER and terminal['phase']=='fold'
        and terminal['container_id']==QIMG_ID and terminal['allocation_id']==QIMG_ALLOCATION
        and terminal['actual_completed'] is True and terminal['actual_terminal_state_verified_prepost'] is True,
        'authentic completed fold only source proof required')
    qvl=sources['QVL_CPU_request'];caps=qvl['caps']
    require(qvl['schema']=='chimera.qvl32.root_native_CPU_lifecycle_request.v1'
        and qvl['owner']==ROOT_OWNER and qvl['PREPARED_TEMPLATE_ONLY_NOT_ADMITTED'] is True
        and (caps['memory_bytes'],caps['memory_swap_bytes'],caps['cpu_threads'],caps['output_reserve_bytes'])==(12*GIB,12*GIB,4,2*GIB)
        and caps['gpu_device_requests']==[] and qvl['output_root'].startswith('/home/cereal/omni-campaign/runtime/qvl32-cpu-'),
        'actual source-backed pending ROOT QVL CPU12/4/2 prepared scope required')
    approved=sources['approved_NAS_scope']
    require(approved['schema']=='chimera.runtime_admission.NAS_EXISTING_COMMITMENT_SCOPE_REVIEW.v2'
        and (approved['approved_exact2x_archive_full_allowance'],approved['VXUG_revised_cap_bytes'],approved['VXUG_free_floor'],approved['approved_scope_required_bytes'])
            ==(132070244352,8*GIB,512*GIB,690415992832)
        and approved['current_QIMG_phase_NAS_exports']==0 and approved['historical128_fulfillment_or_discharge_not_claimed'] is True,
        'current approved NAS backup/VX/floor source scope differs')
    root=sources['QIMG_ROOT_metadata']
    require(root['schema']=='chimera.qimg_first32.own_ROOTFS_metadata_future_adoption.v1'
        and root['owner']==QIMG_OWNER and root['root_policy_adopted'] is True
        and root['reservation_bytes']==2*GIB and root['filesystem']=='rootfs' and root['accounting']=='additional',
        'source-backed original ROOT QIMG metadata future required')
    sequence=sources['own_provider_sequence'];normal=sources['own_provider_hold_receipt']
    require(sequence['schema']=='chimera.qimg_first32.future_sequence_hold.v1'
        and sequence['acting_identity']==sources['previous_phase_identity'] and sequence['future_owner']=='codex-native-'+NATIVE
        and sequence['promise_id']=='restored_exact_GPU6_future_restoration_full_hardware_v1'
        and sequence['provider_exact_original_id']=='e4bff2d8b50cce1fda75d50b1b3def27ebecb50e1a5c02e17150f84b66873d31'
        and sequence['actual_quiesced_hold_receipt_sha256']==request['sources']['own_provider_hold_receipt']['sha256']
        and sequence['release_before_future_execution'] is True and sequence['normal_terminal_and_release_required_before_restoration'] is True
        and sequence['reservation_edits'] is False and sequence['foreign_job_interruption'] is False
        and sequence['future_restoration_obligation_preserved'] is True and sequence['sequence_does_not_admit_future_resources'] is True
        and sequence['binding_generation']==6 and sequence['coordination_only'] is True and sequence['execution_authority'] is False,
        'actual own exact full-provider restoration source sequence must remain nonoverlapping and preserved')
    result=normal['result']
    require(normal['exit_code']==0 and result['ok'] is True and result['action']=='pause'
        and result['native_identity']==sources['previous_phase_identity']
        and result['target_original']['id']==sequence['provider_exact_original_id']
        and result['target_original']['config_sha256']=='9e472748c7e31b9550ad0fd0b4c21e035ca27732ef3417a57939390ef2f6fb64'
        and result['target_original']['host_config_sha256']=='ea1c3117a65fafaf08eba2094f3b60e2bb6a59988c376f4299f4c4fc9f153352'
        and result['target_original']['mounts_sha256']=='72334510dbead96e73bc6a38e5860cf0cf937d8e71f20e96de28371d9d4a5146'
        and result['target_original']['registry_sha256']=='1cf516407220cc3465755616012e06f97ea4349e1a7f1fa23c6f6defe1a91ea7'
        and all(row['running']==row['pending']==0 for row in result['submission_fence']['queues']),
        'authentic normal exact own original GPU6 quiesced hold required; fresh native observer rechecks actual fingerprints')
    require(scope['actual_backup_dependency_reconciled'] is True
        and scope['backup_dependency_reconciliation_schema']=='exact2x.backup_deferred_quiescent.v1'
        and scope['actual_backup_v3_transition_proved'] is False
        and scope['NAS_current_common_guard_bytes_once']==844196848640
        and scope['NAS_common_guard_base_excluding_K3_bytes']==692026605568
        and scope['historical_37_48_K3_total130GiB_bytes']==139586437120
        and scope['NAS_common_guard_current_K3_writer_bytes_once']==WHOLE_WRITER_BYTES
        and scope['actual_new_guard_activated'] is False
        and scope['backup_reservation_bytes_preserved']==132070244352,
        'exact genuine quiescent deferral and full common130 source accounting required')
    request['backup_dependency_reconciliation']=backup_quiescent_deferral_source_proof(sources,request)
    return scope


def own_readonly_call_chain(facts,request):
    """Bind this source-executing PID and its exact own SSH/shell ancestor chain.

    No session/cgroup descendant exemption. A reviewed ssh.service endpoint is
    required, and only the actual exact ancestor PIDs are returned for this
    observation. Arbitrary siblings remain unclassified.
    """
    processes={row['pid']:row for row in facts['processes']}
    pid=os.getpid();require(pid in processes,'actual source-executing caller absent from full facts')
    myself=processes[pid]
    require(myself['comm'] in ('python','python3','python3.11','python3.12','python3.13','python3.14')
        and myself['exe']==str(Path(sys.executable).resolve()),'exact own current Python caller executable differs')
    source=Path(__file__).resolve();source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    own_group=myself['cgroup_path'];chain=[];seen=set()
    host_caller=request.get('owned_host_caller')
    if host_caller is not None:
        caller_source=Path(host_caller['source_path'])
        require(caller_source.is_absolute() and caller_source.is_relative_to(RUNTIME)
            and caller_source.resolve()==caller_source and not caller_source.is_symlink()
            and caller_source.name=='k3_capacity_host_loop_v1.py' and caller_source.stat().st_size<=65536
            and hashlib.sha256(caller_source.read_bytes()).hexdigest()==host_caller['source_sha256'],
            'exact owned host caller source provenance required')
        caller_config,caller_config_sha=read(Path(host_caller['config_path']),host_caller['config_sha256'])
        require(caller_config['schema']=='neural_forge.k3_capacity_host_loop_config.v1' and caller_config['run_uuid']==RUN
            and caller_config['host_caller_source_pin']['sha256']==host_caller['source_sha256']
            and caller_config['scripts']['preflight']['sha256']==source_sha,
            'actual own host caller/preflight immutable source pins differ')
        require(request['mode']=='allocated' and caller_config['sources']==request['sources']
            and caller_config['allocation_id']==request['current_own_allocation']['allocation_id']
            and caller_config['container_id']==request['current_own_allocation']['container_id'],
            'actual own detached caller must bind complete sources and current normal allocation')
    supervisor=request.get('owned_prospective_supervisor')
    if supervisor is not None:
        require(request['mode']=='prospective' and host_caller is None,'exact own prospective readonly supervisor only')
        supervisor_source=Path(supervisor['source_path'])
        require(supervisor_source.is_absolute() and supervisor_source.is_relative_to(RUNTIME/'capacity_calls')
            and supervisor_source.resolve()==supervisor_source and not supervisor_source.is_symlink()
            and supervisor_source.name=='prospective_supervisor.py' and supervisor_source.stat().st_size<=65536
            and supervisor['source_sha256']==PROSPECTIVE_SUPERVISOR_SHA
            and hashlib.sha256(supervisor_source.read_bytes()).hexdigest()==PROSPECTIVE_SUPERVISOR_SHA,
            'exact source-pinned own prospective supervisor required')
        config_path=Path(supervisor['config_path'])
        require(config_path.parent==supervisor_source.parent and config_path.name=='prospective-background-config.json',
            'exact same own prospective call config required')
        config,_=read(config_path,supervisor['config_sha256'])
        require(config['schema']=='neural_forge.k3_NAS37_48.prospective_background_config.v1'
            and config['run_uuid']==RUN and config['sources']==request['sources']
            and config['preflight_pin']=={'name':source.name,'sha256':source_sha}
            and config['supervisor_source_pin']=={'name':supervisor_source.name,'sha256':PROSPECTIVE_SUPERVISOR_SHA}
            and config['request_pin']==request['prospective_original_request_pin'],
            'complete pinned prospective source/config/request boundary differs')
        original,_=read(supervisor_source.parent/config['request_pin']['name'],config['request_pin']['sha256'])
        require(original['mode']=='prospective' and original['sources']==request['sources']
            and original['request_boundary']==request['request_boundary']
            and original['own_remaining_write_bytes']==request['own_remaining_write_bytes']==103035350992
            and original['remaining_copy_bytes']==request['remaining_copy_bytes']==0,
            'actual original exclusive prospective request source differs')
    if own_group=='/sys/fs/cgroup/system.slice/ssh.service':
        return {'source':{'path':str(source),'sha256':source_sha},'self_pid':os.getpid(),
            'exact_process_bindings':[],'self_binding':{k:myself[k] for k in ('pid','start_ticks','ppid','comm','exe','cgroup_path')},
            'adopted_endpoint':'exact ssh.service cgroup already classified by original source policy',
            'shared_session_or_ancestry_wildcard':False,'model_payload_reads':0}
    while pid in processes and pid not in seen and len(chain)<8:
        row=processes[pid];seen.add(pid)
        if row['cgroup_path']=='/sys/fs/cgroup/system.slice/ssh.service':
            require(row['comm']=='sshd','exact adopted ssh.service endpoint must be sshd')
            return {'source':{'path':str(source),'sha256':source_sha},'self_pid':os.getpid(),
                'exact_process_bindings':chain,'adopted_endpoint':{k:row[k] for k in ('pid','start_ticks','ppid','comm','cgroup_path')},
                'shared_session_or_ancestry_wildcard':False,'model_payload_reads':0}
        require(row['cgroup_path']==own_group,'own readonly SSH caller ancestors changed cgroup')
        if pid!=os.getpid():
            if supervisor is not None and pid==supervisor['pid']:
                require(all(row[k]==supervisor[k] for k in ('pid','start_ticks','ppid','comm','exe','cgroup_path'))
                    and row['ppid']==1 and row['exe']==str(Path(sys.executable).resolve())
                    and row['comm'] in ('python','python3','python3.11','python3.12','python3.13','python3.14'),
                    'exact detached own prospective supervisor birth/executable differs')
                chain.append({k:row[k] for k in ('pid','start_ticks','ppid','comm','exe','cgroup_path')})
                return {'source':{'path':str(source),'sha256':source_sha},'self_pid':os.getpid(),
                    'exact_process_bindings':chain,'adopted_endpoint':'exact own detached source-pinned readonly prospective supervisor',
                    'shared_session_or_ancestry_wildcard':False,'model_payload_reads':0}
            elif host_caller is not None and pid==host_caller['pid']:
                require(all(row[k]==host_caller[k] for k in ('pid','start_ticks','ppid','comm','exe','cgroup_path')),
                    'exact own pinned host caller birth/executable differs')
                require(row['comm'] in ('python','python3','python3.11','python3.12','python3.13','python3.14')
                    and row['exe']==str(Path(sys.executable).resolve()),'actual pinned own host caller executable differs')
                if row['ppid']==1:
                    chain.append({k:row[k] for k in ('pid','start_ticks','ppid','comm','exe','cgroup_path')})
                    return {'source':{'path':str(source),'sha256':source_sha},'self_pid':os.getpid(),
                        'exact_process_bindings':chain,'adopted_endpoint':'exact own detached host caller with pinned source/config/current allocation',
                        'shared_session_or_ancestry_wildcard':False,'model_payload_reads':0}
            else:
                require(row['comm'] in ('sshd','bash','sh','dash'),'unknown own host caller ancestor requires source proof')
            if row['comm'] not in ('sshd','python','python3','python3.11','python3.12','python3.13','python3.14'):
                require(row['exe'] in ('/usr/bin/bash','/bin/bash','/usr/bin/dash','/bin/dash'),
                    'exact own caller shell executable differs')
        chain.append({k:row[k] for k in ('pid','start_ticks','ppid','comm','exe','cgroup_path')})
        pid=row['ppid']
    raise RuntimeError('bounded exact own SSH caller chain did not reach adopted ssh.service')



def current_tts_proof(sources, request):
    pin = {'name':'TTS_ACTUAL_PUBLIC_SOURCE_COPY_V1.json',
           'sha256':'1777c5e091ff9189df5fb3a8fce0f6fe92cdffdb4f6e5565cd071dbdd5aa2697'}
    require(request['sources'].get('TTS_actual_public_source') == pin,
            'exact complete actual TTS public source packet required')
    packet = sources['TTS_actual_public_source']
    require(packet['schema']=='neural_forge.k3_exact_TTS_public_source_copy.v1'
            and packet['read_only_public_metadata_copy'] is True
            and packet['peer_adoption_or_runtime_actions'] is False, 'exact TTS source-copy scope required')
    manifest_raw = packet['manifest_raw_utf8'].encode('utf-8')
    require(hashlib.sha256(manifest_raw).hexdigest()=='fac30997d3437be6b76ba45654543837e1e3741a9514b6f1e048d39b4b823626',
            'actual owner TTS manifest full SHA differs')
    manifest = json.loads(manifest_raw)
    raw_files = packet['raw_files_utf8']; entries = manifest['raw_files']; groups = {}
    require(len(entries)==manifest['raw_file_count']==79 and len(raw_files)==79
            and set(raw_files)=={entry['name'] for entry in entries}, 'complete exact TTS raw roster required')
    for entry in entries:
        raw = raw_files[entry['name']].encode('utf-8')
        require(len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],
                'actual TTS raw member hash differs')
        groups.setdefault(entry['role'], []).append(raw)
    require(sum(len(raw.encode('utf-8')) for raw in raw_files.values())==manifest['raw_bytes']==4256371,
            'complete TTS raw member bytes differ')
    # Repeated output-metadata basenames are authentic; every consumed proof role is unique.
    roles = {role:raws[0] for role,raws in groups.items() if len(raws)==1}
    def body(role):
        require(role in roles, 'exact unique actual TTS proof role required')
        return json.loads(roles[role])
    tts = body('immutable_server_request.json'); normal = body('NORMAL_CPU_ADMISSION.json')
    locked = body('locked_admission.json'); created = body('created.json'); start = body('actual_start.json')
    public = body('actual_Docker_Config_secret_suppressed_HostConfig_Mounts_full.json')
    contract = body('phase_contract.json'); scope = sources['speech_scope']
    canonical = lambda value: hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    semantic = canonical(tts)
    require(tts==body('immutable_local_request.json') and semantic=='a80b5d1d9c2d9d23068fb448d420b67201caa5c7c0fe25cbdef2723005e3d0e7',
            'actual immutable TTS request/semantic hash differs')
    require(tts['schema']=='chimera.speech_native.cpu_request.v1' and tts['native']==scope['native']
            and tts['owner']==scope['owner'] and tts['run_uuid']==scope['run_uuid']
            and tts['phase_kind']=='tts' and tts['phase_id']=='tts-v1'
            and tts['cpu_job_uuid']=='629f6553-2503-4529-a5ca-db309acd2a2b'
            and tts['binding']['harness_id']==scope['native'] and tts['binding']['session_generation']==2
            and tts['binding']['session_id']=='3cccb64a-1ae0-4a69-b1f2-47e00f41d565'
            and tts['binding']['context_sha256']=='15d33489c6ea0a155de9c98847ff6b6188a97cfa8cb066467c9af54a77402e83'
            and tts['binding']['transport']=='live' and tts['binding']==manifest['historical_START_binding_preserved'],
            'exact actual TTS g2 native/owner/run/phase/job binding differs')
    require(tts['runtime_root']=='/home/cereal/omni-campaign/runtime/speech-native-01a11531-c0bf-74f0-9ec5-d7d7072790b8-880527b5-9d0f-4ba4-a8e0-f83aef34123e'
            and tts['code_root']==tts['runtime_root']+'/code/tts-v1'
            and tts['output_root']==tts['runtime_root']+'/output/tts-v1'
            and tts['source_root']==sources['speech_runtime_request']['source_root']
            and (tts['memory_cap_bytes'],tts['cpu_threads'],tts['campaign_cap_bytes'],tts['runtime_metadata_cap_bytes'],tts['output_cap_bytes'])
                ==(24*GIB,2,18*GIB,2*GIB,8*GIB)
            and tts['image']==sources['speech_runtime_request']['image']
            and tts['container_name']=='speech-native-'+tts['cpu_job_uuid'], 'exact TTS roots/image/finite scope differs')
    lifecycle = packet['lifecycle_source_utf8'].encode('utf-8')
    require(len(lifecycle)==118459 and hashlib.sha256(lifecycle).hexdigest()==tts['lifecycle_pin']['sha256']
            =='450b87fdb3574ae84f45d14a94b042ca1cbb9fed48e2cc072f771c46ae547a5f'
            and tts['lifecycle_pin']['path']==tts['runtime_root']+'/control/remote_cpu_lifecycle_v6.py',
            'complete actual TTS source-control bytes differ')
    code = {role[5:]:raw for role,raw in roles.items() if role.startswith('code_')}
    code['PHASE_CONTRACT.json'] = roles['phase_contract.json']
    require(set(code)==set(tts['code_files']) and len(code)==45 and manifest['code_files']==tts['code_files'],
            'complete actual TTS declared code roster differs')
    for name, raw in code.items():
        require(len(raw)==tts['code_files'][name]['bytes'] and hashlib.sha256(raw).hexdigest()==tts['code_files'][name]['sha256'],
                'complete TTS physical code source hash differs')
    require(contract==json.loads(code['PHASE_CONTRACT.json']) and contract['phase_kind']=='tts'
            and contract['phase_id']=='tts-v1' and contract['native']==scope['native']
            and contract['command_argv']==tts['command_argv'] and contract['source_guard_required'] is True
            and contract['source_completion_pin']['sha256']==request['sources']['speech_completion']['sha256']
            and hashlib.sha256(roles['phase_contract.json']).hexdigest()==tts['phase_contract_pin']['sha256']
            and body('source_completion_pin.json')==sources['speech_completion'], 'TTS phase/source22 completion contract differs')
    require(normal['schema']=='chimera.speech_native.normal_cpu_admission.v1' and normal['approved'] is True
            and normal['native']==scope['native'] and normal['owner']==scope['owner']
            and normal['run_uuid']==scope['run_uuid'] and normal['session_id']==tts['binding']['session_id']
            and normal['phase_kind']=='tts' and normal['phase_id']=='tts-v1' and normal['cpu_job_uuid']==tts['cpu_job_uuid']
            and normal['immutable_request_sha256']==locked['request_sha256']==created['request_sha256']==semantic
            and normal['no_swap'] is True and normal['no_GPU'] is True
            and normal['normal_PG_transaction_and_advisory_lock_held'] is True
            and normal['normal_remote_mutation_locks_held'] is True
            and normal['GPU_allocator_allocation'] is False and normal['foreign_actions'] is False
            and normal['final_fresh_admission']['admitted'] is True and locked['admission']['admitted'] is True,
            'authentic normal TTS locked admission/request join differs')
    for obj in (locked, created):
        require(obj['native']==scope['native'] and obj['owner']==scope['owner'] and obj['phase_id']=='tts-v1',
                'actual TTS lock/create owner differs')
    require(created['cpu_job_uuid']==tts['cpu_job_uuid'] and normal['container_id']==created['container_id']
            ==start['actual']['container_id']==public['container_id']==manifest['container_id']
            =='3243dbc689e6f0e04c420efb92c574c24a6a8b25c49b93a7dec7a30f0897b000'
            and start['native']==scope['native'] and start['phase_id']=='tts-v1'
            and start['foreign_actions'] is False and start['GPU_allocator_allocation'] is False,
            'actual exact TTS create/START container differs')
    for pin, role in ((normal['locked_admission_pin'],'locked_admission.json'),
                      (normal['actual_created_container_pin'],'created.json'),
                      (start['normal_worker_admission_pin'],'NORMAL_CPU_ADMISSION.json')):
        require(pin['bytes']==len(roles[role]) and pin['sha256']==hashlib.sha256(roles[role]).hexdigest(),
                'complete referenced TTS lock/create/normal proof pin differs')
    require(normal['locked_admission_pin']==created['locked_admission_pin']==start['locked_admission_pin']
            and normal['actual_created_container_pin']==start['created_pin'], 'TTS exact referenced proof paths differ')
    guard = start['actual']; host = public['HostConfig']; cfg = public['Config']
    for field in ('Config_sha256','normalized_HostConfig_sha256','Mounts_sha256'):
        require(guard[field]==created['guard'][field]==public['actual_normal_contract'][field]
                and guard[field] is not None, 'actual START/create/public TTS fingerprints differ')
    require(public['actual_Config_sha256']==guard['Config_sha256']
            and canonical({**host,'OomKillDisable':False})==guard['normalized_HostConfig_sha256']
            and host.get('OomKillDisable') in (None,False)
            and canonical(sorted(public['Mounts'],key=lambda item:json.dumps(item,sort_keys=True)))==guard['Mounts_sha256'],
            'complete public TTS Config/Host/Mount fingerprints differ')
    require(public['Path']==tts['command_argv'][0] and public['Args']==tts['command_argv'][1:]
            and cfg['User']=='1000:1000' and cfg['Entrypoint']==[tts['command_argv'][0]] and cfg['Cmd']==tts['command_argv'][1:]
            and public['Image']==tts['image'] and public['Name']=='/'+tts['container_name'], 'actual TTS image/argv differs')
    require(host['Memory']==host['MemorySwap']==24*GIB and host['NanoCpus']==2_000_000_000
            and host['Runtime']=='runc' and host['NetworkMode']=='none' and host['ReadonlyRootfs'] is True
            and host['PidsLimit']==256 and host['RestartPolicy']['Name']=='no'
            and not host.get('DeviceRequests') and not host.get('Devices') and not host.get('Privileged'),
            'actual TTS finite24/2/noSwap/noGPU/networknone contract differs')
    expected = [{'source':tts['code_root'],'target':'/speech-code','read_only':True},
                {'source':tts['source_root'],'target':'/speech-source','read_only':True},
                {'source':tts['runtime_root']+'/input','target':'/speech-input','read_only':True},
                {'source':tts['runtime_root']+'/runtime','target':'/speech-runtime','read_only':False},
                {'source':tts['output_root'],'target':'/speech-output','read_only':False}]
    actual = [{'source':item['Source'],'target':item['Destination'],'read_only':not item['RW']}
              for item in public['Mounts'] if item['Type']=='bind']
    require(sorted(actual,key=lambda item:item['target'])==sorted(expected,key=lambda item:item['target'])
            and all((item['Type']=='bind' and item.get('Propagation')=='rprivate')
                    or (item['Type']=='tmpfs' and item['Destination']=='/tmp') for item in public['Mounts']),
            'actual complete five readonly/write TTS mounts differ')
    return {'request':tts,'guard':guard,'public':public,'code':code,
            'Path_Args_sha256':canonical({'Path':public['Path'],'Args':public['Args']})}


def current_tts_holder(row, proof):
    tts = proof['request']; guard = proof['guard']; limits = row['resource_limits']; labels = row['identity_labels']
    require(row['id']==guard['container_id'] and row['pid']==guard['State']['Pid']
            and row['started_at']==guard['State']['StartedAt'] and row['name']=='/'+tts['container_name']
            and row['image_id']==tts['image'] and labels.get('ai.local.owner')==tts['owner']
            and labels.get('chimera.native.session')==tts['native'] and labels.get('chimera.run.uuid')==tts['run_uuid']
            and labels.get('chimera.cpu.job_uuid')==tts['cpu_job_uuid'] and labels.get('chimera.speech.phase')=='tts-v1'
            and 'ai.local.allocation' not in labels, 'exact current actual TTS owner/job/container birth differs')
    fingerprints = row['known_cpu_peer_fingerprints']
    require(all(fingerprints[field]==guard[field] for field in ('Config_sha256','normalized_HostConfig_sha256','Mounts_sha256'))
            and fingerprints['Path_Args_sha256']==proof['Path_Args_sha256'], 'fresh actual TTS source/argv/mount fingerprint differs')
    require(limits['Memory']==limits['MemorySwap']==24*GIB and limits['NanoCpus']==2_000_000_000
            and limits['Runtime']=='runc' and not limits['DeviceRequests']
            and row['cgroup_path']==guard['cgroup_counters']['path'], 'fresh actual TTS finite cap/cgroup differs')
    return 24*GIB,2


def current_tts_public_code(proof):
    # Public code/control only, no model/audio payload or imports of peer code.
    tts = proof['request']; root = Path(tts['code_root'])
    require(root.is_absolute() and root.resolve()==root and root.is_dir() and not root.is_symlink(),
            'exact current public TTS code root required')
    entries = list(root.rglob('*'))
    require(len(entries)<=256 and not any(path.is_symlink() for path in entries)
            and {path.relative_to(root).as_posix() for path in entries if path.is_file()}==set(tts['code_files']),
            'current exact closed public TTS code roster differs')
    for name,pin in tts['code_files'].items():
        path = root/name; before = path.stat(); raw = path.read_bytes(); after = path.stat()
        require(path.resolve()==path and before.st_size==len(raw)==pin['bytes'] and len(raw)<=1<<20
                and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
                and hashlib.sha256(raw).hexdigest()==pin['sha256'], 'current public TTS physical code source changed')
    path = Path(tts['lifecycle_pin']['path']); before = path.stat(); raw = path.read_bytes(); after = path.stat()
    require(path.resolve()==path and not path.is_symlink() and len(raw)==118459
            and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
            and hashlib.sha256(raw).hexdigest()==tts['lifecycle_pin']['sha256'], 'current public TTS source-control changed')
    group = Path(proof['guard']['cgroup_counters']['path']); stat = group.stat()
    require(group.resolve()==group and (stat.st_dev,stat.st_ino)==
            (proof['guard']['cgroup_counters']['device'],proof['guard']['cgroup_counters']['inode']),
            'actual current TTS kernel cgroup birth differs')


def speech_terminal_proof(sources, request, actual_tts):
    pin = {'name':'SPEECH_TERMINAL_PUBLIC_SOURCE_COPY_V1.json',
           'sha256':'965e4f295f872a782922f4798a9c93b66b1a7925a1e3bc6f95fc6fd95c0fe154'}
    require(request['sources'].get('speech_terminal_public_source')==pin,
            'complete authenticated current speech terminal source packet required')
    packet=sources['speech_terminal_public_source']
    require(packet['schema']=='neural_forge.k3_exact_speech_terminal_public_source_copy.v1'
            and packet['read_only_public_metadata_copy'] is True
            and packet['peer_adoption_or_runtime_actions'] is False
            and packet['new_execution_authority_claimed'] is False,'speech terminal source-copy scope differs')
    raw=packet['manifest_raw_utf8'].encode('utf-8')
    require(len(raw)==78076 and hashlib.sha256(raw).hexdigest()==
            '7e1dc8cc45147294f5c8508471cc1806bcd95d1c4b09ece0c87c20567352619f',
            'exact completed actual speech public manifest differs')
    manifest=json.loads(raw);entries=manifest['raw_files'];files=packet['raw_files_utf8'];groups={}
    require(len(entries)==manifest['raw_file_count']==len(files)==82
            and set(files)=={entry['name'] for entry in entries},'complete82 speech terminal member roster required')
    for entry in entries:
        raw=files[entry['name']].encode('utf-8')
        require(len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],
                'complete speech terminal raw member hash differs')
        groups.setdefault(entry['role'],[]).append(raw)
    require(sum(len(raw.encode('utf-8')) for raw in files.values())==manifest['raw_bytes']==4916597,
            'complete speech terminal raw bytes differ')
    def raw_role(role):
        require(role in groups and len(groups[role])==1,'unique exact speech terminal proof role required')
        return groups[role][0]
    tts=actual_tts['request'];guard=actual_tts['guard']
    require(json.loads(raw_role('immutable_server_request.json'))==tts
            and json.loads(raw_role('actual_start.json'))['actual']==guard
            and manifest['native']==tts['native'] and manifest['cpu_job_uuid']==tts['cpu_job_uuid']
            and manifest['container_id']==guard['container_id']
            and manifest['historical_START_binding_preserved']==tts['binding'],
            'complete speech terminal actual request/START/owner/job differs')
    completion_raw=raw_role('output_metadata_complete_native_fold_receipt.json')
    require(len(completion_raw)==654584 and hashlib.sha256(completion_raw).hexdigest()==
            'a3d24973def026436dc9d971e8ad0667fbf9de026c37797bf2a0448494e64a94',
            'complete actual donor TTS completion source differs')
    completion=json.loads(completion_raw)
    require(completion['status']=='complete_physical_source_component_fold_with_strict_native_replay'
            and len(completion['folds'])==33 and completion['teacher_native_replays']==2
            and completion['strict_child_native_replays']==2 and completion['strict_complete_state_reload'] is True
            and completion['source_component_only'] is True
            and completion['one_final_shared_multimodal_checkpoint_completed'] is False,
            'actual completed donor TTS scope differs')
    raw=packet['normal_release_raw_utf8'].encode('utf-8')
    require(len(raw)==33287 and hashlib.sha256(raw).hexdigest()==
            'f6858acede0d79ad26f911f4ca02bdd66b9840bcdae726c498ee3e4c77f96aa1',
            'exact actual normal speech release full SHA differs')
    release=json.loads(raw);released=release['actual'];state=released['State']
    require(release['native']==tts['native'] and release['owner']==tts['owner']
            and release['phase_id']==tts['phase_id'] and release['cpu_job_uuid']==tts['cpu_job_uuid']
            and released['container_id']==guard['container_id'] and state==manifest['fresh_actual_State']
            and state['StartedAt']==guard['State']['StartedAt']
            and state['Status']=='exited' and state['Running'] is False and state['Paused'] is False
            and state['Pid']==state['ExitCode']==0 and state['OOMKilled'] is False
            and state['FinishedAt']=='2026-10-07T12:38:59.329116986Z'
            and release['at']=='2026-10-07T12:46:28.337010+00:00'
            and release['normal_CPU_release']['terminal_resources_returned_by_kernel'] is True
            and release['physical_container_retained'] is True and release['resource_actions'] is False
            and release['foreign_actions'] is False and release['GPU_allocator_allocation'] is False
            and release['normal_CPU_release']['container_or_results_deleted'] is False,
            'exact donor TTS terminal normal release identity/resources differ')
    for field in ('Config_sha256','normalized_HostConfig_sha256','Mounts_sha256'):
        require(released[field]==guard[field]==manifest['actual_normal_contract'][field],
                'terminal normal release/source fingerprints differ')
    artifacts={entry['path']:entry for entry in release['output_artifacts']}
    completion_pin=artifacts[tts['output_root']+'/native_fold/complete_native_fold_receipt.json']
    require(completion_pin['bytes']==len(completion_raw)
            and completion_pin['sha256']==hashlib.sha256(completion_raw).hexdigest()
            and release['campaign_usage']['retained_objects_deleted'] is False,
            'normal release must retain/join exact complete donor output')
    raw=packet['typed_owner_phase_event_raw_utf8'].encode('utf-8')
    require(len(raw)==6380 and hashlib.sha256(raw).hexdigest()==
            '2b7b1c4fc81ff0fd8de944a8c2f79e88b2e30a5f4997f38b888bfcb9da09ae06',
            'exact authenticated current owner speech future statement required')
    typed=json.loads(raw);events=[]
    def find(value):
        if isinstance(value,dict):
            if value.get('event_id')==116780:events.append(value)
            for member in value.values():find(member)
        elif isinstance(value,list):
            for member in value:find(member)
    find(typed)
    require(len(events)==1 and typed['execution_authority'] is False
            and typed['foreign_private_inboxes_read'] is False,'exact public owner event scope differs')
    event=events[0];message=event['message']
    require(event['execution_authority'] is False and event['channel']['kind']=='project'
            and event['channel']['channel_key']=='chimera-omni' and message['type']=='STATUS'
            and message['sender_session_id']==tts['binding']['session_id']
            and message['payload']['execution_authority'] is False
            and 'Both speech model phases now terminal/currentCPU-RAM0, finished donor-fold futureCPU-RAM-write0; no further model phase planned in this milestone.' in message['body']
            and tts['native'] in message['body']
            and 'f6858acede0d79ad26f911f4ca02bdd66b9840bcdae726c498ee3e4c77f96aa1' in message['body']
            and '7e1dc8cc45147294f5c8508471cc1806bcd95d1c4b09ece0c87c20567352619f' in message['body'],
            'authenticated owner model-milestone future0 statement differs')
    return {'event_id':116780,'donor_model_milestone_future_RAM_bytes':0,
            'donor_model_milestone_future_CPU_threads':0,'ROOT_reserve_bytes_once':20*GIB,
            'raw_shared_multimodal_core_completion_claimed':False}



def speech_scope(sources, request, proofs, p):
    for role, expected in SPEECH_SOURCE_PINS.items():
        require(request['sources'][role] == expected, 'exact actual speech source pin differs: ' + role)
    scope=sources['speech_scope'];intent=sources['speech_intent'];runtime=sources['speech_runtime_request']
    completion=sources['speech_completion'];release=sources['speech_release'];contract=sources['speech_runtime_contract']
    require(scope['schema']=='neural_forge.speech_native.current_future_scope.v1'
        and scope['native']=='01a11531-c0bf-74f0-9ec5-d7d7072790b8'
        and scope['owner']=='codex-native-'+scope['native'] and scope['run_uuid']=='880527b5-9d0f-4ba4-a8e0-f83aef34123e'
        and scope['cpu_job_uuid']=='43d4239d-eb2f-4b89-bb78-439729b16f71'
        and scope['sequential_runtime_STT_TTS_phases'] is True
        and (scope['remaining_native_peak_RAM_bytes'],scope['CPU_threads_once'],scope['root_write_reserve_bytes_once'])==(24*GIB,2,20*GIB)
        and (scope['campaign_root_write_reserve_bytes'],scope['runtime_root_write_reserve_bytes'],scope['phase_output_cap_bytes'])==(18*GIB,2*GIB,8*GIB)
        and scope['full_native_speech_future_discharged'] is False,
        'source-backed sequential native speech future scope differs')
    require(all(scope[key]==0 for key in ('completed_acquisition_future_RAM_bytes','completed_acquisition_future_CPU_threads',
        'completed_acquisition_remaining_write_bytes','retained_acquisition_disk_credited_bytes')), 'acquisition-only zero and retained disk required')
    for role, source_pin in scope['source_pins'].items():
        require(source_pin==request['sources'][role], 'complete speech scope source join differs')
    require(completion['status']=='complete' and completion['complete_asset_count']==completion['expected_asset_count']==22
        and completion['payload_source_bytes']==completion['expected_source_bytes']==8_607_865_275
        and len(completion['asset_manifest'])==22 and all(r['status']=='complete' for r in completion['asset_manifest'])
        and sum(r['size'] for r in completion['asset_manifest'])==8_607_865_275
        and completion['contract_sha256']=='206b0e856fea25ba13e502cb7efdb4854345067eab7bd57ce96eea71a27d62fa'
        and completion['native_folding_done'] is False and completion['shared_core_integration_done'] is False,
        'actual complete22 speech source acquisition only required')
    require(release['schema']=='chimera.speech_source.actual_natural_CPU_release_local_audit.v1'
        and release['status']=='passed' and release['native']=='01a114aa-94c4-7572-bc5b-a995982555b4',
        'actual authenticated acquisition-only natural release required')
    p.resolve_proof([{'name':Path(row['path']).name,'sha256':row['sha256']} for row in release['evidence']],proofs)
    terminal=release['natural_release']
    require(terminal['container_id']=='b7d2382f9e20cb10ba090c0a4d4f058cb96fdfdcaf342080cdf909007b773bdf'
        and terminal['cpu_job_uuid']=='b5e2d966-a15f-45cd-ab3e-0583cbdae0b2'
        and terminal['FinishedAt']=='2026-10-07T07:10:50.924912989Z'
        and terminal['Status']=='exited' and terminal['Running'] is False and terminal['Pid']==terminal['ExitCode']==0
        and terminal['OOMKilled'] is False and terminal['kernel_process_CPU_RAM_resources_returned'] is True
        and terminal['own_completed_acquisition_future_RAM_bytes']==terminal['own_completed_acquisition_future_CPU_threads']==0
        and terminal['container_and_source_outputs_retained'] is True
        and terminal['foreign_release_reconcile_or_provider_mutation'] is False
        and release['retained_disk_accounting']['retained_disk_remains_charged'] is True,
        'exact naturally released completed acquisition identity differs')
    require(intent['schema']=='chimera.speech_native.own_intent.v1' and intent['native']==scope['native']
        and intent['owner']==scope['owner'] and intent['run_uuid']==scope['run_uuid']
        and (intent['memory_cap_bytes'],intent['cpu_threads'],intent['campaign_retained_future_bytes'],intent['runtime_metadata_bytes'])==(24*GIB,2,18*GIB,2*GIB)
        and intent['no_GPU'] is True, 'exact new native speech prospective intent differs')
    require(runtime['schema']=='chimera.speech_native.cpu_request.v1' and runtime['native']==scope['native']
        and runtime['owner']==scope['owner'] and runtime['run_uuid']==scope['run_uuid']
        and runtime['binding']==intent['binding'] and runtime['runtime_root']==intent['runtime_root']
        and runtime['cpu_job_uuid']==scope['cpu_job_uuid'] and runtime['phase_kind']=='runtime' and runtime['phase_id']=='runtime-v1'
        and (runtime['memory_cap_bytes'],runtime['cpu_threads'],runtime['campaign_cap_bytes'],runtime['runtime_metadata_cap_bytes'])==(2*GIB,2,18*GIB,2*GIB)
        and runtime['container_name']=='speech-native-'+scope['cpu_job_uuid']
        and runtime['image']=='sha256:5070872ac7aef716eadc60ec92b0f944521c7899d90c6899a7b1ed1e883942e1'
        and runtime['source_root']=='/home/cereal/omni-campaign/runtime/speech-src-01a114aa-9772e4d5-d09b-4c54-9650-699586a9529d/output/snapshots'
        and runtime['phase_contract_pin']['sha256']==request['sources']['speech_runtime_contract']['sha256'],
        'exact prospective new native runtime request/source/job differs')
    require(contract['schema']=='chimera.speech_native.phase_contract.v1' and contract['native']==scope['native']
        and contract['phase_id']=='runtime-v1' and contract['source_completion_pin']['sha256']==request['sources']['speech_completion']['sha256']
        and (contract['output_cap_bytes'],contract['campaign_cap_bytes'],contract['runtime_metadata_cap_bytes'])==(8*GIB,18*GIB,2*GIB),
        'actual runtime source completion/phase contract join differs')
    phase_sources=sources['speech_phase_sources']
    require(phase_sources['source_execution_performed'] is False
        and set(phase_sources['sources'])=={'build_phase_request.py','remote_cpu_lifecycle.py'}, 'actual two public phase sources required')
    for row in phase_sources['sources'].values():
        raw=row['source_utf8'].encode('utf-8')
        require(len(raw)==row['pin']['bytes'] and hashlib.sha256(raw).hexdigest()==row['pin']['sha256'], 'complete public phase source bytes differ')
    return scope


def classify(facts,policy,proposal,adoption,proofs,p,service,request):
    active,outside,_=p.membership(facts)
    by_id={row['id']:row for row in facts['containers']};processes={row['pid']:row for row in facts['processes']}
    protected=service.validate_facts(facts)
    require(adoption['root_policy_adopted'] is True and adoption['accounting_id']==FAMILY
        and (adoption['memory_bound_bytes'],adoption['cpu_threads_bound'])==(32*GIB,48),'exact full adopted ordinary family required')
    require(len(proposal['service_roster'])==38,'exact38 physical ordinary service source identities required')
    rows=[]
    for known in proposal['service_roster']:
        if known['id']==service.CLAMD and known['id'] not in by_id:
            require(protected['protected_restarting_service'] is True,'only exact protected clamd may restart/PID0')
            continue
        row=by_id.get(known['id'])
        require(row is not None and row['name'].lstrip('/')==known['name'].lstrip('/')
            and row['image_id']==known['image_id'],'ordinary physical service identity drift requires reclassification')
        rows.append(dict({k:row[k] for k in ('id','pid','cgroup_path')},classification='established_service',accounting_id=FAMILY))
    require(OLD_K3_ID not in by_id,'failed historical K3 must not be a live recovery peer')
    qimg=by_id.get(QIMG_ID);qimg_active=qimg is not None
    require(not qimg_active,'completed normal-released QIMG fold holder must remain inactive')
    current_jobs=[]
    if qimg_active:
        require(qimg['identity_labels'].get('ai.local.owner')==QIMG_OWNER
            and qimg['identity_labels'].get('ai.local.allocation')==QIMG_ALLOCATION,
            'exact active current QIMG holder required; source roster changed')
        rows.append(dict({k:qimg[k] for k in ('id','pid','cgroup_path')},classification='enforced_foreign_job',accounting_id='QIMG_current_48GiB_8CPU'))
        current_jobs.append(('QIMG_current_48GiB_8CPU',qimg,48*GIB,8))
    else:
        terminal=request['loaded_sources']['QIMG_current_terminal']
        require(request['sources']['QIMG_current_terminal']['sha256']=='2322ca8d350fedf7e9c38e019f4e22ec27a44738ad8ed9a8d5c5458578333c72'
            and terminal['schema']=='chimera.qimg_first32.actual_recovery_fold_watch_status.v1'
            and terminal['owner']==QIMG_OWNER and terminal['allocation_id']==QIMG_ALLOCATION
            and terminal['container_id']==QIMG_ID and terminal['resource_actions']==0,
            'exact public current QIMG terminal metadata required')
        observation=terminal['latest_observation'];state=observation['physical_model_state']
        require(observation['terminal_observed'] is True and observation['allocation_id']==QIMG_ALLOCATION
            and observation['container_id']==QIMG_ID and state['Running'] is False and state['Pid']==0
            and state['Status']=='exited' and state['FinishedAt']=='2026-10-07T06:19:32.974942749Z'
            and not any(r['identity_labels'].get('ai.local.owner')==QIMG_OWNER for r in facts['containers']),
            'current QIMG terminal holder absence requires fresh complete no-successor frame')
    request['QIMG_current_holder_active']=qimg_active
    qvl_active=False
    if 'QVL_current_allocation' in request['loaded_sources']:
        allocation=request['loaded_sources']['QVL_current_allocation']
        row=by_id.get(allocation['container_id'])
        require(row is not None and allocation['owner']==ROOT_OWNER and allocation['state']=='running'
            and allocation['server']=='ai' and allocation['kind']=='generic' and allocation['gpu_index']==4
            and row['identity_labels'].get('ai.local.owner')==ROOT_OWNER
            and row['identity_labels'].get('ai.local.allocation')==allocation['allocation_id']
            and row['identity_labels'].get('ai.local.gpu-uuid')==allocation['gpu_uuid'],
            'exact raw normal root QVL allocation/current physical holder required')
        devices=row['resource_limits']['DeviceRequests']
        require(len(devices)==1 and devices[0]['DeviceIDs']==[allocation['gpu_uuid']]
            and devices[0]['Capabilities']==[['gpu']],'exact single root QVL GPU4 request required')
        rows.append(dict({k:row[k] for k in ('id','pid','cgroup_path')},classification='enforced_foreign_job',accounting_id='QVL_current_16GiB_4CPU'))
        current_jobs.append(('QVL_current_16GiB_4CPU',row,16*GIB,4));qvl_active=True

    speech=request['loaded_sources']['speech_scope'];runtime=request['loaded_sources']['speech_runtime_request']
    require('b7d2382f9e20cb10ba090c0a4d4f058cb96fdfdcaf342080cdf909007b773bdf' not in by_id,
        'completed acquisition holder must remain absent from fresh active frame')
    speech_rows=[row for row in facts['containers'] if row['identity_labels'].get('ai.local.owner')==speech['owner']]
    require(len(speech_rows)<=1,'new native speech phases must remain serial and use reviewed exact current request')
    speech_active=bool(speech_rows)
    tts_proof=current_tts_proof(request['loaded_sources'],request)
    terminal_speech=speech_terminal_proof(request['loaded_sources'],request,tts_proof)
    require(not speech_active,'owner-released completed speech scope requires no active speech holder; new phase needs exact new source proof')
    request['speech_native_terminal_future_closed']=True
    speech_cap=speech_cpu=0
    if speech_active:
        row=speech_rows[0]
        speech_cap,speech_cpu=current_tts_holder(row,tts_proof)
        current_tts_public_code(tts_proof)
        rows.append(dict({k:row[k] for k in ('id','pid','cgroup_path')},classification='enforced_foreign_job',accounting_id='speech_native_TTS_24GiB_2CPU'))
        current_jobs.append(('speech_native_TTS_24GiB_2CPU',row,speech_cap,speech_cpu))
    request['speech_native_current_runtime_active']=speech_active
    request['speech_native_current_cap_bytes']=speech_cap
    request['speech_native_current_CPU_threads']=speech_cpu
    pg=by_id.get(PG_ID)
    if pg is not None:
        require(pg['name']=='/embeddinggemma2-pg-task3-20261006' and pg['image_id']==PG_IMAGE
            and not pg['resource_limits']['DeviceRequests'],'exact qualification PG identity required')
        rows.append(dict({k:pg[k] for k in ('id','pid','cgroup_path')},classification='enforced_foreign_job',accounting_id='qualification_PG'))
        current_jobs.append(('qualification_PG',pg,512<<20,1))
    own=[r for r in facts['containers'] if r['identity_labels'].get('ai.local.owner')=='codex-native-'+NATIVE]
    if request['mode']=='prospective':require(not own,'prospective recovery cannot omit a live own worker')
    else:
        allocation=request['current_own_allocation']
        require(len(own)==1 and allocation['owner']=='codex-native-'+NATIVE and allocation['state']=='running'
            and allocation['server']=='ai' and allocation['kind']=='generic'
            and own[0]['id']==allocation['container_id']
            and own[0]['identity_labels'].get('ai.local.allocation')==allocation['allocation_id']
            and own[0]['identity_labels'].get('ai.local.gpu-uuid')==request['identity']['gpu_uuid']
            and own[0]['image_id']==request['identity']['image'] and own[0]['name']=='/'+request['identity']['name'],
            'exact selected normal committed own recovery allocation required')
        rows.append(dict({k:own[0][k] for k in ('id','pid','cgroup_path')},classification='enforced_foreign_job',accounting_id='K3_next49_60_source_funded_cap'))
        current_jobs.append(('K3_next49_60_source_funded_cap',own[0],OWN_RAM_BYTES,OWN_CPU_THREADS))
    require(sorted([{k:r[k] for k in ('id','pid','cgroup_path')} for r in rows],key=lambda r:r['id'])==active,
        'unknown current Docker workload refuses this source-backed scope')
    # Preserve reviewed exact groups. Absent exact process anchors have no current
    # member. PID1/2 are rebased only as the existing reviewed builder does.
    rules=copy.deepcopy(policy['outside_process_groups']);live_rules=[]
    plumbing={r['pid'] for r in facts['fixed_containerd_plumbing_processes']}
    for rule in rules:
        if rule['scope'] in ('process_identity','process_ancestry'):
            anchor=rule['anchor'];live=processes.get(anchor['pid'])
            if live is None:continue
            if anchor['pid'] in plumbing:continue
            if anchor['pid'] in (1,2):
                require(live['comm'] in ('systemd','kthreadd'),'known init/kernel ancestry differs')
                anchor['start_ticks']=live['start_ticks']
        live_rules.append(rule)
    # Current shim rows have full pre/post exact birth and executable proof from
    # the unchanged service helper; they remain inside the funded32/48 family.
    for shim in facts['fixed_containerd_plumbing_processes']:
        if any(rule['scope']=='exact_cgroup' and rule['cgroup_path']==shim['cgroup_path'] for rule in live_rules):continue
        live_rules.append({'scope':'process_identity','anchor':{k:shim[k] for k in ('pid','start_ticks')},
            'classification':'established_service','accounting_id':FAMILY,'source_evidence_pins':request['ordinary_source_pins']})
    foreign=request['loaded_sources']['foreign_copy_worker']
    foreign_members=[r for r in facts['processes'] if r['cgroup_path']==foreign['cgroup_path']]
    if foreign_members:
        require(len(foreign_members)==1 and all(foreign_members[0][k]==foreign[k]
            for k in ('pid','start_ticks','ppid','comm','exe','cgroup_path')),
            'exact source-attributed foreign copy worker birth differs')
        for path_key,sha_key in (('unit_path','unit_sha256'),('script_path','script_sha256')):
            file=Path(foreign[path_key])
            require(file.is_file() and not file.is_symlink() and file.stat().st_size<=65536
                and hashlib.sha256(file.read_bytes()).hexdigest()==foreign[sha_key],
                'exact public foreign unit/script source pin changed')
        approval,_=read(Path(foreign['public_approval_path']),foreign['public_approval_sha256'])
        binding,_=read(Path(foreign['corrected_source_binding_path']),foreign['corrected_source_binding_sha256'])
        require(approval['destination']==foreign['destination'] and approval['limits']['MemoryMax_bytes']==268435456
            and approval['limits']['CPUQuota_pct']==100 and approval['reviewed_archive_allowance_bytes']==132070244352
            and approval['reviewed_NAS_promises_bytes']==881907614096 and approval['source_removal_authorized'] is False
            and binding['worker_sha256']==foreign['script_sha256'] and binding['original_approval_sha256']==foreign['public_approval_sha256'],
            'actual public source/bounds/NAS-only writer promise differs')
        live_rules.append({'scope':'exact_cgroup','cgroup_path':foreign['cgroup_path'],
            'classification':'enforced_foreign_job','accounting_id':'exact_foreign_copy_worker_256MiB_1CPU',
            'source_evidence_pins':[request['sources']['foreign_copy_worker']]})
    caller=request['own_readonly_call_chain']
    for binding in caller['exact_process_bindings']:
        already=[rule for rule in live_rules if rule['scope']=='process_identity' and rule['anchor']['pid']==binding['pid']]
        require(not already,'own readonly caller overlaps an existing dated exact process identity')
        live_rules.append({'scope':'process_identity','anchor':{k:binding[k] for k in ('pid','start_ticks')},
            'classification':'established_service','accounting_id':FAMILY,'source_evidence_pins':request['ordinary_source_pins']})
    vxug=request['loaded_sources']['VXUG_validator']
    require(vxug['schema']=='neural_forge.k3_exact_VXUG_validator_planning_scope.v1'
        and vxug['planning_RAM_bytes']==114144128 and vxug['source_CPU_threads']==1
        and vxug['NAS_new_metadata_bound_bytes']==4951768
        and vxug['planning_envelope_is_enforced_max'] is False,'exact source-informed validator planning scope required')
    vxug_process=processes.get(vxug['pid']);vxug_active=vxug_process is not None
    if vxug_active:
        require(all(vxug_process[k]==vxug[k] for k in ('pid','start_ticks','ppid','comm','exe','cgroup_path'))
            and vxug_process['threads']==1 and vxug_process['RSS_bytes']<=vxug['planning_RAM_bytes'],
            'exact validator birth/source shape or current RAM exceeds reviewed planning scope')
        members=[r for r in facts['processes'] if r['cgroup_path']==vxug['cgroup_path']]
        require(len(members)==1 and members[0]['pid']==vxug['pid'],'validator exact cgroup membership changed')
        script=Path(vxug['script_path'])
        require(script.is_file() and not script.is_symlink() and script.stat().st_size==vxug['script_bytes']
            and hashlib.sha256(script.read_bytes()).hexdigest()==vxug['script_sha256'],'validator exact source bytes changed')
        snapshot=vxug['source_metadata_snapshot'];tree_pin=snapshot['tree_pin']
        read(Path(tree_pin['path']),tree_pin['sha256'])
        provenance,_=read(Path(snapshot['provenance_pin']['path']),limit=4096)
        require(provenance['blob_bytes']==snapshot['blob_bytes_sum'],
            'validator relevant source blob context changed; final provenance status rewrite is permitted')
        outside_for_policy=[r for r in outside if r['pid']!=vxug['pid']]
    else:outside_for_policy=outside
    outside_rows=p.outside_classifications(live_rules,outside_for_policy,facts,proofs)
    if vxug_active:
        outside_rows.append(dict({k:vxug_process[k] for k in ('pid','start_ticks','cgroup_path')},
            classification='source_informed_foreign_validator',accounting_id='exact_VXUG_validator_source_planning',
            source_evidence_pins=[request['sources']['VXUG_validator']],enforced_max_claimed=False))
    require(sorted([{k:r[k] for k in ('pid','start_ticks','cgroup_path')} for r in outside_rows],key=lambda r:r['pid'])==outside,
        'complete exact outside PID accounting coverage must remain once')
    request['VXUG_validator_current_active']=vxug_active
    require(all((r['classification']=='established_service' and r['accounting_id']==FAMILY) or
        (r['classification']=='enforced_foreign_job' and r['accounting_id']=='exact_foreign_copy_worker_256MiB_1CPU') or
        (r['classification']=='source_informed_foreign_validator' and r['accounting_id']=='exact_VXUG_validator_source_planning') for r in outside_rows),
        'unknown or additional outside job requires exact current scope funding')
    groups={g['cgroup_path']:g for g in facts['cgroups']}
    require(len(groups)==len(facts['cgroups']) and not facts['vanished_process_cgroups'],'complete actual cgroups required')
    growth=[];paths=[];qual_ram=qual_cpu=0
    for key,row,ram,cpus in current_jobs:
        g=groups[row['cgroup_path']];measured=p.growth(g)
        require(measured['cap_bytes']==ram and p.quota_threads(g['cpu.max'])==cpus
            and g['memory.swap.max']==g['memory.swap.current']=='0','actual full finite cap/quota/noSwap differs: '+key)
        require(row['resource_limits']['Memory']==row['resource_limits']['MemorySwap']==ram,
            'Docker enforced memory/swap cap differs')
        paths.append(p.canonical_group(row['cgroup_path']));growth.append(dict(measured,accounting_id=key,cpu_threads=cpus))
        if key.startswith('qualification_'):qual_ram+=ram;qual_cpu+=cpus
    require(len(set(paths))==len(paths) and all(not a.is_relative_to(b) for a in paths for b in paths if a!=b),'duplicate enforced cgroup charge')
    if foreign_members:
        foreign_group=groups[foreign['cgroup_path']];measured=p.growth(foreign_group)
        require(measured['cap_bytes']==268435456 and p.quota_threads(foreign_group['cpu.max'])==1
            and foreign_group['memory.swap.max']==foreign_group['memory.swap.current']=='0',
            'actual finite foreign copy worker256MiB1CPUnoSwap differs')
        growth.append(dict(measured,accounting_id='exact_foreign_copy_worker_256MiB_1CPU',cpu_threads=1))
    if vxug_active:
        g=groups[vxug['cgroup_path']];current=integer(int(g['memory.current']),'actual validator memory.current')
        stat=g['memory.stat']
        require(isinstance(stat,dict) and {'file','shmem','file_dirty','file_writeback','unevictable'}<=set(stat),
            'complete actual validator memory.stat required')
        for key,value in stat.items():integer(value,'validator memory.stat.'+key)
        credit=min(current,max(0,stat['file']-stat['shmem']-stat['file_dirty']-stat['file_writeback']-stat['unevictable']))
        nonreclaimable=current-credit
        require(nonreclaimable<=vxug['planning_RAM_bytes'],'actual validator memory exceeds source-informed planning envelope')
        growth.append({'cgroup_path':vxug['cgroup_path'],'current_bytes':current,'reclaimable_file_bytes':credit,
            'nonreclaimable_current_bytes':nonreclaimable,'remaining_growth_reserved_bytes':max(0,vxug['planning_RAM_bytes']-nonreclaimable),
            'planning_envelope_bytes':vxug['planning_RAM_bytes'],'enforced_max_claimed':False,
            'actual_memory_max':g['memory.max'],'cpu_max':g['cpu.max'],'full_memory_stat':stat,
            'accounting_id':'exact_VXUG_validator_source_planning','cpu_threads':1})
    return rows,outside_rows,growth,qual_ram,qual_cpu,qvl_active


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args();request,_=read(args.request,args.request_sha256)
    control=args.request.parent
    require(control.is_relative_to(RUNTIME) and control.is_dir() and control.resolve()==control,'existing canonical own runtime control leaf required')
    require(request['schema']==SCHEMA and request['mode'] in ('prospective','allocated'),'closed selected request schema/mode required')
    fresh(request['at']);attempt=str(uuid.UUID(request['attempt_uuid']));require(attempt==request['attempt_uuid'],'canonical new attempt UUID required')
    sources={};proofs={};total=0
    for role,row in request['sources'].items():
        require(Path(row['name']).name==row['name'] and row['name'].endswith('.json'),'bounded staged metadata leaf required')
        body,sha=read(control/row['name'],row['sha256']);total+=(control/row['name']).stat().st_size
        require(total<=64<<20 and len(sources)<128,'bounded source metadata roster required')
        sources[role]=body;proofs[(row['name'],sha)]=body
    for role,pin in [('policy',POLICY_SHA),('adoption',ADOPTION_SHA),('proposal',PROPOSAL_SHA),('HY_zero',HY_SHA),
                     ('H3_CPU_zero',H3_CPU_SHA),('QVL',QVL_SHA),('source_future',FUTURE_SHA),('QIMG_scope',QIMG_SCOPE_SHA)]:
        require(request['sources'][role]['sha256']==pin,'exact reviewed source bytes required: '+role)
    identity=sources['identity'];binding=sources['binding']
    require(identity['run_uuid']==RUN and identity['native_thread_id']==NATIVE and identity['runtime_root']==str(RUNTIME)
        and identity['output_root']==str(OUTPUT) and identity['native_binding']==binding
        and identity['native_working_output_root']==str(WORKING_OUTPUT)
        and identity['controller_output_root']==str(OUTPUT)
        and identity['readonly_resume_source_roots']==RO_ROOTS and identity['closed48_continuity_sha256']==CLOSED48_SHA
        and identity['resume_source_mounted_read_only'] is True
        and identity['new_layers']==[49,60] and identity['only_this_window_scope'] is True
        and request['sources']['identity']['sha256']==IDENTITY_SHA and request['sources']['binding']['sha256']==BINDING_SHA
        and binding['session_id']==FUTURE_SESSION_ID and binding['session_generation']==FUTURE_GENERATION and binding['context_sha256']==FUTURE_CONTEXT_SHA
        and identity['native_working_copy_mounted_read_only'] is False
        and identity['original_failed_source_mounted_read_only'] is True
        and identity['checkpoint_retention']=='--snapshot-keep 0: preserve all existing and new native checkpoints'
        and 'readonly_saved_prefix_root' not in identity,'selected genuine own recovery identity/current binding required')
    request['identity']=identity
    request['loaded_sources']=sources
    if request['mode']=='allocated':request['current_own_allocation']=sources['allocation']
    else:require('allocation' not in sources,'prospective packet has no actual own allocation')
    p=code(control/request['math_source_name'],MATH_SHA,'k3_frozen_pure_capacity_math')
    service=code(control/request['service_source_name'],SERVICE_SHA,'k3_frozen_service_observer')
    speech=speech_scope(sources,request,proofs,p)
    selected=current_NAS_scope(sources,request)
    collector=control/request['facts_source_name']
    require(collector.is_file() and not collector.is_symlink() and collector.resolve()==collector
        and hashlib.sha256(collector.read_bytes()).hexdigest()==FACTS_SHA,'exact selected own facts collector required')
    argv=[sys.executable,'-B',str(collector),'--identity',str(control/request['sources']['identity']['name']),
        '--identity-sha256',request['sources']['identity']['sha256'],'--binding',str(control/request['sources']['binding']['name']),
        '--binding-sha256',request['sources']['binding']['sha256'],'--observation-uuid',attempt,
        '--output-directory',str(control),'--service-helper',str(control/request['service_source_name'])]
    completed=subprocess.run(argv,capture_output=True,text=True,timeout=65)
    require(completed.returncode==0,'owned readonly facts collector failed: '+completed.stderr[-2048:])
    ref=json.loads(completed.stdout);facts_path=Path(ref['path'])
    require(facts_path==control/('foreign_classification_facts-'+attempt+'.json'),'actual own facts output path differs')
    facts,_=read(facts_path,ref['sha256']);fresh(facts['at'])
    require(facts['run_uuid']==RUN and facts['binding']==binding and facts['read_only'] is True
        and facts['provider_or_allocator_mutations'] is False,'actual own complete facts identity differs')
    request['own_readonly_call_chain']=own_readonly_call_chain(facts,request)
    policy=sources['policy']
    require(policy['schema']=='chimera.qimg_first32.capacity_classification_policy.v1'
        and policy['root_reviewed_resource_policy'] is True,'exact source classification policy required')
    rows,outside,growth,qual_ram,qual_cpu,qvl_active=classify(facts,policy,sources['proposal'],sources['adoption'],proofs,p,service,request)
    require(sources['HY_zero']['schema']=='chimera.qimg_first32.HY_discharged_future_closed_public_proof.v1'
        and sources['H3_CPU_zero']['schema']=='chimera.qimg_first32.H3_CPU_discharged_future_closed_public_proof.v1','completed-scope discharge schemas differ')
    qvl=sources['QVL']['structural_resource_proposal']
    require((qvl['host_memory_bytes'],qvl['CPU_cores'],qvl['new_output_disk_bytes'],qvl['no_swap'])==(16*GIB,4,2*GIB,True),
        'actual pending root QVL source proposal differs')
    qscope=sources['QIMG_scope']['QIMG_future_scope']
    require(qscope['root_snapshot_conservative_max_once_bytes']==50*GIB
        and qscope['conservative_unresolved_extra_bytes']==2*GIB
        and qscope['postcapture_disk_remaining_bytes_once']==55_878_508_944,
        'exact conservative QIMG current/successor RAM and postcapture disk source scope differs')
    # Preserve the original source-holder schema/identity. These are existing
    # QIMG sequence holds, never a fabricated hold signed by a new K3 actor.
    held=[r for r in policy['future_promises'] if r['category'] in ('provider_restoration','H3_reserved_future')]
    require(len(held)==8 and {r['gpu_index'] for r in held if r['category']=='H3_reserved_future'}=={3,5},'all8 original future obligations retained')
    for row in held:
        hold=proofs[p.pin_key(row['sequence_hold_pin'])]
        require(hold['schema']=='chimera.qimg_first32.future_sequence_hold.v1'
            and hold['promise_id']==row['promise_id'] and hold['future_owner']==row['owner']
            and hold['release_before_future_execution'] is True and hold['reservation_edits'] is False
            and hold['foreign_job_interruption'] is False
            and hold['through_scope']=='QIMG_FIRST32_ALL60_READBACK_AND_ALL_EXACT_HEALTHY_RESTORATIONS',
            'actual preserved QIMG/root source hold differs')
        p.resolve_proof(hold['actual_coordination_evidence_pins'],proofs)
    # Full50 H3 remains pending. Its eventual budget is not replaced by the
    # completed CPU measurement or guessed here; all current unknown work still
    # refuses classification and future launch requires fresh normal admission.
    require(sources['source_future']['H3_other_source_fold_or_GPU_future_scopes_discharged'] is False,'full H3 future must remain pending')
    backup_current_absence=backup_quiescent_deferral_current_absence(request['backup_dependency_reconciliation'],facts)
    observation=host();current_growth=sum(g['remaining_growth_reserved_bytes'] for g in growth)
    vxug=sources['VXUG_validator'];NAS=os.statvfs(vxug['NAS_output_root'])
    NAS_validator={'path':vxug['NAS_output_root'],'source_informed_metadata_reserve_bytes':vxug['NAS_new_metadata_bound_bytes'] if request['VXUG_validator_current_active'] else 0,
        'actual_available_bytes':NAS.f_bavail*NAS.f_frsize,'exact_validator_current_active':request['VXUG_validator_current_active'],
        'existing_archive_bytes_recharged':0,'Lexar_or_root_payload_reserve_bytes':0,'enforced_max_claimed':False}
    prospective_ram=OWN_RAM_BYTES if request['mode']=='prospective' else 0
    prospective_cpu=OWN_CPU_THREADS if request['mode']=='prospective' else 0
    qualification_ram=max(0,6*GIB-qual_ram);qualification_cpu=max(0,2-qual_cpu)
    require(not qvl_active,'historical GPU4 QVL allocation cannot substitute current prepared CPU12/4 scope')
    qvl_pending_ram=12*GIB;qvl_pending_cpu=4
    qimg_active=request['QIMG_current_holder_active']
    qimg_pending_ram=qscope['root_snapshot_conservative_max_once_bytes']-(48*GIB if qimg_active else 0)
    qimg_pending_cpu=0 if qimg_active else 8
    speech_active=request['speech_native_current_runtime_active']
    speech_pending_ram=0 if request['speech_native_terminal_future_closed'] else max(0,24*GIB-request['speech_native_current_cap_bytes']);speech_pending_cpu=0 if request['speech_native_terminal_future_closed'] else max(0,2-request['speech_native_current_CPU_threads'])
    nas_active=any(g['accounting_id']=='exact_foreign_copy_worker_256MiB_1CPU' for g in growth)
    nas_pending_ram=0 if nas_active else 268435456;nas_pending_cpu=0 if nas_active else 1
    ram=current_growth+prospective_ram+32*GIB+qualification_ram+qimg_pending_ram+qvl_pending_ram+16*GIB+speech_pending_ram+nas_pending_ram
    cpu=sum(g['cpu_threads'] for g in growth)+prospective_cpu+48+qualification_cpu+qimg_pending_cpu+qvl_pending_cpu+speech_pending_cpu+nas_pending_cpu
    own_write=integer(request['own_remaining_write_bytes'],'actual selected own window remaining new writes')
    copy_remaining=integer(request['remaining_copy_bytes'],'actual still-unmaterialized copy bytes')
    require(request['remaining_copy_bytes']==0 or request['mode']=='prospective','allocated launch requires completed copy')
    require(copy_remaining==0 and own_write==WHOLE_WRITER_BYTES,'selected all12 new37-48 projection and RO resume copy0 required')
    if request['mode']=='allocated':require(request['request_boundary']['next_window']=='49-60','exact actual next37-48 boundary required')
    disk=25_769_803_776+16*GIB
    nas_base=selected['NAS_approved_base_bytes']  # Original foreign predicate is reported separately, not an approved commitment.
    nas_own=own_write  # New own qualification is already inside WHOLE_WRITER_BYTES.
    nas_required=max(selected['NAS_current_common_guard_bytes_once'],nas_base+nas_own)
    root_disk=8*GIB+6*GIB+speech['root_write_reserve_bytes_once']+2*GIB+2*GIB+GIB+selected['ROOT_own_metadata_reserve_bytes']
    fresh(request['at']);fresh(facts['at'])
    # A second full frame detects peer drift during classification/host reads.
    service.compare_frames((facts['containers'],facts['service_runtime_observations'][0]),service.observe_container_frame())
    result={'schema':'neural_forge.k3_owned_launch_capacity.v2','identity':identity,'binding':binding,'at':dt.datetime.now(dt.timezone.utc).isoformat(),
        'attempt_uuid':attempt,'mode':request['mode'],'request_boundary':request['request_boundary'],
        'facts_at':facts['at'], 'foreign_copy_worker_source_scope':sources['foreign_copy_worker'],
        'actual_backup_dependency_reconciled':True,'backup_dependency_reconciliation_schema':'exact2x.backup_deferred_quiescent.v1',
        'actual_backup_v3_transition_proved':False,'backup_dependency_reconciliation':request['backup_dependency_reconciliation'],
        'backup_current_absence':backup_current_absence,
        'source_informed_VXUG_validator_scope':vxug,'NAS_validator_metadata_scope':NAS_validator,
        'request_pin':{'name':args.request.name,'sha256':args.request_sha256},'facts_pin':{'name':facts_path.name,'sha256':ref['sha256']},
        'source_pins':request['sources'],'RAM_required_bytes':ram,'CPU_required_threads':cpu,
        'own_readonly_call_chain':request['own_readonly_call_chain'],
        'remaining_disk_required_bytes':{'lexar':disk,'server_rootfs':root_disk,'NAS_ai_data':nas_required},'host':observation,
        'current_growth_rows':growth,'complete_container_classifications':rows,'complete_outside_process_classifications':outside,
        'ordinary_once':{'RAM_bytes':32*GIB,'cpu_threads':48},'qualification_residual':{'RAM_bytes':qualification_ram,'cpu_threads':qualification_cpu},
        'QIMG_future_successor_delta_RAM_bytes':qscope['conservative_unresolved_extra_bytes'],'QIMG_current_plus_successor_CPU_once':8,
        'QIMG_current_holder_active':request['QIMG_current_holder_active'],
        'QIMG_pending_scope_once':{'RAM_bytes':qimg_pending_ram,'cpu_threads':qimg_pending_cpu,
            'full_source_future_scope_discharged':False,'normal_release_claimed':True,'closed_phase':'fold','CPU1933_and_native_acceptance_claimed':False},
        'speech_acquisition_completed_scope':{'assets':22,'bytes':8_607_865_275,'future_RAM_bytes':0,'future_CPU_threads':0,'remaining_writes_bytes':0,'retained_disk_credit_bytes':0},
        'speech_native_pending_scope_once':{'RAM_bytes':speech_pending_ram,'cpu_threads':speech_pending_cpu,
            'current_exact_runtime_active':False,'current_exact_TTS_phase_active':speech_active,'active_phase_cap_RAM_bytes':request['speech_native_current_cap_bytes'],'active_phase_CPU_threads':request['speech_native_current_CPU_threads'],'sequential_peak_RAM_bytes':24*GIB,'CPU_threads_once':2,'root_write_reserve_bytes_once':20*GIB,'full_native_future_discharged':False,'donor_model_milestone_future_discharged':request['speech_native_terminal_future_closed'],'terminal_owner_event_id':116780,'raw_shared_multimodal_core_completion_claimed':False},
        'root_disk_components':{'ordinary_once_bytes':8*GIB,'qualification_once_bytes':6*GIB,'speech_native_campaign_once_bytes':18*GIB,'speech_native_runtime_once_bytes':2*GIB,'QVL_once_bytes':2*GIB,'QIMG_once_bytes':2*GIB,'NAS_metadata_once_bytes':GIB,'own_stage_and_receipts_reserve_bytes':selected['ROOT_own_metadata_reserve_bytes']},
        'QVL_pending':{'RAM_bytes':qvl_pending_ram,'cpu_threads':qvl_pending_cpu,'new_disk_bytes':2*GIB,'active_current_growth_substituted':qvl_active},
        'host_headroom_RAM_bytes':16*GIB,'Lexar_headroom_bytes':16*GIB,
        'disk_components':{'remaining_copy_bytes':copy_remaining,'K3_selected_window_remaining_bytes':own_write,
            'K3_write_destination':str(WORKING_OUTPUT),'K3_NAS_qualification_reserve_bytes':0,'K3_NAS_qualification_embedded_in_whole_writer_bytes':536870912,
            'QIMG_post_fold_remaining_Lexar_bytes':25_769_803_776,'QVL_ROOT_pending_new_bytes':2*GIB,'Lexar_headroom_bytes':16*GIB},
        'NAS_current_selected_scope':selected,'NAS_required_components':{'current_approved_base_bytes':selected['NAS_approved_base_bytes'],
            'frozen_foreign_worker_operational_guard_bytes':selected['NAS_backup_frozen_operational_guard_bytes'],
            'current_approved_base_once_bytes':nas_base,'current_common_guard_bytes_once':selected['NAS_current_common_guard_bytes_once'],
            'current_common_guard_includes_K3_next49_60_writer_once':True,'historical_K3_37_48_total130_bytes':139586437120,'actual_new_backup_guard_activated':False,
            'frozen_foreign_predicate_currently_enforced':False,'foreign_predicate_applied_as_K3_budget_gate':False,
            'foreign_predicate_would_fit_alongside_own_writes':observation['NAS_available_bytes']>=selected['NAS_backup_frozen_operational_guard_bytes']+nas_own,'own_native_and_qualification_remaining_bytes':nas_own},
        'NAS_backup_pending_scope_once':{'RAM_bytes':nas_pending_ram,'cpu_threads':nas_pending_cpu,'exact_current_growth_substituted':nas_active},
        'preserved_actual_source_future_rows':held,'H3_full50_source_future_discharged':False,'HY_completed_scope_future_bytes':0,
        'H3_completed_CPU_measurement_future_bytes':0,'remaining_disk_materialization_credit_bytes':0,
        'memory_admitted':observation['MemAvailable_bytes']>=ram,'cpu_admitted':observation['host_logical_CPU_count']>=cpu,
        'disk_admitted':observation['disk_available_bytes']>=disk and observation['rootfs_available_bytes']>=root_disk
            and observation['NAS_available_bytes']>=nas_required
            and NAS_validator['actual_available_bytes']>=NAS_validator['source_informed_metadata_reserve_bytes'],
        'actual_locked_admission_completed':False,'provider_or_allocator_mutations':False,'model_payload_reads':0,'overall92_admitted':False}
    result['admitted']=all(result[k] for k in ('memory_admitted','cpu_admitted','disk_admitted'))
    for row in request['sources'].values():read(control/row['name'],row['sha256'])
    path=control/('capacity-'+attempt+'.json');raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    require(len(raw)<=8<<20,'bounded capacity metadata output required')
    with path.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
    require(path.read_bytes()==raw,'complete capacity metadata readback differs')
    print(json.dumps({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'admitted':result['admitted'],
        'actual_locked_admission_completed':False,'provider_or_allocator_mutations':False}))
    return 0 if result['admitted'] else 2


if __name__=='__main__':raise SystemExit(run())
