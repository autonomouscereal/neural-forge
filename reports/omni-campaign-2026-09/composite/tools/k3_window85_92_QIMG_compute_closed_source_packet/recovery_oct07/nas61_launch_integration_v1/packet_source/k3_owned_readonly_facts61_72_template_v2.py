"""Bounded read-only current workload identity facts; never resource admission.

CLI supplies pinned K3 identity and current own binding. No Env, Cmd, commandline, model bytes or
credentials are collected. Only the existing own metadata directory is written.
"""
import datetime, hashlib, json, os, pathlib, subprocess, uuid, importlib.util, re

EXPECTED_RUN='__UNRESOLVED_NEXT61_72_RUN_UUID__'
IDENTITY_SHA='__UNRESOLVED_NEXT61_72_IDENTITY_SHA256__'
BINDING_SHA='__UNRESOLVED_NEXT61_72_BINDING_SHA256__'
CLOSED60_SHA='__UNRESOLVED_ACTUAL_CLOSED60_SHA256__'
FUTURE_SESSION_ID='__UNRESOLVED_GENUINE_SESSION_ID__'
FUTURE_GENERATION=None
FUTURE_CONTEXT_SHA='__UNRESOLVED_GENUINE_CONTEXT_SHA256__'
OWN_RAM_BYTES=None
OWN_CPU_THREADS=None
assert str(uuid.UUID(EXPECTED_RUN))==EXPECTED_RUN
assert EXPECTED_RUN!='764524b1-96a9-4636-be74-942f9c54754c'
assert str(uuid.UUID(FUTURE_SESSION_ID))==FUTURE_SESSION_ID
assert type(FUTURE_GENERATION) is int and FUTURE_GENERATION>=6
assert all(isinstance(x,str) and re.fullmatch(r'[0-9a-f]{64}',x) is not None for x in (IDENTITY_SHA,BINDING_SHA,CLOSED60_SHA,FUTURE_CONTEXT_SHA))
assert type(OWN_RAM_BYTES) is int and OWN_RAM_BYTES>0 and OWN_CPU_THREADS==22
OUTPUT_STORAGE_ROOT='__UNRESOLVED_EXISTING_OUTPUT_STORAGE_ROOT__'
assert OUTPUT_STORAGE_ROOT in ('/mnt/ai_data','/mnt/lexar')
WHOLE_WRITER_BYTES=166556705792
WRITER_SHA='d85a82c01d844daf2b89ab2b4071f5d6b4f7b616ae581e65ade55767d96ad78e'
CLOSED37_48_ROOT='/mnt/ai_data/omni-campaign/k3-nas-window37-48-oct07-764524b1-96a9-4636-be74-942f9c54754c/model'
RO_ROOTS={'legacy_closed36':'/mnt/lexar/omni-campaign/k3-diagnostic-recovery-oct07-048cff57-935c-4e2b-a31b-96a3c7fa6318','closed37_48':CLOSED37_48_ROOT,'closed49_60':'/mnt/ai_data/omni-campaign/k3-nas-window49-60-oct07-03603908-570d-4c90-96cb-ee3a63a9a139/model'}

P = pathlib.Path

def require(ok, message):
    if ok is not True: raise RuntimeError(message)

def command(argv):
    return subprocess.check_output(argv, timeout=8)

def counters(group):
    group = P(group)
    require(group.is_relative_to(P('/sys/fs/cgroup')) and group.resolve() == group and group.is_dir(), 'canonical observed live cgroup required')
    result = {'cgroup_path': str(group)}
    for name in ('memory.max', 'memory.current', 'memory.peak', 'memory.swap.max', 'memory.swap.current', 'cpu.max', 'cpuset.cpus.effective', 'pids.current'):
        path = group / name
        result[name] = path.read_text().strip() if path.is_file() else None
    for name in ('memory.stat', 'memory.events', 'cpu.stat'):
        path = group / name
        result[name] = {k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())} if path.is_file() else None
    return result

def running():
    return helper.observe_container_frame()


def processes():
    directories = [p for p in P('/proc').iterdir() if p.name.isdecimal()]
    require(len(directories) <= 8192, 'bounded host process inventory required')
    rows = []
    for directory in directories:
        pid = int(directory.name)
        if pid == os.getpid(): continue
        try:
            raw_stat = (directory / 'stat').read_text(); tail = raw_stat[raw_stat.rfind(')') + 2:].split()
            status = dict(line.split(':', 1) for line in (directory / 'status').read_text().splitlines() if ':' in line)
            group = next(line.split('::', 1)[1] for line in (directory / 'cgroup').read_text().splitlines() if line.startswith('0::'))
            try: executable = os.readlink(directory / 'exe')
            except (PermissionError, FileNotFoundError): executable = None
            rows.append({'pid': pid, 'ppid': int(tail[1]), 'start_ticks': int(tail[19]), 'comm': (directory / 'comm').read_text().strip(),
                         'exe': executable, 'RSS_bytes': int(status.get('VmRSS', '0 kB').split()[0]) * 1024,
                         'HWM_bytes': int(status.get('VmHWM', '0 kB').split()[0]) * 1024,
                         'threads': int(status.get('Threads', '0').strip()), 'cpu_allowed_list': status.get('Cpus_allowed_list', '').strip(),
                         'cpu_ticks': int(tail[11]) + int(tail[12]), 'cgroup_path': str(P('/sys/fs/cgroup') / group.lstrip('/'))})
        except (FileNotFoundError, ProcessLookupError): continue
    return sorted(rows, key=lambda row: row['pid'])

# The caller supplies the selected genuine recovery identity and current own binding.
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--identity',type=P,required=True)
parser.add_argument('--identity-sha256',required=True)
parser.add_argument('--binding',type=P,required=True)
parser.add_argument('--binding-sha256',required=True)
parser.add_argument('--observation-uuid',required=True)
parser.add_argument('--output-directory',type=P,required=True)
parser.add_argument('--service-helper',type=P,required=True)
args=parser.parse_args()
def pinned_metadata(path,sha):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=65536,'bounded regular own metadata required')
    raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==sha,'exact original own metadata pin differs')
    return json.loads(raw)
i=pinned_metadata(args.identity,args.identity_sha256)
binding=pinned_metadata(args.binding,args.binding_sha256)
NATIVE='01a11108-009c-72f3-b2ba-e0d48fca1435'
RUN=i['run_uuid']
require(str(uuid.UUID(RUN))==RUN and RUN==EXPECTED_RUN,
        'selected canonical recovery UUID required')
ROOT=P(i['runtime_root'])
require(ROOT.parent==P('/home/cereal/omni-campaign/runtime')
        and ROOT.name=='k3-nas-window61-72-oct07-'+RUN and ROOT.is_absolute(),
        'selected recovery runtime root differs')
require(i['native_thread_id']==NATIVE and i['owner']=='codex-native-'+NATIVE
        and i['private_relay_thread']=='95f6c2bd-7139-46f6-b5d6-228d1b9753f7'
        and i['first_session_meta_sha256']=='816471fc9bb8d0131f8d7eeba55f7e2c5aa3e7cf77ff10c69a09939dfca9cd71'
        and i['output_root']=='/home/cereal/omni-campaign/outputs/k3-nas-window61-72-oct07-'+RUN+'/control-output'
        and i['historical_runtime_root']=='/home/cereal/omni-campaign/runtime/k3-continuation-oct06-bde8b657-90d4-41e8-83b3-7908616aeaf1'
        and i['historical_output_root']=='/mnt/lexar/omni-campaign/k3-continuation-oct06-bde8b657-90d4-41e8-83b3-7908616aeaf1'
        and i['native_working_output_root']==OUTPUT_STORAGE_ROOT+'/omni-campaign/k3-nas-window61-72-oct07-'+RUN+'/model'
        and i['new_output_filesystem']==('NAS' if OUTPUT_STORAGE_ROOT=='/mnt/ai_data' else 'Lexar')
        and i['readonly_resume_source_roots']==RO_ROOTS
        and i['resume_source_mounted_read_only'] is True
        and i['new_layers']==[61,72] and i['only_this_window_scope'] is True
        and args.identity_sha256==IDENTITY_SHA and args.binding_sha256==BINDING_SHA and i['closed60_continuity_sha256']==CLOSED60_SHA
        and i['controller_output_root']==i['output_root']
        and i['native_working_copy_mounted_read_only'] is False
        and i['original_failed_source_mounted_read_only'] is True
        and i['checkpoint_retention']=='--snapshot-keep 0: preserve all existing and new native checkpoints'
        and 'readonly_saved_prefix_root' not in i,
        'selected own identity and explicit historical ancestry pins required')
require(binding==i['native_binding'] and binding['harness_id']==NATIVE
        and binding['thread_id']==i['private_relay_thread']
        and binding['session_id']==FUTURE_SESSION_ID and binding['session_generation']==FUTURE_GENERATION
        and binding['context_sha256']==FUTURE_CONTEXT_SHA,'exact current own recovery binding required')
CONTROL=args.output_directory
require(CONTROL.is_absolute() and CONTROL.is_relative_to(ROOT) and CONTROL.is_dir() and CONTROL.resolve()==CONTROL
        and not CONTROL.is_symlink(),'selected recovery-owned canonical metadata directory required')
observation=str(uuid.UUID(args.observation_uuid))
require(observation==args.observation_uuid,'new canonical own observation UUID required')
HELPER_SHA='06ad5e26e88d66653e430055683309b6dbc83bf4d0d32b98fad4c6fcf7afe783'
helper_path=args.service_helper
require(helper_path.is_absolute() and helper_path.is_relative_to(ROOT) and helper_path.is_file()
        and not helper_path.is_symlink() and helper_path.stat().st_size<=65536
        and hashlib.sha256(helper_path.read_bytes()).hexdigest()==HELPER_SHA,'exact existing readonly service helper in own controls required')
helper_spec=importlib.util.spec_from_file_location('k3_readonly_service_restart',helper_path)
helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)

observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
before_frame=running();before,protected_service=before_frame
rows=processes()
plumbing=helper.observe_plumbing(rows)

groups = sorted({row['cgroup_path'] for row in rows} | {row['cgroup_path'] for row in before})
group_rows = []
vanished_groups = []
for group in groups:
    try: group_rows.append(counters(group))
    except FileNotFoundError: vanished_groups.append(group)
after_frame=running();helper.compare_frames(before_frame,after_frame)
# No PID0 row is an active container. The separate immutable identity/state
# record protects the original physical service and the full funded family.
require(not vanished_groups, 'actual observed cgroups vanished; recollect')

body = {'schema': 'neural_forge.k3_recovery_foreign_facts.v1', 'at': observed_at,
        'native': NATIVE, 'run_uuid': RUN, 'owner': i['owner'], 'binding': binding, 'observation_uuid': observation,
        'identity_sha256':args.identity_sha256, 'binding_sha256':args.binding_sha256,
        'runtime_root':str(ROOT), 'output_root':i['output_root'],
        'historical_runtime_root':i['historical_runtime_root'], 'historical_output_root':i['historical_output_root'],
        'native_working_output_root':i['native_working_output_root'], 'controller_output_root':i['controller_output_root'],
        'native_working_copy_mounted_read_only':i['native_working_copy_mounted_read_only'],
        'original_failed_source_mounted_read_only':i['original_failed_source_mounted_read_only'],
        'checkpoint_retention':i['checkpoint_retention'],
        'readonly_resume_source_roots':i['readonly_resume_source_roots'],
        'resume_source_mounted_read_only':i['resume_source_mounted_read_only'],
        'containers': before, 'processes': rows, 'cgroups': group_rows, 'vanished_process_cgroups': vanished_groups,
        'protected_established_service_objects':[protected_service],
        'service_runtime_observations':[before_frame[1],after_frame[1]],
        'fixed_containerd_plumbing_processes':plumbing,
        'service_restart_helper_sha256':HELPER_SHA,
        'existing_readonly_shim_identity_evidence':{'name':'actual_existing_readonly_shim_identity_v1-20261006T225922121362Z.json',
            'sha256':'77a2b2e64708e0c6ae776f7fe4b390e4c8810e9c3a24f0f2f42546f2b63ede03'},
        'host_logical_CPU_count': os.cpu_count(), 'read_only': True, 'admission': False, 'future_envelopes_proven': False,
        'S3_release_inferred': False, 'provider_or_allocator_mutations': False, 'model_payload_reads': 0,
        'excluded_sensitive_surfaces': ['Env', 'Cmd', 'cmdline', 'environ', 'credentials', 'model/source payloads']}
helper.validate_facts(body)
raw = (json.dumps(body, separators=(',', ':'), allow_nan=False) + '\n').encode()
require(len(raw) <= 8 << 20, 'bounded sanitized output required')
path = CONTROL / ('foreign_classification_facts-' + observation + '.json')
with path.open('xb') as handle:
    handle.write(raw); handle.flush(); os.fsync(handle.fileno())
actual = path.read_bytes()
require(actual == raw, 'complete metadata byte readback differs')
print(json.dumps({'path': str(path), 'sha256': hashlib.sha256(actual).hexdigest(), 'bytes': len(actual),
                  'containers': len(before), 'processes': len(rows), 'cgroups': len(group_rows), 'admission': False, 'resource_mutations': False}))
