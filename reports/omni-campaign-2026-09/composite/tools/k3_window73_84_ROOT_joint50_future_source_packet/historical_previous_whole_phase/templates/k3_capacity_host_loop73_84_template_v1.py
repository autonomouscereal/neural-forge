"""Owned filesystem request/response caller for fresh K3 launch preflights.

Reads only exact own source/control metadata and Driver launch requests. Writes
new own metadata leaves and invokes the pinned readonly preflight; no allocator,
provider, Docker start/stop/update or foreign runtime mutation is implemented.
"""
import argparse
import ctypes
import re
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


EXPECTED_RUN='__UNRESOLVED_NEXT73_84_RUN_UUID__'
IDENTITY_SHA='__UNRESOLVED_NEXT73_84_IDENTITY_SHA256__'
BINDING_SHA='__UNRESOLVED_NEXT73_84_BINDING_SHA256__'
CLOSED72_SHA='__UNRESOLVED_ACTUAL_CLOSED72_SHA256__'
FUTURE_SESSION_ID='__UNRESOLVED_GENUINE_SESSION_ID__'
FUTURE_GENERATION=None
FUTURE_CONTEXT_SHA='__UNRESOLVED_GENUINE_CONTEXT_SHA256__'
OWN_RAM_BYTES=None
OWN_CPU_THREADS=None
assert str(uuid.UUID(EXPECTED_RUN))==EXPECTED_RUN
assert EXPECTED_RUN!='764524b1-96a9-4636-be74-942f9c54754c'
assert str(uuid.UUID(FUTURE_SESSION_ID))==FUTURE_SESSION_ID
assert type(FUTURE_GENERATION) is int and FUTURE_GENERATION>=6
assert all(isinstance(x,str) and re.fullmatch(r'[0-9a-f]{64}',x) is not None for x in (IDENTITY_SHA,BINDING_SHA,CLOSED72_SHA,FUTURE_CONTEXT_SHA))
assert type(OWN_RAM_BYTES) is int and OWN_RAM_BYTES>0 and OWN_CPU_THREADS==22
OUTPUT_STORAGE_ROOT='__UNRESOLVED_EXISTING_OUTPUT_STORAGE_ROOT__'
assert OUTPUT_STORAGE_ROOT in ('/mnt/ai_data','/mnt/lexar')
WHOLE_WRITER_BYTES=180943168512
WRITER_SHA='f183d67fc519f4896415b5084c1be069a5574bfbb1cc534b7b4969ebed9577fa'
CLOSED37_48_ROOT='/mnt/ai_data/omni-campaign/k3-nas-window37-48-oct07-764524b1-96a9-4636-be74-942f9c54754c/model'
RO_ROOTS=None
PREVIOUS_RUN_UUID=None
assert isinstance(RO_ROOTS,dict) and len(RO_ROOTS)==4
assert isinstance(PREVIOUS_RUN_UUID,str) and str(uuid.UUID(PREVIOUS_RUN_UUID))==PREVIOUS_RUN_UUID

RUN=EXPECTED_RUN
RUNTIME=Path('/home/cereal/omni-campaign/runtime/k3-nas-window73-84-oct07-'+RUN)
OUTPUT=Path('/home/cereal/omni-campaign/outputs/k3-nas-window73-84-oct07-'+RUN+'/control-output')
WORKING=OUTPUT_STORAGE_ROOT+'/omni-campaign/k3-nas-window73-84-oct07-'+RUN+'/model'
RESUME='/mnt/lexar/omni-campaign/k3-diagnostic-recovery-oct07-048cff57-935c-4e2b-a31b-96a3c7fa6318'


def require(ok,message):
    if ok is not True:raise RuntimeError(message)


def read(path,sha=None,limit=16<<20):
    require(path.is_absolute() and path.resolve()==path and not path.is_symlink()
        and path.is_file() and path.stat().st_size<=limit,'canonical bounded own metadata required')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'own metadata changed during read')
    actual=hashlib.sha256(raw).hexdigest()
    if sha is not None:require(actual==sha,'own full metadata SHA differs')
    def unique(pairs):
        result={}
        for key,value in pairs:require(key not in result,'duplicate JSON key');result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda v:require(False,'nonfiniteJSON')),actual,raw


def write(path,body):
    raw=(json.dumps(body,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
    require(path.read_bytes()==raw,'own complete metadata readback differs')
    return hashlib.sha256(raw).hexdigest()


def publish_sha(path, expected):
    # One fixed65-byte sidecar is durable before atomic publication. Linux
    # RENAME_NOREPLACE preserves any existing immutable boundary evidence.
    path = Path(path)
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            "explicit full publication SHA256 required")
    require(not path.exists() and not path.is_symlink(), "immutable SHA sidecar already exists")
    pending = path.with_name(path.name + ".pending")
    raw = (expected + "\n").encode("ascii")
    with pending.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    require(pending.read_bytes() == raw, "complete pending SHA sidecar readback differs")
    if os.name == "nt":
        os.rename(pending, path)  # Windows rename refuses an existing destination.
    else:
        rename = ctypes.CDLL(None, use_errno=True).renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(pending), -100, os.fsencode(path), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(path))
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    require(path.read_bytes() == raw, "complete published SHA sidecar readback differs")


def sidecar_sha(path):
    # Retain bounded waiting compatibility with an in-progress previous writer;
    # complete malformed metadata refuses rather than weakening a full SHA pin.
    path = Path(path)
    require(not path.is_symlink(), "SHA publication sidecar may not be a symlink")
    if not path.exists():
        return None
    require(path.is_file() and path.stat().st_size <= 80, "regular bounded SHA publication sidecar required")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or len(raw) < 64):
        return None
    require(re.fullmatch(rb"[0-9a-f]{64}\n?", raw) is not None, "complete SHA publication metadata malformed")
    return raw[:64].decode("ascii")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True)
    args=parser.parse_args();config,_,_=read(args.config,args.config_sha256)
    base=args.config.parent
    require(base.is_relative_to(RUNTIME) and config['schema']=='neural_forge.k3_capacity_host_loop_config.v1',
        'exact own runtime caller config required')
    require(config['run_uuid']==RUN and config['output_root']==str(OUTPUT),'selected genuine own runtime/output required')
    host_pin=config['host_caller_source_pin'];host_source=Path(__file__).resolve()
    require(host_source==base/host_pin['name'] and hashlib.sha256(host_source.read_bytes()).hexdigest()==host_pin['sha256'],
        'exact selected current owned host caller source required')
    require(Path(config['driver_config_name']).name==config['driver_config_name'],'own exact driver config leaf required')
    driver_config,driver_config_sha,_=read(base/config['driver_config_name'])
    require(driver_config['capacity_caller']['host_caller_config_sha256']==args.config_sha256,
        'actual immutable driver config must bind selected own host caller config SHA')
    requests=OUTPUT/'capacity_requests';responses=OUTPUT/'capacity_responses';calls=RUNTIME/'capacity_calls'
    for directory in (requests,responses,calls):
        require(directory.is_dir() and directory.resolve()==directory and not directory.is_symlink(),
            'existing canonical own caller metadata directory required')
    sources=config['sources'];scripts=config['scripts'];required={'facts','preflight','service','math'}
    require(set(scripts)==required and 'allocation' in sources,'four pinned selected scripts and raw normal own allocation required')
    require({role:pin['name'] for role,pin in scripts.items()}=={'facts':'k3_owned_readonly_facts_NAS73_84_v1.py','preflight':'k3_owned_capacity_preflight_NAS73_84_v1.py','service':'k3_established_service_observer_peer_fingerprints_v1.py','math':'dynamic_capacity_producer_v2.py'},'exact selected NAS generation6 script leaves required')
    for roster in (sources,scripts):
        for pin in roster.values():
            require(Path(pin['name']).name==pin['name'],'own staged source/control leaf required')
            path=base/pin['name'];require(path.is_file() and not path.is_symlink() and path.resolve()==path
                and path.stat().st_size<=16<<20 and hashlib.sha256(path.read_bytes()).hexdigest()==pin['sha256'],
                'all actual own caller source/control pins must be exact before serving')
    identity,_,_=read(base/sources['identity']['name'],sources['identity']['sha256'])
    require(driver_config['identity']==identity and identity['run_uuid']==RUN
        and identity.get('controller_output_root')==str(OUTPUT) and identity['output_root']==str(OUTPUT)
        and identity.get('native_working_output_root')==WORKING
        and identity.get('native_working_copy_mounted_read_only') is False
        and identity.get('original_failed_source_mounted_read_only') is True
        and identity.get('checkpoint_retention')=='--snapshot-keep 0: preserve all existing and new native checkpoints'
        and 'readonly_saved_prefix_root' not in identity
        and identity.get('new_output_filesystem')==('NAS' if OUTPUT_STORAGE_ROOT=='/mnt/ai_data' else 'Lexar')
        and identity.get('readonly_resume_source_roots')==RO_ROOTS and identity.get('resume_source_mounted_read_only') is True
        and identity.get('closed72_continuity_sha256')==CLOSED72_SHA and sources['identity']['sha256']==IDENTITY_SHA and sources['binding']['sha256']==BINDING_SHA
        and identity.get('new_layers')==[73,84] and identity.get('only_this_window_scope') is True
        and identity.get('native_binding',{}).get('session_generation')==FUTURE_GENERATION
        and identity['native_binding'].get('session_id')==FUTURE_SESSION_ID and identity['native_binding'].get('context_sha256')==FUTURE_CONTEXT_SHA
        and sources['own73_84_writer_extent']['sha256']==WRITER_SHA,
        'exact honest generation6 RO closed36 and unique NAS37-48/controller identity required')
    allocation,_,_=read(base/sources['allocation']['name'],sources['allocation']['sha256'])
    require(allocation['allocation_id']==config['allocation_id'] and allocation['container_id']==config['container_id']
        and allocation['state']=='running','exact current raw normal own allocation required')
    print(json.dumps({'serving':True,'requests':str(requests),'responses':str(responses),'run_uuid':RUN}),flush=True)
    while True:
        require(not (OUTPUT/'capacity_caller_stop.json').is_symlink(),'own stop metadata may not be a symlink')
        if (OUTPUT/'capacity_caller_stop.json').is_file():
            stop,_,_=read(OUTPUT/'capacity_caller_stop.json',limit=4096)
            require(stop=={'run_uuid':RUN,'stop':True},'exact own caller stop metadata required')
            return 0
        pending=sorted(path for path in requests.glob('launch-request-*.json')
            if sidecar_sha(responses/('response-'+path.stem[len('launch-request-'):]+'.sha256')) is None)
        require(len(pending)<=64,'bounded own launch request roster required')
        for path in pending:
            sidecar=path.with_suffix('.sha256')
            published_sha=sidecar_sha(sidecar)
            if published_sha is None:continue
            require(sidecar.resolve()==sidecar,'canonical own launch SHA sidecar required')
            request,request_sha,_=read(path,published_sha,limit=65536)
            attempt=str(uuid.UUID(request['attempt_uuid']))
            require(attempt==request['attempt_uuid'] and path.name=='launch-request-'+attempt+'.json','new canonical exact own request UUID required')
            response_path=responses/('response-'+attempt+'.json')
            if response_path.exists():continue
            read(base/config['driver_config_name'],driver_config_sha)
            # Bind the request to the one currently selected immutable driver,
            # its config and committed allocation. No peer capacity is consumed.
            require(request['schema']=='neural_forge.k3_driver_capacity_request.v1' and request['run_uuid']==RUN
                and request['allocation_id']==config['allocation_id'] and request['container_id']==config['container_id']
                and request['driver_sha256']==config['driver_sha256'] and request['config_sha256']==driver_config_sha,
                'own driver/config/allocation request pins differ')
            require(type(request['own_remaining_write_bytes']) is int and request['own_remaining_write_bytes']==WHOLE_WRITER_BYTES
                and request['remaining_copy_bytes']==0 and request['next_window']=='73-84'
                and request['resume_layer'] is None,'only exact new NAS37-48 computed writes and RO resume copy0 required')
            boundary={key:request[key] for key in ('run_uuid','allocation_id','container_id','driver_sha256','config_sha256',
                'next_window','resume_layer','own_remaining_write_bytes','native_tool_sha256','projection_sha256','admission_sha256',
                'identity_sha256','source_pins_sha256','allocation_receipt_sha256','recovery_packet_sha256','request_boundary_id')}
            require(request['request_boundary_id']==attempt,'canonical exact request boundary UUID required')
            projection_raw=(json.dumps(request['projection'],sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
            require(hashlib.sha256(projection_raw).hexdigest()==request['projection_sha256']
                and request['projection']['required_new_bytes']==request['own_remaining_write_bytes'],
                'exact Driver.admit disk projection full SHA/bytes required')
            boundary['driver_request_sha256']=request_sha
            control=calls/attempt;control.mkdir(exist_ok=False)
            for roster in (sources,scripts):
                for pin in roster.values():
                    original=base/pin['name'];raw=original.read_bytes()
                    require(hashlib.sha256(raw).hexdigest()==pin['sha256'],'selected immutable source changed before copy')
                    target=control/pin['name']
                    if target.exists():require(target.read_bytes()==raw,'duplicate source leaf differs');continue
                    with target.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
                    require(target.read_bytes()==raw,'exact own staged metadata/source byte readback differs')
            actual={'schema':'neural_forge.k3_owned_capacity_preflight_request.v2',
                'at':dt.datetime.now(dt.timezone.utc).isoformat(),'attempt_uuid':attempt,'mode':'allocated',
                'request_boundary':boundary,'own_remaining_write_bytes':request['own_remaining_write_bytes'],'remaining_copy_bytes':0,
                'sources':sources,'ordinary_source_pins':[sources[role] for role in ('adoption','proposal')],
                'facts_source_name':scripts['facts']['name'],'service_source_name':scripts['service']['name'],'math_source_name':scripts['math']['name']}
            tail=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
            group=next(line.split('::',1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
            actual['owned_host_caller']={'pid':os.getpid(),'ppid':int(tail[1]),'start_ticks':int(tail[19]),
                'comm':Path('/proc/self/comm').read_text().strip(),'exe':str(Path(sys.executable).resolve()),
                'cgroup_path':str(Path('/sys/fs/cgroup')/group.lstrip('/')),
                'source_path':str(host_source),'source_sha256':host_pin['sha256'],
                'config_path':str(args.config),'config_sha256':args.config_sha256}
            actual_path=control/'preflight-request.json';actual_sha=write(actual_path,actual)
            process=subprocess.run([sys.executable,'-B',str(control/scripts['preflight']['name']),
                '--request',str(actual_path),'--request-sha256',actual_sha],capture_output=True,text=True,timeout=100)
            result={'schema':'neural_forge.k3_driver_capacity_response.v1','at':dt.datetime.now(dt.timezone.utc).isoformat(),
                'run_uuid':RUN,'attempt_uuid':attempt,'driver_request_sha256':request_sha,'request_boundary':boundary,
                'preflight_request_sha256':actual_sha,'exit_code':process.returncode,'admitted':False,
                'host_caller_config_sha256':args.config_sha256,'source_scripts':scripts,'provider_or_allocator_mutations':False}
            if process.returncode in (0,2):
                pin=json.loads(process.stdout);capacity_path=Path(pin['path'])
                require(capacity_path==control/('capacity-'+attempt+'.json'),'actual own capacity path differs')
                capacity,capacity_sha,raw=read(capacity_path,pin['sha256'])
                require(capacity['request_boundary']==boundary and capacity['attempt_uuid']==attempt
                    and capacity['mode']=='allocated','actual source capacity differs from exact own launch request')
                target=responses/('capacity-'+attempt+'.json')
                with target.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
                require(target.read_bytes()==raw,'own mounted capacity complete byte readback differs')
                result.update(capacity_pin={'name':target.name,'sha256':capacity_sha},admitted=capacity['admitted'])
            else:
                result['error_tail']=(process.stderr or process.stdout)[-4096:]
            response_sha=write(response_path,result)
            publish_sha(response_path.with_suffix('.sha256'),response_sha)
            print(json.dumps({'response':str(response_path),'admitted':result['admitted'],'attempt_uuid':attempt}),flush=True)
        time.sleep(1)


if __name__=='__main__':raise SystemExit(main())
