"""Extract authentic60 raw9 and prior49 raw freeze locally, after natural exit.

Consumes the complete reviewed terminal collector's decoded bytes and a genuine
postterminal connect receipt. No connection, database, runtime or model action.
Future UUID/filesystem default to unresolved and no allocation/cap is claimed.
"""
import argparse,base64,hashlib,json,os,re,uuid
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUTPUT_PARENT=Path('D:/codex-owned-source-receipts/k3-window61-72-source-only-9d98292e804f')
RECOVERY=HERE.parent
PRIOR=RECOVERY/'recovery_resource_review/actual49_60_binding_proposal_v1'
RUN='03603908-570d-4c90-96cb-ee3a63a9a139'
CID='f924bdf0fa81ff21e26b791623a115e6812bd579b3a1d048b69b451ddf0a7514'
A='aebe07fe-2905-470c-8a10-f683a770faeb'
META='/home/cereal/omni-campaign/outputs/k3-nas-window49-60-oct07-'+RUN+'/control-output'
MODEL='/mnt/ai_data/omni-campaign/k3-nas-window49-60-oct07-'+RUN+'/model'
DRIVER=META+'/continuation/'+RUN
SEAL=MODEL+'/ancestry/window-49-60-'+RUN
ROLES={'closed60_continuity':SEAL+'/window_continuity.json',
    'terminal_receipt':SEAL+'/receipts/k3_dense_pass_receipt.json',
    'full_ffn_validation':SEAL+'/inputs/full_ffn_validation.json',
    'snapshot_manifest':SEAL+'/fold/snapshots/snapshots_manifest.json',
    'fitter_execution':DRIVER+'/window49_60_launch/fitter-49-60.execution.json',
    'controller_terminal_result':DRIVER+'/terminal_result.json',
    'resources_jsonl':DRIVER+'/resources.jsonl','controller_events_jsonl':DRIVER+'/events.jsonl'}
FREEZE_SHA='f17c6912d1e3ba5f62f7943ba9f8e004d9c8def3be54e5820bce174899110647'

def require(ok,message):
    if ok is not True:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encode(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def decode(raw):
    def unique(items):
        out={}
        for key,value in items:require(key not in out,'duplicate JSON key');out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda value:require(False,'nonfinite metadata'))
def pinned(path,expected):
    require(isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected) is not None,'complete raw SHA required')
    path=Path(path);require(path.is_file() and not path.is_symlink() and path.stat().st_size<=64<<20,'bounded regular metadata required')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(stamp(before)==stamp(after) and sha(raw)==expected,'full raw byte/stat pin differs');return raw
def member(row):
    require(type(row['bytes']) is int and 0<=row['bytes']<=64<<20,'bounded complete member required')
    raw=base64.b64decode(row['raw_base64'],validate=True)
    require(len(raw)==row['bytes'] and sha(raw)==row['sha256'],'full raw member SHA/size differs');return raw
def write(path,raw):
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    require(path.read_bytes()==raw,'exclusive local byte readback differs')
    return {'path':str(path.resolve()),'sha256':sha(raw)}

def assemble(args):
    expanded=pinned(args.release_expanded,args.release_expanded_sha256);physical=decode(expanded)
    evidence=physical['terminal_evidence'];state=physical['docker_state']
    require(physical['ok'] is True and physical['container_id']==CID and physical['read_only'] is True
        and physical['container_preserved'] is True and physical['source_freeze_sha256']==FREEZE_SHA
        and physical['exact49_60_production_source_closure'] is True,'authentic exact49 terminal collector required')
    require(evidence['complete'] is True and evidence['tensor_payload_reads']==0
        and evidence['model_artifact_success_claimed'] is False and evidence['docker_state']==state
        and state['Status']=='exited' and state['Running'] is state['Paused'] is state['Restarting'] is False
        and state['Pid']==state['ExitCode']==0 and state['OOMKilled'] is False,'successful natural49-60 terminal required')
    require(evidence['actual_closed60_metadata_join']['metadata_joins_complete'] is True
        and evidence['actual_closed60_metadata_join']['full_prefix_layers']==60,'authentic full closed60 metadata join required')
    rows=evidence['files'];require(len({row['path'] for row in rows})==len(rows),'duplicate raw member path')
    full={row['path']:member(row) for row in rows}
    require(set(evidence['docker_logs'])=={'stdout','stderr'},'both complete Docker logs required')
    for row in evidence['docker_logs'].values():member(row)
    require(all(path in full and path not in evidence['explicit_absences'] for path in ROLES.values()),'all eight actual60 raw roles required')
    raw={role:full[path] for role,path in ROLES.items()}
    for role,body in raw.items():
        if role.endswith('_jsonl'):
            require(bool(body),'empty actual JSONL proof')
            for line in body.splitlines():
                if line:decode(line)
        else:decode(body)
    execution=decode(raw['fitter_execution']);terminal=decode(raw['controller_terminal_result'])
    require(type(execution['pid']) is int and execution['pid']>0
        and execution['pid']==evidence['actual_fitter_identity']['namespace_pid'] and execution['exit_code']==0
        and terminal['stage']=='bounded_window_complete' and terminal['window_scope']=='49-60'
        and terminal['full_prefix_layers']==60 and terminal['allocation']['allocation_id']==A
        and terminal['allocation']['container_id']==CID,'actual execution namespace PID and whole-terminal source join required')
    freeze_raw=full[META+'/execution_source_freeze.json'];require(sha(freeze_raw)==FREEZE_SHA,'full immutable actual49 freeze raw required')
    freeze=decode(freeze_raw);require(freeze['schema']=='neural_forge.k3_NAS49_60_execution_source_freeze.v1'
        and freeze['allocation_id']==A and freeze['container_id']==CID and freeze['identity']['run_uuid']==RUN,
        'actual full prior49 freeze schema/identity/allocation required')
    binding_raw=pinned(args.binding_receipt,args.binding_receipt_sha256);binding=decode(binding_raw);data=binding['response']['data']
    require(binding['exit_code']==0 and binding['response']['ok'] is True
        and binding['argv']==['C:/Users/cereal/.agents/skills/coordinate-peer-agents/scripts/session.py','connect']
        and data['harness_id']=='01a11108-009c-72f3-b2ba-e0d48fca1435'
        and data['thread_id']=='95f6c2bd-7139-46f6-b5d6-228d1b9753f7'
        and type(data['session_generation']) is int and data['session_generation']>=6
        and str(uuid.UUID(data['session_id']))==data['session_id']
        and re.fullmatch('[0-9a-f]{64}',data['context_sha256']) is not None,'genuine current own normal connect required')
    parse=lambda s:datetime.fromisoformat(s.replace('Z','+00:00'))
    require(parse(state['FinishedAt'])<=parse(binding['at'])<=datetime.now(timezone.utc),'genuine postterminal connect timestamp required')
    raw['binding_receipt']=binding_raw;require(len(raw)==9,'exact9 raw roles required')
    if args.new_run_uuid is not None:require(str(uuid.UUID(args.new_run_uuid))==args.new_run_uuid and args.new_run_uuid!=RUN,'new canonical61 UUID required')
    out=args.output_directory.resolve();require(out.parent==OUTPUT_PARENT and OUTPUT_PARENT.is_dir() and not OUTPUT_PARENT.is_symlink() and not out.exists() and not out.is_symlink(),'exclusive own D local extraction leaf required')
    # Full provenance is the actual49 source bundle/configs; no later allocation
    # is created, inferred or substituted from the previous allocation.
    names={'source_closure':PRIOR/'source_bundle_launch_v1/SOURCE_CLOSURE49_60.json',
        'source_binding':PRIOR/'source_bundle_launch_v1/SOURCE_BINDING_RESULT.json',
        'identity':PRIOR/'launch_receipts_v1/identity.json',
        'catalog':PRIOR/'configs_launch_v1/ro_multi_origin_catalog.json',
        'observation':PRIOR/'configs_launch_v1/ro_multi_origin_observation.json',
        'driver_config':PRIOR/'configs_launch_v1/continuation_config.json',
        'allocation':PRIOR/'launch_receipts_v1/actual_allocation.json',
        'recovery_plan':PRIOR/'configs_launch_v1/recovery_plan49_60.json'}
    prior_refs={name:{'path':str(path.resolve()),'sha256':sha(path.read_bytes())} for name,path in names.items()}
    require(prior_refs['driver_config']['sha256']==freeze['config_sha256']
        and all(prior_refs[name]['sha256']==freeze['control_files'][leaf] for name,leaf in
            [('identity','identity.json'),('catalog','ro_multi_origin_catalog.json'),('observation','ro_multi_origin_observation.json'),
             ('allocation','actual_allocation.json'),('recovery_plan','recovery_plan49_60.json')]),'actual prior source/config/catalog/allocation pins differ from rawfreeze')
    out.mkdir(exist_ok=False);refs={}
    for role,body in raw.items():refs[role]=write(out/(role+('.jsonl' if role.endswith('_jsonl') else '.json')),body)
    prior_refs['source_freeze']=write(out/'prior49_execution_source_freeze.json',freeze_raw)
    spec={'schema':'neural_forge.k3_actual61_72_local_source_binding_inputs.v1','run_uuid':args.new_run_uuid,
        'new_output_filesystem':args.output_filesystem,'operating_cap_proposal':{'memory_bytes':104*(1<<30),'CPU_threads':22,'swap_bytes':0},
        'inputs':refs,'previous49':{'source_bundle':str((PRIOR/'source_bundle_launch_v1').resolve()),'inputs':prior_refs}}
    result={'schema':'neural_forge.k3_actual60_local_raw9_extraction_receipt.v1','source_only':True,
        'extractor_sha256':sha(Path(__file__).read_bytes()),'release_expanded_sha256':sha(expanded),
        'raw9_members':refs,'full_prior49_freeze':prior_refs['source_freeze'],'actual_native_namespace_pid':execution['pid'],
        'actual60_continuity_sha256':refs['closed60_continuity']['sha256'],'actual_next_allocation':None,
        'actual_next_capacity':None,'runtime_or_remote_actions':False,'can_arm':False}
    pins={}
    for name,value in [('ACTUAL61_SOURCE_BINDING_INPUTS_V1.json',spec),('ACTUAL60_RAW9_EXTRACTION_RECEIPT_V1.json',result)]:pins[name]=write(out/name,encode(value))
    print(json.dumps(pins,sort_keys=True))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('release-expanded','binding-receipt'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--output-directory',type=Path,required=True);p.add_argument('--new-run-uuid')
    p.add_argument('--output-filesystem',choices=('NAS','Lexar'))
    assemble(p.parse_args())
if __name__=='__main__':main()
