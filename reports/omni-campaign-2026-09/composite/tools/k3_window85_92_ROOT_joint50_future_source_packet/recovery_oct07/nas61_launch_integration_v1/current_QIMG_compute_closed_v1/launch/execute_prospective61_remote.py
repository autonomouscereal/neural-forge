"""Publish one honest source-owner prospective v2 request, then exec unchanged preflight.

This runs foreground in the existing SSH control chain. No daemon, host caller,
allocator, provider action, invented Driver request or native phase assertion.
"""
import datetime, hashlib, json, os, uuid
from pathlib import Path

RUN=I['run_uuid']
assert str(__import__('uuid').UUID(RUN))==RUN and RUN!='03603908-570d-4c90-96cb-ee3a63a9a139'
WRITER=166556705792
def canonical(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert I['run_uuid']==RUN and I['new_layers']==[61,72]
assert str(uuid.UUID(ATTEMPT))==ATTEMPT
assert hashlib.sha256(canonical(CLOSURE)).hexdigest()==CLOSURE_SHA
assert CLOSURE['schema']=='neural_forge.k3_owned_capacity_source_closure.v15'
assert CLOSURE['selected_run_uuid']==RUN and len(CLOSURE['sources'])==70 and len(CLOSURE['scripts'])==5
root=Path(I['runtime_root']);base=root/'production_control'
assert root==Path('/home/cereal/omni-campaign/runtime/k3-nas-window61-72-oct07-'+RUN)
for folder in (root,base,Path(I['output_root']),Path(I['native_working_output_root'])):
    assert folder.is_dir() and folder.resolve()==folder and not folder.is_symlink()
verified={}
for pin in [*CLOSURE['sources'].values(),*CLOSURE['scripts'].values()]:
    name=pin['name'];assert Path(name).name==name and name not in verified
    path=base/name;assert path.is_file() and path.resolve()==path and not path.is_symlink()
    raw=path.read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256']
    verified[name]=raw
assert len(verified)==75 and json.loads(verified[CLOSURE['sources']['identity']['name']])==I
for relative,pin in CLOSURE['source_binding']['production_file_pins'].items():
    path=root/relative;assert path.is_file() and not path.is_symlink() and path.resolve()==path
    assert path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256']
assert CLOSURE['source_binding']['writer_bytes_once']==WRITER
assert CLOSURE['source_binding']['common_NAS_floor_compare_bytes']==692026605568+(WRITER if I['new_output_filesystem']=='NAS' else 0)
calls=root/'capacity_calls';calls.mkdir(exist_ok=True)
assert calls.resolve()==calls and not calls.is_symlink()
control=calls/ATTEMPT;control.mkdir(exist_ok=False)
def exclusive(path,raw):
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    assert path.read_bytes()==raw
for name,raw in verified.items():exclusive(control/name,raw)
sources={role:{key:pin[key] for key in ('name','sha256')} for role,pin in CLOSURE['sources'].items()}
scripts=CLOSURE['scripts']
request={'schema':'neural_forge.k3_owned_capacity_preflight_request.v2',
    'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt_uuid':ATTEMPT,'mode':'prospective',
    'request_boundary':{'purpose':'SOURCE_OWNER_NEXT61_72_PROSPECTIVE','next_window':'61-72',
        'request_boundary_id':ATTEMPT,'source_closure_sha256':CLOSURE_SHA,'driver_emitted_request':False},
    'own_remaining_write_bytes':WRITER,'remaining_copy_bytes':0,'sources':sources,
    'ordinary_source_pins':[sources[role] for role in ('adoption','proposal')],
    'facts_source_name':scripts['facts']['name'],'service_source_name':scripts['service']['name'],
    'math_source_name':scripts['math']['name'],'driver_emitted_request':False,
    'purpose':'SOURCE_OWNER_NEXT61_72_PROSPECTIVE'}
raw=canonical(request);request_sha=hashlib.sha256(raw).hexdigest()
path=control/'source-owner-prospective.json'
# No reader starts until both complete immutable files have been read back.
exclusive(path.with_suffix('.sha256'),(request_sha+'\n').encode())
exclusive(path,raw)
args=['/usr/bin/python3','-B',str(control/scripts['preflight']['name']),
    '--request',str(path),'--request-sha256',request_sha]
os.execv(args[0],args)
