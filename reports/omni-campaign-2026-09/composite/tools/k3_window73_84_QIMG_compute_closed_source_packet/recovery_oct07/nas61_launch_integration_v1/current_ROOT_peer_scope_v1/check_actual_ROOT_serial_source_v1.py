"""Focused pure public-metadata joins; no candidate module/main import."""
from pathlib import Path
import ast,copy,hashlib,json,os

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encode(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def require(ok,message):
    if ok is not True:raise RuntimeError(message)
def pin(path):
    raw=path.read_bytes();return {'path':str(path.resolve()),'sha256':sha(raw),'bytes':len(raw)}
def seal(value):
    raw=encode(value);path=HERE/'ACTUAL_ROOT_SERIAL_FOCUSED_CHECKS_V1.json'
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    assert path.read_bytes()==raw
    print(json.dumps(pin(path)))

def main():
    source=HERE/'k3_owned_capacity_preflight61_72_ROOT_serial_v1.py'
    tree=ast.parse(source.read_bytes());namespace={'hashlib':hashlib,'json':json,'require':require}
    constants={'GIB','ROOT_OWNER','QIMG_OWNER','CURRENT_PEER_SOURCE_SHA','ROOT_SERIAL_SOURCE_SHA','ROOT_SERIAL_MEMBER_PINS'}
    functions={'current_ROOT_serial_scope','current_peer_public_source'}
    nodes=[node for node in tree.body if (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets))
        or isinstance(node,ast.FunctionDef) and node.name in functions]
    require(len(nodes)==8,'only two pure functions and six exact constants executed')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pure_actual_ROOT_join','exec'),namespace)
    proof_path=HERE/'ROOT_CURRENT_SERIAL_H3_AUDIO_ACE_PUBLIC_PROOF_V1.json'
    proof=json.loads(proof_path.read_bytes())
    peer_path=Path(proof['retained_completed_phase_proof']['path'])
    peer=json.loads(peer_path.read_bytes());require(sha(peer_path.read_bytes())==namespace['CURRENT_PEER_SOURCE_SHA'],'complete original peer proof must match')
    sources={'ROOT_current_serial_scope':proof,'current_peer_holder_source':peer}
    request={'sources':{role:{'name':path.name,'sha256':sha(path.read_bytes())} for role,path in
        [('ROOT_current_serial_scope',proof_path),('current_peer_holder_source',peer_path)]}}
    namespace['current_peer_public_source'](sources,request)
    actual=namespace['current_ROOT_serial_scope'](sources,request)
    require((actual['RAM_bytes'],actual['CPU_threads'],actual['new_ROOT_write_bytes'])==(17179869184,4,9663676416),'exact serial envelope once')
    results=[{'case':'complete authentic eight public raws plus existing full ROOT1075-only terminal proof','pass':True}]
    # Exact packet pin refusal, without relaxing any hash checks.
    bad_request=copy.deepcopy(request);bad_request['sources']['ROOT_current_serial_scope']['sha256']='0'*64
    try:namespace['current_ROOT_serial_scope'](sources,bad_request)
    except RuntimeError as error:results.append({'case':'wrong source packet fullSHA','pass':True,'refusal':str(error)})
    else:raise AssertionError('wrong proof SHA accepted')
    # Pure semantic guard tests use actual raw metadata with one changed field.
    # Only the function's in-memory member expectation is adjusted to reach that
    # guard; these perturbations are never authored as an actual source packet.
    original_pins=copy.deepcopy(namespace['ROOT_SERIAL_MEMBER_PINS'])
    for label,role,key,value in [('wrong prepared ACE cap','ACE_actual_staged_request','caps',{'memory_bytes':12884901888,'memory_swap_bytes':12884901888,'cpu_threads':4,'output_reserve_bytes':8589934592,'gpu_device_requests':[]}),
        ('wrong genuine ROOT owner','ROOT_serial_adoption','native','01a11108-009c-72f3-b2ba-e0d48fca1435')]:
        wrong=copy.deepcopy(sources);row=wrong['ROOT_current_serial_scope']['members'][role]
        body=json.loads(row['raw_utf8']);body[key]=value;raw=encode(body)
        row.update(raw_utf8=raw.decode(),bytes=len(raw),sha256=sha(raw))
        namespace['ROOT_SERIAL_MEMBER_PINS']=copy.deepcopy(original_pins)
        namespace['ROOT_SERIAL_MEMBER_PINS'][role]={'sha256':row['sha256'],'bytes':row['bytes']}
        try:namespace['current_ROOT_serial_scope'](wrong,request)
        except RuntimeError as error:results.append({'case':label,'pass':True,'refusal':str(error),'test_only_adjusted_member_expectation_to_reach_semantic_guard':True})
        else:raise AssertionError(label+' accepted')
    namespace['ROOT_SERIAL_MEMBER_PINS']=original_pins
    old=ast.parse((BASE/'packet_source/k3_owned_capacity_preflight61_72_template_v3.py').read_bytes())
    def functions_of(t):return {n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    before=functions_of(old);after=functions_of(tree)
    changed=[name for name in before if before[name]!=after[name]]
    require(changed==['run'] and set(after)-set(before)=={'current_ROOT_serial_scope'},'existing classifier/source/unknown/math guards preserved')
    for path in [HERE/'bind_actual61_ROOT_serial_v1.py',HERE/'launch/launch_actual61_ROOT_serial_v1.py',*list((HERE/'launch').glob('*_remote.py')),source]:compile(path.read_bytes(),str(path),'exec')
    seal({'schema':'neural_forge.k3_ROOT_serial_actual_public_metadata_focused_checks.v1','result':'PASS_SOURCE_ONLY',
        'source':pin(source),'proof':pin(proof_path),'actual_ROOT1075_only_terminal_join':request['ROOT1075_exact_completed_phase_future_once'],
        'actual_serial_scope':actual,'cases':results,'existing_changed_ASTs':changed,'new_function_ASTs':['current_ROOT_serial_scope'],
        'classify_and_unknown_actor_guards_unchanged':True,'facts_and_host_changes':0,'actual60':None,
        'runtime_or_DB_Relay_or_resource_actions':False,'candidate_main_or_module_import_executed':False,
        'semantic_perturbations_test_only_not_actual_claims':True,'new_source_normal_promoted_commit':None})

if __name__=='__main__':main()
