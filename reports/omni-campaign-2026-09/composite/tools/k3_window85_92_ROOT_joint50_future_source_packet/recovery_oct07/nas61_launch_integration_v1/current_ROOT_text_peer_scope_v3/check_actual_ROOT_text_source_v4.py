"""Four focused authentic public-source joins, without candidate module imports."""
from pathlib import Path
import ast,copy,hashlib,json,os
HERE=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encode(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def require(ok,message):
    if ok is not True:raise RuntimeError(message)
def pin(path):
    raw=path.read_bytes();return {'path':str(path.resolve()),'sha256':sha(raw),'bytes':len(raw)}

def main():
    source=HERE/'k3_owned_capacity_preflight61_72_ROOT_text_v2.py'
    tree=ast.parse(source.read_bytes());namespace={'hashlib':hashlib,'json':json,'require':require}
    constants={'GIB','ROOT_OWNER','QIMG_OWNER','CURRENT_PEER_SOURCE_SHA','ROOT_SERIAL_SOURCE_SHA','ROOT_SERIAL_MEMBER_PINS'}
    functions={'current_ROOT_serial_scope','current_peer_public_source'}
    nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)) or isinstance(n,ast.FunctionDef) and n.name in functions]
    require(len(nodes)==8,'execute only two pure joins and six exact constants')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pure_actual_ROOT_text_join','exec'),namespace)
    proof_path=HERE/'ROOT_CURRENT_SERIAL_H3_AUDIO_ACE_TEXT_PUBLIC_PROOF_V2.json'
    proof=json.loads(proof_path.read_bytes());require(len(proof['members'])==11,'all eleven complete public raws')
    peer_path=Path(proof['retained_completed_phase_proof']['path']);peer=json.loads(peer_path.read_bytes())
    require(sha(peer_path.read_bytes())==namespace['CURRENT_PEER_SOURCE_SHA'],'retained actual terminal peer proof')
    sources={'ROOT_current_serial_scope':proof,'current_peer_holder_source':peer}
    request={'sources':{role:{'name':p.name,'sha256':sha(p.read_bytes())} for role,p in [('ROOT_current_serial_scope',proof_path),('current_peer_holder_source',peer_path)]}}
    namespace['current_peer_public_source'](sources,request)
    actual=namespace['current_ROOT_serial_scope'](sources,request)
    require((actual['RAM_bytes'],actual['CPU_threads'],actual['new_ROOT_write_bytes'])==(17179869184,4,10737418240),'serial16/4 and ACE8/audio1/text1 writes once')
    results=[{'case':'complete authentic eleven public raws and full prior ROOT1075-only terminal proof','pass':True}]
    bad=copy.deepcopy(request);bad['sources']['ROOT_current_serial_scope']['sha256']='0'*64
    try:namespace['current_ROOT_serial_scope'](sources,bad)
    except RuntimeError as error:results.append({'case':'wrong full source packet SHA','pass':True,'refusal':str(error)})
    else:raise AssertionError('wrong packet SHA accepted')
    original=copy.deepcopy(namespace['ROOT_SERIAL_MEMBER_PINS'])
    for label,key,value in [('wrong adopted text cap','current_text_memory_bytes',6*(1<<30)),('wrong genuine text owner','native','01a11108-009c-72f3-b2ba-e0d48fca1435')]:
        wrong=copy.deepcopy(sources);row=wrong['ROOT_current_serial_scope']['members']['ROOT_text_serial_adoption']
        body=json.loads(row['raw_utf8']);body[key]=value;raw=encode(body)
        row.update(raw_utf8=raw.decode(),bytes=len(raw),sha256=sha(raw))
        namespace['ROOT_SERIAL_MEMBER_PINS']=copy.deepcopy(original)
        namespace['ROOT_SERIAL_MEMBER_PINS']['ROOT_text_serial_adoption']={'sha256':sha(raw),'bytes':len(raw)}
        try:namespace['current_ROOT_serial_scope'](wrong,request)
        except RuntimeError as error:results.append({'case':label,'pass':True,'refusal':str(error),'test_only_member_expectation_adjusted':True})
        else:raise AssertionError(label+' accepted')
    namespace['ROOT_SERIAL_MEMBER_PINS']=original
    old=ast.parse((HERE.parent/'current_ROOT_peer_scope_v1/k3_owned_capacity_preflight61_72_ROOT_serial_v1.py').read_bytes())
    def funcs(t):return {n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    before=funcs(old);after=funcs(tree);changed=[n for n in before if before[n]!=after[n]]
    require(set(changed)=={'run','current_ROOT_serial_scope'} and set(after)==set(before),'only existing ROOT pure join and run diagnostic changed')
    wrapper=HERE/'launch/launch_actual61_ROOT_text_v3.py';binder=HERE/'bind_actual61_ROOT_text_v3.py'
    for p in [source,binder,wrapper,*list((HERE/'launch').glob('*_remote.py'))]:compile(p.read_bytes(),str(p),'exec')
    manifest=json.loads((HERE/'launch/LAUNCH_HELPER_SOURCE_PINS_V1.json').read_bytes())
    for name,row in manifest.items():
        p=HERE/'launch'/name;require(pin(p)['sha256']==row['sha256'] and p.stat().st_size==row['bytes'],'complete helper pin roster')
    require("new_ROOT_write_bytes']==10737418240" in (HERE/'launch/read_fresh_prospective61_remote.py').read_text(),'actual readback accepts new10GiB scope')
    value={'schema':'neural_forge.k3_ROOT_text_actual_public_metadata_focused_checks.v3','result':'PASS_SOURCE_ONLY','source':pin(source),'proof':pin(proof_path),'binder':pin(binder),'wrapper':pin(wrapper),'actual_scope':actual,'cases':results,'existing_changed_ASTs':changed,'other_function_ASTs_unchanged':True,'candidate_main_or_module_import_executed':False,'semantic_perturbations_test_only_not_actual_claims':True,'actual60':None,'runtime_or_DB_Relay_or_resource_actions':False,'new_source_normal_promoted_commit':None}
    p=HERE/'ACTUAL_ROOT_TEXT_FOCUSED_CHECKS_V4.json';raw=encode(value)
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    require(p.read_bytes()==raw,'exclusive checks readback');print(json.dumps(pin(p)))

if __name__=='__main__':main()
