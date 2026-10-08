"""Fixed three-phase NEXT-only source checks; no runtime imports or fixtures."""
from pathlib import Path
import argparse,ast,copy,hashlib,json,zipfile
ROSTER_SHA='0a97905c19a565a81877cc2ef18f1469abc6dfc8cccd39bafeaae9609e8c8eb9'
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def require(ok,message):
    if ok is not True:raise RuntimeError(message)
def functions(raw):
    out={}
    for n in ast.parse(raw).body:
        if isinstance(n,ast.FunctionDef):out[n.name]=ast.dump(n,include_attributes=False)
        elif isinstance(n,ast.ClassDef):
            for child in n.body:
                if isinstance(child,ast.FunctionDef):out[n.name+'.'+child.name]=ast.dump(child,include_attributes=False)
    return out
def run(root):
    raw=(root/'FUTURE_ROOT_TEXT_REVIEWED_SOURCE_ROSTER_V1.json').read_bytes();assert sha(raw)==ROSTER_SHA
    roster=json.loads(raw);assert roster['source_only'] is True and roster['deployable'] is False
    assert all(roster[k] is None for k in ('actual_prior_terminal','next_UUID','filesystem','allocation','capacity_admission'))
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual==set(roster['members'])|{'FUTURE_ROOT_TEXT_REVIEWED_SOURCE_ROSTER_V1.json','check_future_ROOT_text_promoted_packet_v1.py'}
    bysha={};compiled=0
    for name,row in roster['members'].items():
        p=root/name;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root)
        b=p.read_bytes();assert sha(b)==row['sha256'] and len(b)==row['bytes'],name
        bysha.setdefault(row['sha256'],b)
        if name.endswith('.py'):compile(b,name,'exec');compiled+=1
    proof=json.loads(bysha['3f12e995d8715d56700d1b32954876808f544ab5244917a033b147fb2e2152ab'])
    assert len(proof['members'])==11
    for row in proof['members'].values():
        b=row['raw_utf8'].encode();assert sha(b)==row['sha256'] and len(b)==row['bytes'];json.loads(b)
    source=bysha[roster['selected_preflight_sha256']];tree=ast.parse(source)
    namespace={'hashlib':hashlib,'json':json,'require':require}
    cs={'GIB','ROOT_OWNER','QIMG_OWNER','CURRENT_PEER_SOURCE_SHA','ROOT_SERIAL_SOURCE_SHA','ROOT_SERIAL_MEMBER_PINS'}
    fs={'current_ROOT_serial_scope','current_peer_public_source'}
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in cs for t in n.targets) or isinstance(n,ast.FunctionDef) and n.name in fs]
    assert len(nodes)==8;exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_pure_ROOT_text_join','exec'),namespace)
    peer=json.loads(bysha[namespace['CURRENT_PEER_SOURCE_SHA']])
    sources={'ROOT_current_serial_scope':proof,'current_peer_holder_source':peer}
    request={'sources':{'ROOT_current_serial_scope':{'name':'ROOT_CURRENT_SERIAL_H3_AUDIO_ACE_TEXT_PUBLIC_PROOF_V2.json','sha256':namespace['ROOT_SERIAL_SOURCE_SHA']},'current_peer_holder_source':{'name':'CURRENT_QIMG_NATIVE_AND_ROOT1075_PUBLIC_PROOF_V3.json','sha256':namespace['CURRENT_PEER_SOURCE_SHA']}}}
    namespace['current_peer_public_source'](sources,request);scope=namespace['current_ROOT_serial_scope'](sources,request)
    assert (scope['RAM_bytes'],scope['CPU_threads'],scope['new_ROOT_write_bytes'])==(17179869184,4,10737418240)
    bad=copy.deepcopy(request);bad['sources']['ROOT_current_serial_scope']['sha256']='0'*64
    try:namespace['current_ROOT_serial_scope'](sources,bad)
    except RuntimeError:pass
    else:raise AssertionError('wrong source SHA accepted')
    original=copy.deepcopy(namespace['ROOT_SERIAL_MEMBER_PINS'])
    for key,value in [('current_text_memory_bytes',6*(1<<30)),('native','01a11108-009c-72f3-b2ba-e0d48fca1435')]:
        wrong=copy.deepcopy(sources);row=wrong['ROOT_current_serial_scope']['members']['ROOT_text_serial_adoption'];v=json.loads(row['raw_utf8']);v[key]=value;b=enc(v)
        row.update(raw_utf8=b.decode(),sha256=sha(b),bytes=len(b));namespace['ROOT_SERIAL_MEMBER_PINS']=copy.deepcopy(original);namespace['ROOT_SERIAL_MEMBER_PINS']['ROOT_text_serial_adoption']={'sha256':sha(b),'bytes':len(b)}
        try:namespace['current_ROOT_serial_scope'](wrong,request)
        except RuntimeError:pass
        else:raise AssertionError('wrong adopted text semantic accepted')
    old=functions(bysha[roster['parent_preflight_sha256']]);new=functions(source)
    assert set(old)==set(new) and {n for n in old if old[n]!=new[n]}=={'current_ROOT_serial_scope','run'}
    q=root/'recovery_oct07/nas61_launch_integration_v1/packet_source/evidence'
    names=json.loads((q/'SOURCE_ONLY_PROPOSAL_VALIDATION_V2.json').read_bytes())['original_qualified_functions_AST_unchanged'];assert len(names)==len(set(names))==99
    qualified=functions((q/'k3_dense_pass_9086.py').read_bytes());native=functions(bysha[roster['native_sha256']]);assert all(native[n]==qualified[n] for n in names)
    assert roster['ROOT_serial_RAM_CPU_write_once']==[17179869184,4,10737418240] and roster['conservative_whole_ROOT_future_bytes_once']==49*(1<<30)
    expected={61:166556705792,73:180943168512,85:40692831840};assert roster['writer_bytes_once']==expected[roster['phase_start']]
    assert roster['NAS_if_selected_common_once']==692026605568+expected[roster['phase_start']]
    assert roster['Lexar_if_selected_own_QIMG_head_once']==expected[roster['phase_start']]+40*(1<<30)
    for name in roster['independent_source_reviews']:
        assert 'PASS' in json.dumps(json.loads((root/name).read_bytes()))
    print(json.dumps({'PASS':True,'source_only':True,'phase_start':roster['phase_start'],'raw_member_checks':len(roster['members']),'compile_only_modules':compiled,'authentic_ROOT_public_raw_members':11,'existing_pure_source_cases':4,'qualified_other_ASTs_unchanged':99,'actual_binding_or_admission_or_runtime_actions':False}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate-root',required=True,type=Path);run(p.parse_args().candidate_root.resolve())
