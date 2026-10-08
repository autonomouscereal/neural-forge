"""Fixed NEXT-only full source checks and the three existing pure metadata cases."""
from pathlib import Path
import argparse,ast,base64,copy,hashlib,json,zlib
ROSTER_SHA='0ae51b0fd0d63ce830ef2b7906890c26baaf6699604987a661b1fe77f234fa01'
def sha(b):return hashlib.sha256(b).hexdigest()
def require(ok,message):
    if ok is not True:raise RuntimeError(message)
def integer(v,label):
    require(type(v) is int and v>=0,label);return v
def functions(raw):
    out={}
    for n in ast.parse(raw).body:
        if isinstance(n,ast.FunctionDef):out[n.name]=ast.dump(n,include_attributes=False)
        elif isinstance(n,ast.ClassDef):
            for c in n.body:
                if isinstance(c,ast.FunctionDef):out[n.name+'.'+c.name]=ast.dump(c,include_attributes=False)
    return out
def run(root):
    raw=(root/'NEXT_QIMG_COMPUTE_CLOSED_SOURCE_ROSTER_V1.json').read_bytes();assert sha(raw)==ROSTER_SHA
    roster=json.loads(raw);assert roster['source_only'] is True and roster['deployable'] is False
    assert all(roster[k] is None for k in ('actual_prior_terminal','next_UUID','filesystem','allocation','capacity_admission'))
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual==set(roster['members'])|{'NEXT_QIMG_COMPUTE_CLOSED_SOURCE_ROSTER_V1.json','check_NEXT_QIMG_promoted_packet_v1.py'}
    bysha={};compiled=0
    for name,row in roster['members'].items():
        p=root/name;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root)
        b=p.read_bytes();assert sha(b)==row['sha256'] and len(b)==row['bytes'],name
        bysha.setdefault(row['sha256'],b)
        if name.endswith('.py'):compile(b,name,'exec');compiled+=1
    source=bysha[roster['selected_preflight_sha256']];tree=ast.parse(source)
    old=functions(bysha[roster['parent_preflight_sha256']]);new=functions(source)
    assert set(new)-set(old)=={'qimg_native_compute_closure'} and not set(old)-set(new)
    assert {n for n in old if old[n]!=new[n]}=={'classify','run'}
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qimg_native_compute_closure')
    ns={'require':require,'integer':integer,'hashlib':hashlib,'json':json,'QIMG_OWNER':'codex-native-01a111e4-2e50-7eb0-916a-74e1058b3a9c'}
    exec(compile(ast.Module(body=[f],type_ignores=[]),'existing_pure_QIMG_source_join','exec'),ns)
    sources={};pins={}
    for role,row in roster['QIMG_proof_roles'].items():
        b=bysha[row['sha256']];assert len(b)<=16777216
        sources[role]=json.loads(b);pins[role]={'name':row['name'],'sha256':row['sha256'],'bytes':len(b)}
    request={'sources':pins,'source_metadata_bytes_except_QIMG_closure':roster['existing_non_QIMG_source_metadata_bytes']}
    result=ns['qimg_native_compute_closure'](sources,request)
    assert result['RAM_bytes']==result['CPU_threads']==0 and result['raw_member_count']==62
    assert result['retained_disk_credit_bytes']==0 and result['foreign_provider_scope_discharged'] is False
    assert result['historical_committed_row_is_current_observation'] is False
    bad=copy.deepcopy(sources);row=bad['QIMG_native_terminal_raw_part_1']['members'][0]
    b=zlib.decompress(base64.b64decode(row['raw_zlib_base64']));b=b[:-1]+bytes([b[-1]^1]);row['raw_zlib_base64']=base64.b64encode(zlib.compress(b,9)).decode()
    try:ns['qimg_native_compute_closure'](bad,request)
    except RuntimeError:pass
    else:raise AssertionError('changed authentic terminal raw accepted')
    bad=copy.deepcopy(sources);row=bad['QIMG_native_compute_closure']['outer']['release'];v=json.loads(zlib.decompress(base64.b64decode(row['raw_zlib_base64'])));v['fresh_committed_released_row']['owner']='wrong-owner';row['raw_zlib_base64']=base64.b64encode(zlib.compress((json.dumps(v)+'\n').encode(),9)).decode()
    try:ns['qimg_native_compute_closure'](bad,request)
    except RuntimeError:pass
    else:raise AssertionError('changed authentic normal release owner accepted')
    q=root/'recovery_oct07/nas61_launch_integration_v1/packet_source/evidence'
    names=json.loads((q/'SOURCE_ONLY_PROPOSAL_VALIDATION_V2.json').read_bytes())['original_qualified_functions_AST_unchanged'];assert len(names)==len(set(names))==99
    qualified=functions((q/'k3_dense_pass_9086.py').read_bytes());native=functions(bysha[roster['native_sha256']]);assert all(native[n]==qualified[n] for n in names)
    for name in roster['independent_source_reviews']:assert 'PASS' in json.dumps(json.loads((root/name).read_bytes()))
    assert roster['ROOT_serial_RAM_CPU_write_once']==[17179869184,4,10737418240] and roster['conservative_whole_ROOT_future_bytes_once']==49*(1<<30)
    assert roster['QIMG_retained_Lexar_ROOT_write_once']==[24*(1<<30),2*(1<<30)] and roster['QIMG_compute_once']==[0,0]
    expected={61:166556705792,73:180943168512,85:40692831840};assert roster['writer_bytes_once']==expected[roster['phase_start']]
    assert roster['NAS_if_selected_common_once']==692026605568+expected[roster['phase_start']]
    assert roster['Lexar_if_selected_own_QIMG_head_once']==expected[roster['phase_start']]+40*(1<<30)
    print(json.dumps({'PASS':True,'source_only':True,'phase_start':roster['phase_start'],'fullraw_member_checks':len(roster['members']),'compile_only_modules':compiled,'existing_pure_actual_metadata_cases':3,'actual_QIMG_terminal_raw_count':result['raw_member_count'],'actual_raw_decoded_bytes':result['decoded_all_proof_bytes'],'qualified_other_ASTs_unchanged':99,'actual_binding_or_admission_or_runtime_actions':False}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate-root',required=True,type=Path);run(p.parse_args().candidate_root.resolve())
