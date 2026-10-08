"""Exact NEXT source roster and the three existing pure joint50 metadata cases."""
from pathlib import Path
import argparse,ast,copy,hashlib,json
ROSTER_SHA='acc65dcef09427d4156773188c513e108d3267e84623f202e383d892ba4936ae'
def sha(b):return hashlib.sha256(b).hexdigest()
def require(ok,label):
    if ok is not True:raise RuntimeError(label)
def functions(raw):
    out={}
    for n in ast.parse(raw).body:
        if isinstance(n,ast.FunctionDef):out[n.name]=ast.dump(n,include_attributes=False)
        elif isinstance(n,ast.ClassDef):
            for c in n.body:
                if isinstance(c,ast.FunctionDef):out[n.name+'.'+c.name]=ast.dump(c,include_attributes=False)
    return out
def run(root):
    raw=(root/'NEXT_ROOT_JOINT50_FUTURE_SOURCE_ROSTER_V1.json').read_bytes()
    assert sha(raw)==ROSTER_SHA
    roster=json.loads(raw)
    assert roster['source_only'] is True and roster['deployable'] is False
    assert all(roster[k] is None for k in ('actual_prior_terminal','next_UUID','filesystem','allocation','capacity_admission'))
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual==set(roster['members'])|{'NEXT_ROOT_JOINT50_FUTURE_SOURCE_ROSTER_V1.json','check_NEXT_ROOT_joint50_promoted_packet_v1.py'}
    bysha={};compiled=0
    for name,row in roster['members'].items():
        p=root/name
        assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root)
        b=p.read_bytes();assert sha(b)==row['sha256'] and len(b)==row['bytes'],name
        bysha.setdefault(row['sha256'],b)
        if name.endswith('.py'):compile(b,name,'exec');compiled+=1
    source=bysha[roster['selected_preflight_sha256']]
    old=functions(bysha[roster['parent_preflight_sha256']]);new=functions(source)
    assert set(new)-set(old)=={'current_ROOT_joint50_future_scope'} and not set(old)-set(new)
    assert {n for n in old if old[n]!=new[n]}=={'run'}
    f=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='current_ROOT_joint50_future_scope')
    standalone=next(n for n in ast.parse(bysha[roster['standalone_consumer_sha256']]).body if isinstance(n,ast.FunctionDef))
    assert ast.dump(f,include_attributes=False)==ast.dump(standalone,include_attributes=False)
    ns={'require':require,'hashlib':hashlib,'json':json,'GIB':1<<30}
    exec(compile(ast.Module(body=[f],type_ignores=[]),'existing_pure_joint50_source_join','exec'),ns)
    row=roster['ROOT_joint50_proof_role'];b=bysha[row['sha256']]
    assert len(b)==row['bytes'] and len(b)<=16777216
    proof=json.loads(b);sources={'ROOT_current_joint50_serial_future':proof}
    request={'sources':{'ROOT_current_joint50_serial_future':row}}
    serial={'RAM_bytes':16*(1<<30),'CPU_threads':4,'new_ROOT_write_bytes':10*(1<<30)}
    result=ns['current_ROOT_joint50_future_scope'](sources,request,serial)
    assert result['additional_ROOT_write_bytes_once']==2*(1<<30)
    assert result['additional_Lexar_write_bytes_once']==28*(1<<30)
    assert result['additional_RAM_bytes']==result['additional_CPU_threads']==0
    assert result['conservative_whole_ROOT_required_bytes']==51*(1<<30)
    assert result['current_actor_adopted'] is False and result['prior_future_or_retained_disk_credit_bytes']==0
    bad=copy.deepcopy(request);bad['sources']['ROOT_current_joint50_serial_future']['sha256']='0'*64
    try:ns['current_ROOT_joint50_future_scope'](sources,bad,serial)
    except RuntimeError:pass
    else:raise AssertionError('wrong immutable source proof SHA accepted')
    bad=dict(serial,RAM_bytes=8*(1<<30))
    try:ns['current_ROOT_joint50_future_scope'](sources,request,bad)
    except RuntimeError:pass
    else:raise AssertionError('wrong retained serial envelope accepted')
    q=root/'recovery_oct07/nas61_launch_integration_v1/packet_source/evidence'
    names=json.loads((q/'SOURCE_ONLY_PROPOSAL_VALIDATION_V2.json').read_bytes())['original_qualified_functions_AST_unchanged']
    assert len(names)==len(set(names))==99
    qualified=functions((q/'k3_dense_pass_9086.py').read_bytes());native=functions(bysha[roster['native_sha256']])
    assert all(native[n]==qualified[n] for n in names)
    for name in roster['independent_source_reviews']:assert 'PASS' in json.dumps(json.loads((root/name).read_bytes()))
    assert roster['ROOT_serial_RAM_CPU_write_once']==[17179869184,4,10737418240]
    assert roster['conservative_whole_ROOT_future_bytes_once']==51*(1<<30)
    assert roster['ROOT_joint50_future_delta_once']=={'RAM_bytes':0,'CPU_threads':0,'ROOT_bytes':2*(1<<30),'Lexar_bytes':28*(1<<30)}
    assert roster['QIMG_retained_Lexar_ROOT_write_once']==[24*(1<<30),2*(1<<30)] and roster['QIMG_compute_once']==[0,0]
    expected={61:166556705792,73:180943168512,85:40692831840};writer=expected[roster['phase_start']]
    assert roster['writer_bytes_once']==writer and roster['NAS_if_selected_common_once']==692026605568+writer
    assert roster['Lexar_if_selected_own_QIMG_ROOT_head_once']==writer+68*(1<<30)
    print(json.dumps({'PASS':True,'source_only':True,'phase_start':roster['phase_start'],'fullraw_member_checks':len(roster['members']),'compile_only_modules':compiled,'existing_pure_actual_metadata_cases':3,'actual_joint50_raw_member_count':proof['member_count'],'qualified_other_ASTs_unchanged':99,'actual_binding_or_admission_or_runtime_actions':False}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate-root',required=True,type=Path);run(p.parse_args().candidate_root.resolve())
