"""Bounded NEXT-only later caller composition; source checks, never runtime."""
import argparse, ast, copy, hashlib, json, os
from pathlib import Path

HERE=Path(__file__).resolve().parent
RECOVERY=HERE.parent
OLD=RECOVERY/'later73_85_ROOT_text_scope_v1'
OLD_SHA='a77c42cccb7629b5e0db1154f8cf96e29af07f08f73517a4f1b9ee4fa45329c2'
SHARED=RECOVERY/'recovery_resource_review/NEXT_QIMG_compute_closure_v1'
SHARED_SHA='574caa6f5f5dd8c88a1a4c55a8056601b09ca450055d1060e25927e6fe9af5cd'

def sha(raw): return hashlib.sha256(raw).hexdigest()
def encode(v): return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def pin(path):
    raw=path.read_bytes();return {'path':str(path.resolve()),'sha256':sha(raw),'bytes':len(raw)}
def read_pin(row):
    path=Path(row['path']);assert all(pin(path)[k]==row[k] for k in ('path','sha256','bytes'));return path.read_bytes()
def write(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    assert path.read_bytes()==raw
def one(s,a,b): assert s.count(a)==1,(s.count(a),a);return s.replace(a,b)
def assign(s,name,expr):
    n=[n for n in ast.parse(s).body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name]
    assert len(n)==1,name;return one(s,ast.get_source_segment(s,n[0]),name+'='+expr)
def functions(raw):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--shared61-contract',type=Path,required=True);p.add_argument('--shared61-contract-sha256',required=True);a=p.parse_args()
    assert pin(a.shared61_contract)['sha256']==a.shared61_contract_sha256
    shared61=json.loads(a.shared61_contract.read_bytes());assert shared61['metadata_count_before_allocation']==70 and shared61['script_count']==5
    read_pin(shared61['binder'])
    sharedpath=SHARED/'NEXT_QIMG_COMPUTE_CLOSED_PREFLIGHT_SOURCE_COMPOSITION_V1.json';assert pin(sharedpath)['sha256']==SHARED_SHA
    shared=json.loads(sharedpath.read_bytes())
    oldpath=OLD/'LATER_ROOT_TEXT_SOURCE_COMPOSITION_V2.json';assert pin(oldpath)['sha256']==OLD_SHA
    old=json.loads(oldpath.read_bytes())
    roles={role:dict(row,name=Path(row['path']).name) for role,row in shared['proof_roles'].items()}
    assert roles==shared61['QIMG_native_compute_proof_roles']
    assert len(roles)==3
    for row in roles.values(): read_pin(row)
    phases=[];observations=[];previous=shared61['binder']
    sharedhelper=(Path(shared61['helper_manifest']['path']).parent/'read_fresh_prospective61_remote.py').read_text()
    extra=sharedhelper[sharedhelper.index("    and capacity['QIMG_pending_scope_once']"):sharedhelper.index("    and capacity['disk_components']['K3_write_destination']")]
    assert "['RAM_bytes']==0" in extra and "['cpu_threads']==0" in extra
    for phase in old['phases']:
        lo,hi=phase['window'];label=f'{lo}_{hi}';count=phase['metadata_before_A']+3
        leaf=HERE/f'phase{label}_QIMG_compute_closed_v1';assert not leaf.exists();leaf.mkdir()
        oldleaf=Path(phase['binder']['path']).parent
        pre=next(row for row in shared['phases'] if row['window']==f'{lo}-{hi}')
        selected=json.loads(read_pin(phase['selected_source_roster']));assert selected['preflight']==pre['parent']
        pre_raw=read_pin(pre['candidate']);compile(pre_raw,pre['candidate']['path'],'exec')
        old_f=functions(read_pin(pre['parent']));new_f=functions(pre_raw)
        assert set(new_f)-set(old_f)=={'qimg_native_compute_closure'} and not(set(old_f)-set(new_f))
        assert all(new_f[k]==old_f[k] for k in old_f if k not in ('classify','run'))
        selected['preflight']=pre['candidate']
        for role,row in selected.items():compile(read_pin(row),row['path'],'exec')
        roster=leaf/'SELECTED_PHASE_QIMG_COMPUTE_CLOSED_SOURCE_PINS_V1.json';write(roster,encode(selected))
        for row in roles.values():write(leaf/row['name'],read_pin(row))
        rootproof=phase['source_proof'];write(leaf/Path(rootproof['path']).name,read_pin(rootproof))
        for name in [f'validate_actual{lo-1}_source_metadata_v2.py',f'ACTUAL{lo}_INPUTS_UNRESOLVED_V2.json']:write(leaf/name,(oldleaf/name).read_bytes())
        controls={}
        for name,row in phase['controls'].items():
            target=leaf/'controls'/name;write(target,read_pin(row));controls[name]=pin(target);assert controls[name]['sha256']==row['sha256']
        source=read_pin(phase['binder']).decode()
        local=f'D:/codex-owned-source-receipts/k3-later73-92-QIMG-compute-closed-source-only/phase{label}'
        source=assign(source,'LOCAL_OUTPUTS','Path('+repr(local)+')')
        source=assign(source,'SOURCE_PINS_SHA',repr(sha(roster.read_bytes())))
        source=assign(source,'PREVIOUS_BINDER','Path('+repr(previous['path'])+')');source=assign(source,'PREVIOUS_BINDER_SHA',repr(previous['sha256']))
        source=source.replace('SELECTED_PHASE_ROOT_TEXT_SOURCE_PINS_V1.json',roster.name)
        source=one(source,'\nRUN=None\n','\nQIMG_CLOSED_PREFLIGHT_SHA='+repr(pre['candidate']['sha256'])+'\nQIMG_CLOSURE_PROOF_ROLES='+repr(roles)+'\nRUN=None\n')
        priorcount=phase['metadata_before_A']
        source=source.replace(f'len(roster)=={priorcount}',f'len(roster)=={count}').replace(f"len(closure['sources']) == {priorcount}",f"len(closure['sources']) == {count}")
        source=source.replace(f'complete source-bound{priorcount}',f'complete source-bound{count}').replace(f'whole exact{priorcount}',f'whole exact{count}').replace(f'exact previous{priorcount-3} plus three',f'exact previous{count-3} plus three')
        insertion="""    for role,ref in QIMG_CLOSURE_PROOF_ROLES.items():
        row=closure['sources'].get(role)
        require(isinstance(row,dict) and {key:row.get(key) for key in ('name','sha256','bytes')}=={key:ref[key] for key in ('name','sha256','bytes')},'complete exact inherited QIMG native compute proof roles required')
        require(read_raw(bundle/'production_control'/row['name'],row['sha256'])==raw_ref({'path':str(HERE/ref['name']),'sha256':ref['sha256']}),'retained QIMG native compute proof full bytes differ')
    require(closure['source_binding']['QIMG_closed_compute_preflight_template_sha256']==QIMG_CLOSED_PREFLIGHT_SHA
        and closure['source_binding']['QIMG_closed_compute_preflight_normal_promoted_commit'] is None,'honest NEXT-only unpromoted current QIMG source provenance required')
"""
        source=one(source,"    read_raw(bundle/'mounts.json', closure['source_binding']['mounts_sha256'])",insertion+"    read_raw(bundle/'mounts.json', closure['source_binding']['mounts_sha256'])")
        source=one(source,"    files['production_control/source_closure.json']=encode(closure)","    closure['source_binding'].update(QIMG_closed_compute_preflight_template_sha256=QIMG_CLOSED_PREFLIGHT_SHA,\n        QIMG_closed_compute_preflight_normal_promoted_commit=None,QIMG_closed_compute_source_only_revision=True,\n        QIMG_closed_compute_public_proof_roles=QIMG_CLOSURE_PROOF_ROLES)\n    files['production_control/source_closure.json']=encode(closure)")
        compile(source,'NEXT later QIMG binder','exec');bp=leaf/f'bind_actual{label}_QIMG_compute_closed_v1.py';write(bp,source.encode());previous=pin(bp)
        helpers={};names={}
        for name,row in phase['helpers'].items():
            text=read_pin(row).decode();newname=name.replace('ROOT_text_v1','QIMG_compute_closed_v1');assert newname!=name;names[name]=newname
            text=text.replace(f"len(CLOSURE['sources'])=={priorcount}",f"len(CLOSURE['sources'])=={count}").replace(f'len(verified)=={priorcount+5}',f'len(verified)=={count+5}').replace(f'whole exact{priorcount}',f'whole exact{count}')
            if name.startswith('read_fresh_prospective'):
                anchor="    and capacity['historical_ROOT12GiB4CPU2GiB_pending_replaced_once']"
                text=one(text,anchor,extra+anchor)
            assert functions(text)==functions(read_pin(row))
            compile(text,newname,'exec');hp=leaf/'launch'/newname;write(hp,text.encode());helpers[newname]=pin(hp)
        manifest=leaf/'launch/LAUNCH_QIMG_COMPUTE_CLOSED_HELPER_SOURCE_PINS_V1.json';write(manifest,encode({name:{k:row[k] for k in ('sha256','bytes')} for name,row in helpers.items()}))
        wrapper=read_pin(phase['wrapper']).decode();wrapper=assign(wrapper,'LOCAL_OUTPUTS','pathlib.Path('+repr(local)+')');wrapper=assign(wrapper,'BINDER','BIND_ROOT/'+repr(bp.name));wrapper=assign(wrapper,'BINDER_SHA',repr(previous['sha256']));wrapper=assign(wrapper,'HELPER_MANIFEST_SHA',repr(sha(manifest.read_bytes())))
        wrapper=wrapper.replace('LAUNCH_ROOT_TEXT_HELPER_SOURCE_PINS_V1.json',manifest.name)
        for oldname,newname in names.items():wrapper=wrapper.replace(oldname,newname)
        compile(wrapper,'NEXT later QIMG wrapper','exec');wp=leaf/'launch'/f'launch_actual{label}_QIMG_compute_closed_v1.py';write(wp,wrapper.encode())
        for raw in (source,wrapper):
            n=next(n for n in ast.parse(raw).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LOCAL_OUTPUTS' for t in n.targets));assert ast.literal_eval(n.value.args[0])==local
        record=copy.deepcopy(phase);record.update(binder=previous,wrapper=pin(wp),controls=controls,helpers=helpers,helper_manifest=pin(manifest),selected_source_roster=pin(roster),source_proof=pin(leaf/Path(rootproof['path']).name),
            QIMG_native_compute_proof_roles={role:pin(leaf/ref['name']) for role,ref in roles.items()},QIMG_pending_compute_once={'RAM_bytes':0,'CPU_threads':0,'fresh_no_successor_holder_frame_required':True},
            QIMG_ROOT_future_bytes_once=2*(1<<30),QIMG_retained_Lexar_future_bytes_once=24*(1<<30),metadata_before_A=count,metadata_with_A=count+1,
            normal_promoted_commit=None,independent_review=None,recurring_PG_gate_or_new_manual_ACK_required=False)
        write(leaf/'QIMG_COMPUTE_CLOSED_PHASE_SOURCE_CONTRACT_V1.json',encode(record));phases.append(record)
        observations.append({'window':[lo,hi],'metadata_before_A':count,'metadata_with_A':count+1,'scripts':5,'old_other10roles_byte_identical':True,'six_controls_byte_identical':True,'math_terminal92_disk_provider_facts_host_unchanged':True,'only_shared_preflight_classify_run_and_new_purejoin_changed':True,'inherited3proofroles_fullhash_and_extent_guard':True,'new_binder_wrapper_localpath_equal':local})
    result=copy.deepcopy(old);result.update(schema='neural_forge.k3_later73_85_NEXT_QIMG_compute_closed_source_composition.v1',phases=phases,
        prior_normal_tested_commit='0bbe9d52bc6690d5b394fa5d4f006c69f2913abf',shared_QIMG_preflight_composition=pin(sharedpath),shared61_QIMG_whole_contract=pin(a.shared61_contract),
        QIMG_pending_compute_once={'RAM_bytes':0,'CPU_threads':0,'fresh_holder_absence_checked_by_classify':True},QIMG_ROOT_future_bytes_once=2*(1<<30),QIMG_retained_Lexar_future_bytes_once=24*(1<<30),
        normal_promoted_commit=None,independent_review=None,runtime_actions=False,recurring_PG_gate_or_new_manual_ACK_required=False)
    cp=HERE/'LATER_QIMG_COMPUTE_CLOSED_SOURCE_COMPOSITION_V1.json';write(cp,encode(result))
    fp=HERE/'LATER_QIMG_COMPUTE_CLOSED_FOCUSED_CHECK_V1.json';write(fp,encode({'schema':'neural_forge.k3_later_NEXT_QIMG_source_check.v1','verdict':'PASS_SOURCE_ONLY_PENDING_INDEPENDENT_REVIEW','composition':pin(cp),'observations':observations,'fixtures_candidate_imports_runtime_remote_Relay_model_actions':False}))
    print(json.dumps({'composition':pin(cp),'focused_check':pin(fp),'phases':[{k:r[k] for k in ('window','binder','wrapper','metadata_before_A')} for r in phases]}))

if __name__=='__main__':main()
