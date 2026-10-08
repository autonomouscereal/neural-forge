"""Bounded NEXT-only later caller composition; source checks, never runtime."""
import argparse, ast, copy, hashlib, json, os
from pathlib import Path

HERE=Path(__file__).resolve().parent
RECOVERY=HERE.parent
OLD=RECOVERY/'later73_85_QIMG_compute_closure_v1'
OLD_SHA='260e62d48042368360cb2b143ffabbcf8ac93cef5a35fb36673983ccb1f4ec26'
SHARED=RECOVERY/'recovery_resource_review/NEXT_ROOT_joint50_future_scope_v1'
SHARED_SHA='f00070e9ca1c56b3ff2b5564c326870fdec3ad639161e315e04240fa52ad2891'

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
    shared61=json.loads(a.shared61_contract.read_bytes())
    assert shared61['metadata_count_before_allocation']==71 and shared61['script_count']==5
    read_pin(shared61['binder']);read_pin(shared61['helper_manifest'])
    sharedpath=SHARED/'NEXT_ROOT_JOINT50_PREFLIGHT_SOURCE_COMPOSITION_V1.json'
    assert pin(sharedpath)['sha256']==SHARED_SHA;shared=json.loads(sharedpath.read_bytes())
    oldpath=OLD/'LATER_QIMG_COMPUTE_CLOSED_SOURCE_COMPOSITION_V2.json'
    assert pin(oldpath)['sha256']==OLD_SHA;old=json.loads(oldpath.read_bytes())
    shared_review=RECOVERY/'NEXT_QIMG_compute_closure_independent_review_v1/NEXT61_ROOT_JOINT50_FUTURE_WHOLE_CALLER_INDEPENDENT_REVIEW_V1.json'
    assert pin(shared_review)['sha256']=='8de19949a95e5af77fa8a21027714dd7e4b18f6772425643dbcca0fd60bd05eb'
    role=shared['proof_role'];ref=dict(shared['proof'],name=Path(shared['proof']['path']).name)
    assert shared61['ROOT_joint50_future_proof_role']=={role:ref}
    read_pin(ref);read_pin(shared['consumer'])
    assert shared['future_delta_once']=={'CPU_threads':0,'RAM_bytes':0,'ROOT_bytes':2*(1<<30),'Lexar_bytes':28*(1<<30)}
    assert shared61['conservative_whole_ROOT_future_bytes_once']==51*(1<<30)
    sharedhelper=(Path(shared61['helper_manifest']['path']).parent/'read_fresh_prospective61_remote.py').read_text()
    extra=sharedhelper[sharedhelper.index("    and capacity['ROOT_joint50_source_scope_once']"):sharedhelper.index("    and capacity['disk_components']['K3_write_destination']")]
    extra=extra.replace("request['sources']",'sources')
    assert extra.count('ROOT_joint50_source_scope_once')==4 and "['additional_RAM_bytes']==0" in extra and "['additional_CPU_threads']==0" in extra
    phases=[];observations=[];previous=shared61['binder']
    for phase in old['phases']:
        lo,hi=phase['window'];label=f'{lo}_{hi}';priorcount=phase['metadata_before_A'];count=priorcount+1
        assert (lo,count,phase['script_count'],phase['mount_count']) in ((73,74,5,12),(85,77,5,13))
        leaf=HERE/f'phase{label}_ROOT_joint50_future_v1';assert not leaf.exists();leaf.mkdir()
        oldleaf=Path(phase['binder']['path']).parent
        pre=next(row for row in shared['phases'] if row['window']==label)
        selected=json.loads(read_pin(phase['selected_source_roster']));assert selected['preflight']==pre['parent']
        pre_raw=read_pin(pre['candidate']);compile(pre_raw,pre['candidate']['path'],'exec')
        old_f=functions(read_pin(pre['parent']));new_f=functions(pre_raw)
        assert set(new_f)-set(old_f)=={'current_ROOT_joint50_future_scope'} and not(set(old_f)-set(new_f))
        assert [k for k in old_f if new_f[k]!=old_f[k]]==['run']
        selected['preflight']=pre['candidate']
        for name,row in selected.items():
            compile(read_pin(row),row['path'],'exec')
            if name!='preflight':assert row==json.loads(read_pin(phase['selected_source_roster']))[name]
        roster=leaf/'SELECTED_PHASE_ROOT_JOINT50_FUTURE_SOURCE_PINS_V1.json';write(roster,encode(selected))
        write(leaf/ref['name'],read_pin(ref))
        rootproof=phase['source_proof'];write(leaf/Path(rootproof['path']).name,read_pin(rootproof))
        qimgroles={}
        for name,row in phase['QIMG_native_compute_proof_roles'].items():
            target=leaf/Path(row['path']).name;write(target,read_pin(row));qimgroles[name]=pin(target)
        for name in [f'validate_actual{lo-1}_source_metadata_v2.py',f'ACTUAL{lo}_INPUTS_UNRESOLVED_V2.json']:
            write(leaf/name,(oldleaf/name).read_bytes())
        controls={}
        for name,row in phase['controls'].items():
            target=leaf/'controls'/name;write(target,read_pin(row));controls[name]=pin(target)
            assert controls[name]['sha256']==row['sha256']
        source=read_pin(phase['binder']).decode()
        local=f'D:/codex-owned-source-receipts/k3-later73-92-ROOT-joint50-future-source-only/phase{label}'
        source=assign(source,'LOCAL_OUTPUTS','Path('+repr(local)+')')
        source=assign(source,'SOURCE_PINS_SHA',repr(sha(roster.read_bytes())))
        source=assign(source,'PREVIOUS_BINDER','Path('+repr(previous['path'])+')')
        source=assign(source,'PREVIOUS_BINDER_SHA',repr(previous['sha256']))
        source=assign(source,'QIMG_CLOSED_PREFLIGHT_SHA',repr(pre['candidate']['sha256']))
        source=one(source,'SELECTED_PHASE_QIMG_COMPUTE_CLOSED_SOURCE_PINS_V1.json',roster.name)
        source=one(source,'\nRUN=None\n','\nROOT_JOINT50_PROOF_ROLE='+repr(ref)+'\nRUN=None\n')
        source=source.replace(f'len(roster)=={priorcount}',f'len(roster)=={count}').replace(f"len(closure['sources']) == {priorcount}",f"len(closure['sources']) == {count}")
        source=source.replace(f'complete source-bound{priorcount}',f'complete source-bound{count}').replace(f'whole exact{priorcount}',f'whole exact{count}').replace(f'exact previous{priorcount-3} plus three',f'exact previous{count-3} plus three')
        insertion="""    joint=closure['sources'].get('ROOT_current_joint50_serial_future')
    require(isinstance(joint,dict) and all(joint.get(key)==ROOT_JOINT50_PROOF_ROLE[key] for key in ('name','sha256','bytes'))
        and closure['source_binding']['ROOT_joint50_future_public_proof']==ROOT_JOINT50_PROOF_ROLE
        and closure['source_binding']['ROOT_joint50_future_preflight_template_sha256']==QIMG_CLOSED_PREFLIGHT_SHA
        and closure['source_binding']['ROOT_joint50_future_preflight_normal_promoted_commit'] is None
        and closure['source_binding']['ROOT_joint50_future_source_only_revision'] is True,
        'exact honest inherited unpromoted ROOT joint50 future proof and source revision required')
    require(read_raw(bundle/'production_control'/joint['name'],joint['sha256'])==raw_ref({'path':str(HERE/ROOT_JOINT50_PROOF_ROLE['name']),'sha256':ROOT_JOINT50_PROOF_ROLE['sha256']}),
        'complete inherited ROOT joint50 future public proof bytes differ')
"""
        source=one(source,"    read_raw(bundle/'mounts.json', closure['source_binding']['mounts_sha256'])",insertion+"    read_raw(bundle/'mounts.json', closure['source_binding']['mounts_sha256'])")
        source=one(source,"    extras={'SOURCE_BINDING_RESULT.json':encode(closure['source_binding']),","    closure['source_binding'].update(ROOT_joint50_future_preflight_template_sha256=QIMG_CLOSED_PREFLIGHT_SHA,\n        ROOT_joint50_future_preflight_normal_promoted_commit=None,ROOT_joint50_future_source_only_revision=True,\n        ROOT_joint50_future_public_proof=ROOT_JOINT50_PROOF_ROLE)\n    extras={'SOURCE_BINDING_RESULT.json':encode(closure['source_binding']),")
        compile(source,'NEXT later ROOT joint50 binder','exec')
        before_functions=functions(read_pin(phase['binder']));after_functions=functions(source)
        binder_changes=[name for name in before_functions if before_functions[name]!=after_functions[name]]
        assert set(before_functions)==set(after_functions)
        assert set(binder_changes)=={'selected','bind','verify_bound_bundle','final_configs'},binder_changes
        bp=leaf/f'bind_actual{label}_ROOT_joint50_future_v1.py';write(bp,source.encode());previous=pin(bp)
        helpers={};names={}
        for name,row in phase['helpers'].items():
            text=read_pin(row).decode();newname=name.replace('QIMG_compute_closed_v2','ROOT_joint50_future_v1');assert newname!=name;names[name]=newname
            text=text.replace(f"len(CLOSURE['sources'])=={priorcount}",f"len(CLOSURE['sources'])=={count}").replace(f'len(verified)=={priorcount+5}',f'len(verified)=={count+5}').replace(f'whole exact{priorcount}',f'whole exact{count}')
            if name.startswith('read_fresh_prospective'):
                anchor="    and capacity['historical_ROOT12GiB4CPU2GiB_pending_replaced_once']"
                text=one(text,anchor,extra+anchor)
                assert "['new_ROOT_write_bytes']==10*(1<<30)" in text
            assert functions(text)==functions(read_pin(row))
            compile(text,newname,'exec');hp=leaf/'launch'/newname;write(hp,text.encode());helpers[newname]=pin(hp)
        manifest=leaf/'launch/LAUNCH_ROOT_JOINT50_FUTURE_HELPER_SOURCE_PINS_V1.json'
        write(manifest,encode({name:{k:row[k] for k in ('sha256','bytes')} for name,row in helpers.items()}))
        wrapper=read_pin(phase['wrapper']).decode()
        wrapper=assign(wrapper,'LOCAL_OUTPUTS','pathlib.Path('+repr(local)+')')
        wrapper=assign(wrapper,'BINDER','BIND_ROOT/'+repr(bp.name));wrapper=assign(wrapper,'BINDER_SHA',repr(previous['sha256']))
        wrapper=assign(wrapper,'HELPER_MANIFEST_SHA',repr(sha(manifest.read_bytes())))
        wrapper=wrapper.replace('LAUNCH_QIMG_COMPUTE_CLOSED_HELPER_SOURCE_PINS_V1.json',manifest.name)
        for oldname,newname in names.items():wrapper=wrapper.replace(oldname,newname)
        restored_wrapper=wrapper.replace(manifest.name,'LAUNCH_QIMG_COMPUTE_CLOSED_HELPER_SOURCE_PINS_V1.json')
        for oldname,newname in names.items():restored_wrapper=restored_wrapper.replace(newname,oldname)
        assert functions(restored_wrapper)==functions(read_pin(phase['wrapper']))
        compile(wrapper,'NEXT later ROOT joint50 wrapper','exec')
        wp=leaf/'launch'/f'launch_actual{label}_ROOT_joint50_future_v1.py';write(wp,wrapper.encode())
        for raw in (source,wrapper):
            n=next(n for n in ast.parse(raw).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LOCAL_OUTPUTS' for t in n.targets))
            assert ast.literal_eval(n.value.args[0])==local
        record=copy.deepcopy(phase)
        record.update(binder=previous,wrapper=pin(wp),controls=controls,helpers=helpers,helper_manifest=pin(manifest),selected_source_roster=pin(roster),source_proof=pin(leaf/Path(rootproof['path']).name),
            QIMG_native_compute_proof_roles=qimgroles,ROOT_joint50_future_proof_role={role:pin(leaf/ref['name'])},
            ROOT_joint50_future_delta_once=shared['future_delta_once'],whole_ROOT_future_bytes_once=51*(1<<30),
            metadata_before_A=count,metadata_with_A=count+1,normal_promoted_commit=None,independent_review=None)
        write(leaf/'ROOT_JOINT50_FUTURE_PHASE_SOURCE_CONTRACT_V1.json',encode(record));phases.append(record)
        observations.append({'window':[lo,hi],'metadata_before_A':count,'metadata_with_A':count+1,'scripts':5,'mounts':phase['mount_count'],
            'old_other10roles_byte_identical':True,'six_controls_byte_identical':True,'native_math_terminal92_provider_facts_host_classify_memory_CPU_unchanged':True,
            'only_preflight_existing_run_changed_and_one_pure_join_added':True,'binder_changed_functions':binder_changes,
            'inherited_joint_proof_role_once_fullhash_extent_and_binding_guard':True,'old_serial16GiB_4CPU_ROOT10GiB_and_speech20_retained':True,
            'wrapper_operation_AST_and_helper_functions_unchanged':True,'new_binder_wrapper_localpath_equal':local})
    result=copy.deepcopy(old)
    result.update(schema='neural_forge.k3_later73_85_NEXT_ROOT_joint50_future_source_composition.v1',phases=phases,
        prior_normal_tested_commit='0bfe6e21ccc695536f4c80ba582a102c9810a096',previous_reviewed_QIMG_composition=pin(oldpath),
        shared_ROOT_joint50_preflight_composition=pin(sharedpath),shared61_ROOT_joint50_whole_contract=pin(a.shared61_contract),
        shared61_ROOT_joint50_independent_review=pin(shared_review),
        ROOT_joint50_future_delta_once=shared['future_delta_once'],whole_ROOT_future_bytes_once=51*(1<<30),
        normal_promoted_commit=None,independent_review=None,runtime_actions=False,source_only=True,deployable=False,can_arm=False,
        source_packet_portable=False,source_workspace_and_prior_component_packet_references_retained=True,
        recurring_PG_gate_or_new_manual_ACK_required=False)
    cp=HERE/'LATER_ROOT_JOINT50_FUTURE_SOURCE_COMPOSITION_V1.json';write(cp,encode(result))
    fp=HERE/'LATER_ROOT_JOINT50_FUTURE_FOCUSED_CHECK_V1.json'
    write(fp,encode({'schema':'neural_forge.k3_later_NEXT_ROOT_joint50_source_check.v1','verdict':'PASS_SOURCE_ONLY_PENDING_INDEPENDENT_REVIEW',
        'composition':pin(cp),'generator':pin(Path(__file__)),'observations':observations,
        'fixtures_candidate_imports_runtime_remote_Relay_model_actions':False}))
    print(json.dumps({'composition':pin(cp),'focused_check':pin(fp),'phases':[{k:r[k] for k in ('window','binder','wrapper','metadata_before_A')} for r in phases]}))

if __name__=='__main__':main()

