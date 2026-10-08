"""Local source-only NEXT derivatives; frozen current49 and parent sources untouched."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
RECOVERY=HERE.parents[1]


def pin(path):
    raw=path.read_bytes()
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def replace(text,old,new):
    assert text.count(old)==1,old[:100]
    return text.replace(old,new)


def main():
    consumer=HERE/'qimg_native_compute_closure_consumer_v1.py'
    function=consumer.read_text().split('def qimg_native_compute_closure(',1)[1]
    function='def qimg_native_compute_closure('+function
    phases=[('61-72',RECOVERY/'nas61_launch_integration_v1/current_ROOT_text_peer_scope_v3/k3_owned_capacity_preflight61_72_ROOT_text_v2.py')]
    composition=json.loads((RECOVERY/'later73_85_ROOT_text_scope_v1/LATER_ROOT_TEXT_SOURCE_COMPOSITION_V2.json').read_bytes())
    for phase in composition['phases']:
        roster=json.loads(Path(phase['selected_source_roster']['path']).read_bytes())
        phases.append((phase['window'],Path(roster['preflight']['path'])))
    outputs=[]
    for window,parent in phases:
        before=parent.read_bytes();text=before.decode()
        text=replace(text,'def classify(',function+'\n\n\ndef classify(')
        text=replace(text,"    request['QIMG_current_holder_active']=qimg_active\n",
            "    request['QIMG_current_holder_active']=qimg_active\n"
            "    qimg_closed=qimg_native_compute_closure(request['loaded_sources'],request)\n"
            "    require(qimg is None and not qimg_active and qimg_closed['container_id'] not in by_id\n"
            "        and not any(row['identity_labels'].get('ai.local.owner')==QIMG_OWNER for row in facts['containers']),\n"
            "        'closed QIMG native compute requires fresh complete absent owner/native/no-successor frame')\n"
            "    qimg_closed['fresh_holder_absence_checked_by_classify']=True\n"
            "    request['QIMG_native_compute_terminal_closed']=True\n"
            "    request['QIMG_native_compute_closure_proof']=qimg_closed\n")
        text=replace(text,'    sources={};proofs={};total=0\n','    sources={};proofs={};total=0;other_total=0\n')
        text=replace(text,"        sources[role]=body;proofs[(row['name'],sha)]=body\n",
            "        sources[role]=body;proofs[(row['name'],sha)]=body\n"
            "        if role not in ('QIMG_native_compute_closure','QIMG_native_terminal_raw_part_1','QIMG_native_terminal_raw_part_2'):\n"
            "            other_total+=(control/row['name']).stat().st_size\n")
        text=replace(text,"    request['loaded_sources']=sources\n",
            "    request['loaded_sources']=sources\n    request['source_metadata_bytes_except_QIMG_closure']=other_total\n")
        text=replace(text,"    qimg_pending_ram=qscope['root_snapshot_conservative_max_once_bytes']-(48*GIB if qimg_active else 0)\n",
            "    qimg_pending_ram=0 if request['QIMG_native_compute_terminal_closed'] else qscope['root_snapshot_conservative_max_once_bytes']-(48*GIB if qimg_active else 0)\n")
        text=replace(text,'    qimg_pending_cpu=0 if qimg_active else 8\n',
            "    qimg_pending_cpu=0 if request['QIMG_native_compute_terminal_closed'] or qimg_active else 8\n")
        text=replace(text,"        'QIMG_future_successor_delta_RAM_bytes':qscope['conservative_unresolved_extra_bytes'],'QIMG_current_plus_successor_CPU_once':8,\n",
            "        'QIMG_future_successor_delta_RAM_bytes':0 if request['QIMG_native_compute_terminal_closed'] else qscope['conservative_unresolved_extra_bytes'],\n"
            "        'QIMG_current_plus_successor_CPU_once':qimg_pending_cpu,\n"
            "        'QIMG_historical_source_unresolved_extra_RAM_bytes':qscope['conservative_unresolved_extra_bytes'],\n"
            "        'QIMG_native_compute_closure_proof':request['QIMG_native_compute_closure_proof'],\n")
        text=replace(text,"'full_source_future_scope_discharged':False,'normal_release_claimed':True,'closed_phase':'fold','CPU1933_and_native_acceptance_claimed':False},",
            "'full_source_future_scope_discharged':False,'normal_release_claimed':True,'closed_phase':'native',\n"
            "            'native_compute_phase_discharged':request['QIMG_native_compute_terminal_closed'],\n"
            "            'retained_disk_credit_bytes':0,'foreign_provider_scope_discharged':False,'CPU1933_and_native_acceptance_claimed':False},")
        after=text.encode();compile(after,parent.name,'exec')
        old={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(before).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        new={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(after).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        changed=[name for name in old if old[name]!=new[name]]
        added=sorted(set(new)-set(old))
        assert changed==['classify','run'] and added==['qimg_native_compute_closure']
        path=HERE/('k3_owned_capacity_preflight'+window.replace('-','_')+'_QIMG_compute_closed_v1.py')
        with path.open('xb') as handle:handle.write(after)
        outputs.append({'window':window,'parent':pin(parent),'candidate':pin(path),'changed_existing_functions':changed,
                        'added_functions':added,'facts_host_math_observer_controls_changed':False})
    out={'schema':'neural_forge.k3_NEXT_QIMG_compute_closed_preflight_source_composition.v1',
         'phases':outputs,'consumer':pin(consumer),'proof_roles':{},'current49_changed':False,
         'actual_next_raw9_UUID_cap_filesystem_allocation_admission':None,
         'normal_promoted_commit':None,'runtime_or_database_or_Relay_actions':False}
    for role,name in [('QIMG_native_compute_closure','QIMG_EXACT_NATIVE_COMPUTE_CLOSURE_PUBLIC_PROOF_V1.json'),
                      ('QIMG_native_terminal_raw_part_1','QIMG_EXACT_NATIVE_TERMINAL_RAW_PART_1_V1.json'),
                      ('QIMG_native_terminal_raw_part_2','QIMG_EXACT_NATIVE_TERMINAL_RAW_PART_2_V1.json')]:
        out['proof_roles'][role]=pin(HERE/name)
    path=HERE/'NEXT_QIMG_COMPUTE_CLOSED_PREFLIGHT_SOURCE_COMPOSITION_V1.json'
    with path.open('x',encoding='utf-8') as handle:handle.write(json.dumps(out,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'path':str(path),'sha256':pin(path)['sha256'],'phase_source_pins':outputs}))


if __name__=='__main__':main()
