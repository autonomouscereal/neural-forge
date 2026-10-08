"""Minimal isolated whole61 source proposal; no binder/launch operation executes."""
import ast
import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
RECOVERY=HERE.parents[1]
OLD=RECOVERY/'nas61_launch_integration_v1/current_ROOT_text_peer_scope_v3'
OUT=RECOVERY/'nas61_launch_integration_v1/current_QIMG_compute_closed_v1'


def encode(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def pin(path):
    raw=path.read_bytes();return {'path':str(path.resolve()),'sha256':sha(raw),'bytes':len(raw)}
def write(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    assert path.read_bytes()==raw
def replace(text,old,new,count=1):
    assert text.count(old)==count,(old[:120],text.count(old))
    return text.replace(old,new)


def main():
    assert not OUT.exists();OUT.mkdir()
    preflight=HERE/'k3_owned_capacity_preflight61_72_QIMG_compute_closed_v1.py'
    proofs={role:{'name':name,**pin(HERE/name)} for role,name in (
        ('QIMG_native_compute_closure','QIMG_EXACT_NATIVE_COMPUTE_CLOSURE_PUBLIC_PROOF_V1.json'),
        ('QIMG_native_terminal_raw_part_1','QIMG_EXACT_NATIVE_TERMINAL_RAW_PART_1_V1.json'),
        ('QIMG_native_terminal_raw_part_2','QIMG_EXACT_NATIVE_TERMINAL_RAW_PART_2_V1.json'))}
    roster=json.loads((OLD/'SELECTED61_ROOT_TEXT_SOURCE_PINS_V3.json').read_bytes())
    roster['preflight']=pin(preflight)
    roster_path=OUT/'SELECTED61_QIMG_COMPUTE_CLOSED_SOURCE_PINS_V1.json';write(roster_path,encode(roster))
    binder=(OLD/'bind_actual61_ROOT_text_v3.py').read_text()
    binder=replace(binder,'launch_integration_ROOT_text_v3','launch_integration_QIMG_compute_closed_v1')
    binder=replace(binder,'RECOVERY=HERE.parent\n',
        'RECOVERY=HERE.parent\nQIMG_CLOSED=Path(__file__).resolve().parent\nQIMG_CLOSED_PREFLIGHT_SHA='+repr(roster['preflight']['sha256'])+'\nQIMG_CLOSURE_PROOF_ROLES='+repr(proofs)+'\n')
    binder=replace(binder,"str(ROOT_SERIAL/'SELECTED61_ROOT_TEXT_SOURCE_PINS_V3.json')","str(QIMG_CLOSED/'"+roster_path.name+"')")
    binder=replace(binder,'54e7ad6d88365aacf2dcbb3511677671fa2fd247b273216b6ac201e2066cfcde',pin(roster_path)['sha256'])
    binder=replace(binder,"require(candidate['preflight']['sha256']==ROOT_SERIAL_PREFLIGHT_SHA and len(raw_ref({'path':candidate['preflight']['path'],'sha256':ROOT_SERIAL_PREFLIGHT_SHA}))==candidate['preflight']['bytes'], 'exact isolated ROOT serial preflight required')",
        "require(candidate['preflight']['sha256']==QIMG_CLOSED_PREFLIGHT_SHA and len(raw_ref({'path':candidate['preflight']['path'],'sha256':QIMG_CLOSED_PREFLIGHT_SHA}))==candidate['preflight']['bytes'], 'exact isolated QIMG closed-compute preflight required')")
    binder=replace(binder,"    roster['ROOT_current_serial_scope']={'name':'ROOT_CURRENT_SERIAL_H3_AUDIO_ACE_TEXT_PUBLIC_PROOF_V2.json'}\n",
        "    roster['ROOT_current_serial_scope']={'name':'ROOT_CURRENT_SERIAL_H3_AUDIO_ACE_TEXT_PUBLIC_PROOF_V2.json'}\n"
        "    for role,ref in QIMG_CLOSURE_PROOF_ROLES.items():\n"
        "        roles_raw[role]=raw_ref({'path':ref['path'],'sha256':ref['sha256']})\n"
        "        require(len(roles_raw[role])==ref['bytes'],'exact public QIMG proof extent required')\n"
        "        roster[role]={'name':ref['name']}\n")
    binder=replace(binder,"'completed49_phase_source','ROOT_current_serial_scope'):","'completed49_phase_source','ROOT_current_serial_scope',*QIMG_CLOSURE_PROOF_ROLES):")
    binder=replace(binder,"set(prior['source_closure']['sources'])|{'closed60_continuity','own61_72_writer_extent','completed49_phase_source','ROOT_current_serial_scope'}",
        "set(prior['source_closure']['sources'])|{'closed60_continuity','own61_72_writer_extent','completed49_phase_source','ROOT_current_serial_scope'}|set(QIMG_CLOSURE_PROOF_ROLES)")
    binder=binder.replace('len(roster)==67','len(roster)==70').replace("len(closure['sources']) == 67","len(closure['sources']) == 70")
    binder=binder.replace('complete source-bound67 metadata','complete source-bound70 metadata').replace('whole exact67 metadata','whole exact70 metadata')
    binder=replace(binder,"    # The actual three-origin observer, plan, configs, A/CID/physicalcaps and\n",
        "    closure['source_binding'].update(QIMG_closed_compute_preflight_template_sha256=QIMG_CLOSED_PREFLIGHT_SHA,\n"
        "        QIMG_closed_compute_preflight_normal_promoted_commit=None,QIMG_closed_compute_source_only_revision=True,\n"
        "        QIMG_closed_compute_public_proof_roles=QIMG_CLOSURE_PROOF_ROLES)\n"
        "    # The actual three-origin observer, plan, configs, A/CID/physicalcaps and\n")
    binder=replace(binder,"    read_raw(bundle/'mounts.json', closure['source_binding']['mounts_sha256'])\n",
        "    for role,ref in QIMG_CLOSURE_PROOF_ROLES.items():\n"
        "        actual=closure['sources'].get(role)\n"
        "        require(isinstance(actual,dict) and all(actual[key]==ref[key] for key in ('name','sha256','bytes')),\n"
        "            'complete exact QIMG native closure proof roles required')\n"
        "    require(closure['source_binding']['QIMG_closed_compute_preflight_template_sha256']==QIMG_CLOSED_PREFLIGHT_SHA\n"
        "        and closure['source_binding']['QIMG_closed_compute_preflight_normal_promoted_commit'] is None,\n"
        "        'honest unpromoted QIMG compute source proposal provenance required')\n"
        "    read_raw(bundle/'mounts.json', closure['source_binding']['mounts_sha256'])\n")
    binder_path=OUT/'bind_actual61_QIMG_compute_closed_v1.py';write(binder_path,binder.encode())
    old_manifest=json.loads((OLD/'launch/LAUNCH_HELPER_SOURCE_PINS_V1.json').read_bytes());manifest={}
    for name in old_manifest:
        body=(OLD/'launch'/name).read_bytes()
        text=body.decode()
        if name in ('execute_prospective61_remote.py','read_fresh_prospective61_remote.py'):
            assert "len(CLOSURE['sources'])==67" in text
            text=text.replace("len(CLOSURE['sources'])==67","len(CLOSURE['sources'])==70").replace('whole exact67/five','whole exact70/five')
            if name=='execute_prospective61_remote.py':text=replace(text,'len(verified)==72','len(verified)==75')
            else:
                text=replace(text,"    and capacity['ROOT_serial_source_scope_once']['source_pin']==request['sources']['ROOT_current_serial_scope']\n",
                    "    and capacity['ROOT_serial_source_scope_once']['source_pin']==request['sources']['ROOT_current_serial_scope']\n"
                    "    and capacity['QIMG_pending_scope_once']['RAM_bytes']==0 and capacity['QIMG_pending_scope_once']['cpu_threads']==0\n"
                    "    and capacity['QIMG_native_compute_closure_proof']['fresh_holder_absence_checked_by_classify'] is True\n"
                    "    and capacity['QIMG_native_compute_closure_proof']['normal_release_sha256']=='6ddcd4815040a345b12e979df272633e09593ad9561dec5a90df95c4036e0b63'\n"
                    "    and capacity['disk_components']['QIMG_post_fold_remaining_Lexar_bytes']==25769803776\n"
                    "    and capacity['root_disk_components']['QIMG_once_bytes']==2147483648\n")
            body=text.encode()
        path=OUT/'launch'/name;write(path,body);manifest[name]={'sha256':sha(body),'bytes':len(body)}
    manifest_path=OUT/'launch/LAUNCH_HELPER_SOURCE_PINS_V1.json';write(manifest_path,encode(manifest))
    wrapper=(OLD/'launch/launch_actual61_ROOT_text_v3.py').read_text()
    wrapper=replace(wrapper,'current_ROOT_text_peer_scope_v3/bind_actual61_ROOT_text_v3.py','current_QIMG_compute_closed_v1/'+binder_path.name)
    wrapper=replace(wrapper,'7aad82d551c8624179a0e87bcb4692edf6c575b5133a1dc304e46e9f40a0a821',pin(binder_path)['sha256'])
    wrapper=replace(wrapper,'launch_integration_ROOT_text_v3','launch_integration_QIMG_compute_closed_v1')
    wrapper=replace(wrapper,'e37314e3cb2844bd07dada39850e7d98b21b4fdbc2bb8b1512c3aac4192a2891',pin(manifest_path)['sha256'])
    wrapper_path=OUT/'launch/launch_actual61_QIMG_compute_closed_v1.py';write(wrapper_path,wrapper.encode())
    contract=json.loads((OLD/'ROOT_TEXT_WHOLE_CALLER_SOURCE_CONTRACT_V4.json').read_bytes())
    contract.update(schema='neural_forge.k3_NEXT61_QIMG_closed_compute_whole_caller_source_proposal.v1',
        preflight=pin(preflight),binder=pin(binder_path),wrapper=pin(wrapper_path),selected_source_roster=pin(roster_path),
        helper_manifest=pin(manifest_path),helpers=manifest,QIMG_native_compute_proof_roles=proofs,
        baseline_ROOT_text_whole_caller_contract=pin(OLD/'ROOT_TEXT_WHOLE_CALLER_SOURCE_CONTRACT_V4.json'),
        metadata_count_before_allocation=70,metadata_count_after_allocation=71,script_count=5,
        QIMG_pending_compute_once={'RAM_bytes':0,'CPU_threads':0,'requires_complete_public_proof_and_fresh_no_successor_holder_frame':True},
        QIMG_retained_Lexar_future_bytes_once=25769803776,QIMG_ROOT_future_bytes_once=2147483648,
        recurring_current_PG_witness_or_new_manual_ACK_required=False,
        optional_initial_current_normal_row_readonly_witness_source=pin(HERE/'capture_current_QIMG_released_row_v1.py'),
        original_GPU0_provider_running_still_unknown_refusal=True,
        local_exclusive_output_root='D:/codex-owned-source-receipts/k3-window61-72-source-only-9d98292e804f/launch_integration_QIMG_compute_closed_v1',
        current49_changed=False,normal_promoted_commit=None,actual_next_admission=None)
    contract_path=OUT/'NEXT61_QIMG_COMPUTE_CLOSED_WHOLE_CALLER_SOURCE_CONTRACT_V1.json';write(contract_path,encode(contract))
    for path in (preflight,binder_path,wrapper_path,*((OUT/'launch').glob('*.py'))):compile(path.read_bytes(),str(path),'exec')
    print(json.dumps({key:pin(path) for key,path in [('preflight',preflight),('binder',binder_path),('wrapper',wrapper_path),('contract',contract_path),('roster',roster_path),('helper_manifest',manifest_path)]}))


if __name__=='__main__':main()
