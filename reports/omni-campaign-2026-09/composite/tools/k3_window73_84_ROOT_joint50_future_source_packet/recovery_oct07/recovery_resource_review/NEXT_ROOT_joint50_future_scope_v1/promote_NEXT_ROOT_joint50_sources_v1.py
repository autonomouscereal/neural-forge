"""Normal D-only promotion of reviewed NEXT ROOT joint50 future writes in three NEW subtrees."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,zipfile
R=Path(__file__).resolve().parents[2]
REPO=Path('D:/codex-owned-source-repositories/k3-window61-72-9d98292e804f/repository')
BEST=REPO.parent/'worktrees/best'
WORKER=REPO.parent/'worktrees/workers/codex/k3-future61-73-85-root-joint50-future-source-only-9d98292e804f'
BRANCH='codex/work/codex/k3-future61-73-85-root-joint50-future-source-only-9d98292e804f'
BASE_COMMIT='0bfe6e21ccc695536f4c80ba582a102c9810a096'
BASE='reports/omni-campaign-2026-09/composite/tools/'
OUT=Path('D:/codex-owned-source-receipts/k3-future61-73-85-ROOT-joint50-future-9d98292e804f/source_promotion_v1')
HELPER=Path('C:/Users/cereal/.agents/skills/coordinate-peer-agents/scripts/git.py')
PHASES={61:('61_72',166556705792,'078187a4cfa2f5f7810f0c0a8fa85f7fcc8304085684d45dfda5b895854881d0'),73:('73_84',180943168512,'21e322039fa65334a49e222997f92a76e640cf68d75e5388244b1e7719baae56'),85:('85_92',40692831840,'655e10cb06105a7933231a958e679d5e45a8b45ae1d9fb89a1e7940fd464f61b')}
def require(ok,label):
    if ok is not True:raise RuntimeError(label)
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def pin(p):
    b=p.read_bytes();return {'path':str(p.resolve()),'sha256':sha(b),'bytes':len(b)}
def write(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    require(p.read_bytes()==b,'exclusive complete source/receipt readback')
def git(path,*args):
    p=subprocess.run(['git','-C',str(path),*args],capture_output=True,timeout=120)
    require(p.returncode==0,'Git: '+p.stderr.decode(errors='replace')[-1500:]);return p.stdout
def state():
    p=Path('C:/Users/cereal/Documents/ai training/neural-forge')
    return {k:git(p,*argv).decode() for k,argv in {'HEAD':['rev-parse','HEAD'],'refs':['show-ref'],'status':['status','--porcelain=v1','--untracked-files=no'],'local_config':['config','--local','--list','--show-origin']}.items()}
def oldtrees():
    names=git(BEST,'ls-tree','--name-only',BASE_COMMIT+':'+BASE.rstrip('/')).decode().splitlines()
    return {n:git(BEST,'rev-parse',BASE_COMMIT+':'+BASE+n).decode().strip() for n in names if n.startswith('k3_window')}
def baseline(subtree):
    names=git(BEST,'ls-tree','-r','--name-only',BASE_COMMIT,'--',subtree).decode().splitlines()
    return {n[len(subtree)+1:]:git(BEST,'show',BASE_COMMIT+':'+n) for n in names}

H=R/'recovery_resource_review/NEXT_ROOT_joint50_future_scope_v1'
QH=R/'recovery_resource_review/NEXT_QIMG_compute_closure_v1'
C61=R/'nas61_launch_integration_v1/current_ROOT_joint50_future_v1'
L=R/'later73_85_ROOT_joint50_future_v1'
REVIEWS={'NEXT_QIMG_compute_closure_independent_review_v1/PUBLIC_ROOT_JOINT50_E886_FUTURE_SCOPE_INDEPENDENT_REVIEW_V1.json':'13bb947af8d292f831f89be232448071c8cb3782ee0cc4dfee9b072b863a27b5','NEXT_QIMG_compute_closure_independent_review_v1/NEXT61_ROOT_JOINT50_FUTURE_WHOLE_CALLER_INDEPENDENT_REVIEW_V1.json':'8de19949a95e5af77fa8a21027714dd7e4b18f6772425643dbcca0fd60bd05eb','NEXT_QIMG_compute_closure_independent_review_v1/NEXT73_85_ROOT_JOINT50_FUTURE_WHOLE_CALLER_INDEPENDENT_REVIEW_V1.json':'b1d4a81234a97fc358d0ab412932c5a41e59e6451d2630f8f3a54e8953d10eaf'}
def source_tree(root):
    return {'recovery_oct07/'+p.relative_to(R).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file() and p.suffix!='.pyc'}
def stage():
    require(git(WORKER,'branch','--show-current').decode().strip()==BRANCH and not git(WORKER,'status','--porcelain').strip(),'exact clean NEW worker')
    require(git(BEST,'rev-parse','HEAD').decode().strip()==BASE_COMMIT and not git(BEST,'status','--porcelain').strip(),'unchanged clean current best')
    require(not OUT.exists(),'exclusive new output');OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'ORIGINAL_C_BEFORE_V1.json',enc(state()));old=oldtrees();write(OUT/'PREVIOUS_OWNED_SOURCE_TREES_V1.json',enc(old))
    additions=source_tree(C61)
    # Preserve all actual source proof and earlier encoding/composition refusals.
    additions.update(source_tree(H))
    for name,expected in REVIEWS.items():
        p=R/name;b=p.read_bytes();v=json.loads(b);require(sha(b)==expected and 'PASS' in v['verdict'],'exact independent source review')
        additions['recovery_oct07/'+name]=b
    v=json.loads((L/'LATER_ROOT_JOINT50_FUTURE_SOURCE_COMPOSITION_V1.json').read_bytes());selected={x['window'][0]:x for x in v['phases']}
    prior49=json.loads((R/'recovery_resource_review/actual49_60_binding_proposal_v1/source_bundle_launch_v1/SOURCE_CLOSURE49_60.json').read_bytes())
    non_qimg=sum(x['bytes'] for x in prior49['sources'].values())+191797+533199
    proofnames={'QIMG_native_compute_closure':'QIMG_EXACT_NATIVE_COMPUTE_CLOSURE_PUBLIC_PROOF_V1.json','QIMG_native_terminal_raw_part_1':'QIMG_EXACT_NATIVE_TERMINAL_RAW_PART_1_V1.json','QIMG_native_terminal_raw_part_2':'QIMG_EXACT_NATIVE_TERMINAL_RAW_PART_2_V1.json'}
    staged={}
    for start,(window,writer,native) in PHASES.items():
        subtree=BASE+'k3_window'+window+'_ROOT_joint50_future_source_packet';require(not (WORKER/subtree).exists(),'NEW future subtree only')
        members=baseline(BASE+'k3_window'+window+'_QIMG_compute_closed_source_packet');members.update(additions)
        if start!=61:
            for p in L.rglob('*'):
                if p.is_file() and p.suffix!='.pyc' and (start==85 or 'phase85_92_ROOT_joint50_future_v1' not in p.parts):members['recovery_oct07/'+p.relative_to(R).as_posix()]=p.read_bytes()
        if start==61:
            source=H/'k3_owned_capacity_preflight61_72_ROOT_joint50_future_v1.py';parent=QH/'k3_owned_capacity_preflight61_72_QIMG_compute_closed_v1.py'
            binder=C61/'bind_actual61_ROOT_joint50_future_v1.py';wrapper=C61/'launch/launch_actual61_ROOT_joint50_future_v1.py';counts=[71,72]
        else:
            phase=selected[start];source=H/('k3_owned_capacity_preflight'+window+'_ROOT_joint50_future_v1.py')
            parent=QH/('k3_owned_capacity_preflight'+window+'_QIMG_compute_closed_v1.py')
            binder=Path(phase['binder']['path']);wrapper=Path(phase['wrapper']['path']);counts=[phase['metadata_before_A'],phase['metadata_with_A']]
        require(any(sha(b)==pin(parent)['sha256'] for b in members.values()),'exact parent preflight retained')
        require(any(sha(b)==native for b in members.values()),'exact native retained')
        members['.gitattributes']=b'* -text\n'
        if start==61:
            members['check_all_three_ROOT_joint50_source_packets_v1.py']=b'"""Run the fixed three existing source-only checks."""\nfrom pathlib import Path\nimport subprocess,sys\nroot=Path(__file__).resolve().parent.parent\nfor window in ("61_72","73_84","85_92"):\n packet=root/("k3_window"+window+"_ROOT_joint50_future_source_packet")\n p=subprocess.run([sys.executable,"-B",str(packet/"check_NEXT_ROOT_joint50_promoted_packet_v1.py"),"--candidate-root",str(packet)],check=False,timeout=90)\n if p.returncode:raise SystemExit(p.returncode)\n'
        roster={'schema':'neural_forge.k3_NEXT_ROOT_joint50_future_reviewed_source_roster.v1','source_only':True,'deployable':False,'phase_start':start,
            'actual_prior_terminal':None,'next_UUID':None,'filesystem':None,'allocation':None,'capacity_admission':None,
            'selected_preflight_sha256':pin(source)['sha256'],'parent_preflight_sha256':pin(parent)['sha256'],'native_sha256':native,
            'standalone_consumer_sha256':pin(H/'ROOT_joint50_future_consumer_v1.py')['sha256'],
            'entrypoints':{'launch':'recovery_oct07/'+wrapper.relative_to(R).as_posix(),'binder':'recovery_oct07/'+binder.relative_to(R).as_posix()},
            'metadata_before_after_A':counts,'script_count':5,'ROOT_serial_RAM_CPU_write_once':[17179869184,4,10737418240],
            'conservative_whole_ROOT_future_bytes_once':51*(1<<30),'speech_ROOT20GiB_preserved':True,'inclusive_ROOT49GiB_not_added':True,
            'ROOT_joint50_future_delta_once':{'RAM_bytes':0,'CPU_threads':0,'ROOT_bytes':2*(1<<30),'Lexar_bytes':28*(1<<30)},
            'ROOT_joint50_proof_role':dict(name='ROOT_CURRENT_JOINT50_SERIAL_FUTURE_PUBLIC_PROOF_V1.json',sha256=pin(H/'ROOT_CURRENT_JOINT50_SERIAL_FUTURE_PUBLIC_PROOF_V1.json')['sha256'],bytes=533199),
            'QIMG_compute_once':[0,0],'QIMG_retained_Lexar_ROOT_write_once':[24*(1<<30),2*(1<<30)],
            'QIMG_proof_roles':{role:dict(name=name,sha256=pin(QH/name)['sha256'],bytes=pin(QH/name)['bytes']) for role,name in proofnames.items()},
            'existing_non_QIMG_source_metadata_bytes':non_qimg,'decoded_QIMG_proof_bytes':38111471,'original_terminal_raw_count':62,
            'current_holder_absence_requires_fresh_complete_census':True,'historical_committed_release_row_is_current_observation':False,
            'recurring_PG_witness_or_new_manual_ACK_required':False,'unknown_running_GPU0_provider_guard_preserved':True,
            'P40_question_is_not_adopted_future':True,'compatibility4GiB2CPU_STATUS_is_not_source_adoption':True,
            'writer_bytes_once':writer,'NAS_if_selected_common_once':692026605568+writer,'Lexar_if_selected_own_QIMG_ROOT_head_once':writer+68*(1<<30),
            'current_actor_adoption':False,'current_runtime_and_all_previous_source_trees_unchanged':True,
            'original_workspace_and_D_component_references_retained':'Exact original C recovery and D APIs/pins remain required. This committed source archive is not a portable executable materialization or a launch; actual prior terminal and fresh source-bound admission remain unresolved.',
            'historical_inner_promotion_nulls':'Reviewed dated source declarations stay byte-exact; only the new outer manifest records this normal promotion.',
            'independent_source_reviews':['recovery_oct07/'+n for n in REVIEWS],
            'members':{n:{'sha256':sha(b),'bytes':len(b)} for n,b in sorted(members.items())}}
        raw=enc(roster);members['NEXT_ROOT_JOINT50_FUTURE_SOURCE_ROSTER_V1.json']=raw
        checker=(H/'check_NEXT_ROOT_joint50_promoted_packet_v1.py').read_text().replace('__ROSTER_SHA256__',sha(raw));compile(checker,'owned_QIMG_checker','exec');members['check_NEXT_ROOT_joint50_promoted_packet_v1.py']=checker.encode()
        for n,b in members.items():write(WORKER/subtree/n,b)
        staged[str(start)]={'subtree':subtree,'members':{n:{'sha256':sha(b),'bytes':len(b)} for n,b in members.items()},'entrypoints':roster['entrypoints']}
    write(OUT/'STAGED_REVIEWED_NEXT_ROOT_JOINT50_SOURCE_V1.json',enc({'worker':str(WORKER),'branch':BRANCH,'base_commit':BASE_COMMIT,'phases':staged,'previous_tree_oids':old,'source_only':True}))
    print(json.dumps({k:len(v['members']) for k,v in staged.items()}))

def commit():
    v=json.loads((OUT/'STAGED_REVIEWED_NEXT_ROOT_JOINT50_SOURCE_V1.json').read_bytes())
    expected=set()
    for phase in v['phases'].values():
        for n,row in phase['members'].items():require(pin(WORKER/phase['subtree']/n)['sha256']==row['sha256'],'exact staged member');expected.add(phase['subtree']+'/'+n)
        git(WORKER,'add','-f','--',phase['subtree'])
    require(set(git(WORKER,'diff','--cached','--name-only').decode().splitlines())==expected,'only three exact NEW future source subtrees')
    git(WORKER,'commit','-m','Prepare NEXT exact ROOT joint50 future writes for K3 61 73 85 source callers')
    commit_id=git(WORKER,'rev-parse','HEAD').decode().strip();require(not git(WORKER,'status','--porcelain').strip(),'clean private source commit')
    for phase in v['phases'].values():
        for n,row in phase['members'].items():require(sha(git(WORKER,'show',commit_id+':'+phase['subtree']+'/'+n))==row['sha256'],'private Git fullblob bytes')
    write(OUT/'PRIVATE_EXACT_FULL_BLOB_COMMIT_V1.json',enc({'commit':commit_id,'files':len(expected),'only_three_new_subtrees':True}));print(json.dumps({'commit':commit_id,'files':len(expected)}))
def promote():
    # One normal test argv checks the fixed three source-only subtrees sequentially.
    argv=[sys.executable,'-B',str(HELPER),'--repo',str(REPO),'promote','--branch',BRANCH,'--check-json',json.dumps(['C:/Python314/python.exe','-B',BASE+'k3_window61_72_ROOT_joint50_future_source_packet/check_all_three_ROOT_joint50_source_packets_v1.py'])]
    p=subprocess.run(argv,capture_output=True,text=True,timeout=240)
    write(OUT/'NORMAL_PROMOTION_INVOCATION_V1.json',enc({'argv':argv,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}));require(p.returncode==0,'normal promotion failed; complete output preserved')
    v=json.loads(p.stdout);require(v['ok'] is True,'normal tested promotion required');write(OUT/'NORMAL_TESTED_NEXT_ROOT_JOINT50_PROMOTION_V1.json',enc(v));print(json.dumps(v))
def package():
    v=json.loads((OUT/'NORMAL_TESTED_NEXT_ROOT_JOINT50_PROMOTION_V1.json').read_bytes());staged=json.loads((OUT/'STAGED_REVIEWED_NEXT_ROOT_JOINT50_SOURCE_V1.json').read_bytes());commit_id=v['best_commit'];packet=OUT/'immutable_packets_v1';packet.mkdir(exist_ok=False);results={}
    for start,phase in staged['phases'].items():
        archive=packet/('K3_WINDOW'+PHASES[int(start)][0]+'_NEXT_ROOT_JOINT50_FUTURE_UNBOUND_SOURCE_PACKET_V1.zip');rows={}
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for n,row in sorted(phase['members'].items()):
                path=phase['subtree']+'/'+n;b=git(BEST,'show',commit_id+':'+path);require(sha(b)==row['sha256'] and len(b)==row['bytes'],'actual promoted source blob')
                oid=git(BEST,'rev-parse',commit_id+':'+path).decode().strip();info=zipfile.ZipInfo(n,(2026,10,8,0,0,0));info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
                rows[n]=dict(row,Git_blob_oid=oid,committed_path=path)
        with zipfile.ZipFile(archive) as z:
            require(z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(rows),'full immutable member/CRC readback')
            for n,row in rows.items():require(sha(z.read(n))==row['sha256'] and len(z.read(n))==row['bytes'],'immutable member bytes')
        manifest={'schema':'neural_forge.k3_NEXT_ROOT_joint50_future_normal_promoted_immutable_source_packet.v1','source_only':True,'deployable':False,'phase_start':int(start),'normal_tested_promoted_commit':commit_id,'private_source_commit':v['source_commit'],'previous_best_commit':v['previous_commit'],'archive':pin(archive),'member_count':len(rows),'members':rows,'all_members_from_actual_promoted_Git_blobs':True,'entrypoints':phase['entrypoints'],'normal_helper_receipt':pin(OUT/'NORMAL_TESTED_NEXT_ROOT_JOINT50_PROMOTION_V1.json'),'actual_prior_terminal':None,'next_UUID':None,'filesystem':None,'allocation':None,'capacity_admission':None,'no_runtime_or_Relay_actions':True,'not_portable_executable_materialization':True}
        m=packet/('IMMUTABLE_WINDOW'+PHASES[int(start)][0]+'_NEXT_ROOT_JOINT50_FUTURE_SOURCE_MANIFEST_V1.json');write(m,enc(manifest));results[start]={'archive':pin(archive),'manifest':pin(m),'members':len(rows)}
    prior=json.loads((OUT/'PREVIOUS_OWNED_SOURCE_TREES_V1.json').read_bytes());after={n:git(BEST,'rev-parse',commit_id+':'+BASE+n).decode().strip() for n in prior};require(after==prior,'all previous own source subtrees preserved')
    before=json.loads((OUT/'ORIGINAL_C_BEFORE_V1.json').read_bytes());now=state();require(before==now,'original C full local Git state unchanged')
    write(OUT/'ORIGINAL_C_AND_PREVIOUS_D_TREES_AFTER_V1.json',enc({'original_C_before':before,'original_C_after':now,'all_C_equal':before==now,'all_previous_D_tree_oids_equal':after==prior,'previous_D_before':prior,'previous_D_after':after}))
    write(OUT/'IMMUTABLE_NEXT_ROOT_JOINT50_FUTURE_PACKETS_V1.json',enc({'commit':commit_id,'packets':results,'C_and_previous_D_preserved':True}));print(json.dumps({'commit':commit_id,'packets':results}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['stage','commit','promote','package']);globals()[p.parse_args().phase]()
