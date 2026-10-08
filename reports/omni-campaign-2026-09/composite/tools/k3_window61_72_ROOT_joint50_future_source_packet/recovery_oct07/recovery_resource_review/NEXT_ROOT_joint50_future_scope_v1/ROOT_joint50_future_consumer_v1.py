"""Exact source-owned pending write join; no actor acceptance or compute expansion."""
def current_ROOT_joint50_future_scope(sources,request,root_serial):
    proof=sources['ROOT_current_joint50_serial_future']
    require(request['sources']['ROOT_current_joint50_serial_future']['sha256']=='65379f9787f9eec44575c6785b33eb4d461cefc91bc1754fdddfcf2dae528708',
        'exact current ROOT joint50 source proof required')
    require(proof['schema']=='neural_forge.k3_current_ROOT_joint50_serial_future_public_proof.v1'
        and proof['member_count']==10 and proof['current_actor_adoption'] is False,
        'complete source-only ROOT joint50 proof required')
    expected={'ROOT_joint50_adoption': {'path': 'C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\all_sources_next_execution_oct07\\qvl32_native_execution_01a10f94\\ROOT_H3_JOINT50_ADDITIVE_SERIAL_FUTURE_SCOPE_ADOPTION_V1_b55fb9ea6ad249b1aee1c7f4f98d38c1.json', 'sha256': 'e886d40173bd3a02065946f9f60a49d89356d7894334e8d5af26369cdf81930f', 'bytes': 7013}, 'independent_lifecycle_source_acceptance': {'path': 'C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\all_sources_next_execution_oct07\\qvl32_native_execution_01a10f94\\H3_joint50_CPU_lifecycle_independent_review_v1\\v1_2_local_alias_contract\\INDEPENDENT_H3_JOINT50_NORMAL_CPU_LIFECYCLE_SOURCE_ACCEPTANCE_V1_2.json', 'bytes': 17816, 'sha256': '10f768299eb6b008ed2e76b947579e853781cffa634596566ec16c9bd40429e3'}, 'final_lifecycle_source_checkpoint': {'path': 'C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\all_sources_next_execution_oct07\\qvl32_native_execution_01a10f94\\H3_joint50_CPU_runtime_route_v1_2_exact_source_aliases_and_refusals\\H3_JOINT50_CPU_LIFECYCLE_SOURCE_CHECKPOINT_V1_2.json', 'bytes': 111224, 'sha256': 'cae960b08e3aff3ab0f2b6b7d22140f267dce37770912d9efb525832c263085f'}, 'final_harness_source_checkpoint': {'path': 'C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\all_sources_next_execution_oct07\\qvl32_native_execution_01a10f94\\H3_joint50_unbound_executable_harness_v2_closed_directories\\PREPARED_UNBOUND_JOINT50_EXECUTABLE_HARNESS_SOURCE_CHECKPOINT_V2.json', 'bytes': 14009, 'sha256': '99ef0a713bdb2864a994e80fc982a470389c4a3ad8efd24a3c645ba0c204b39d'}, 'final_harness_source_acceptance': {'path': 'C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\all_sources_next_execution_oct07\\qvl32_native_execution_01a10f94\\H3_joint50_executable_harness_independent_review_v1\\v2_closed_directories\\INDEPENDENT_UNBOUND_JOINT50_CLOSED_DIRECTORY_SOURCE_ACCEPTANCE_V2.json', 'bytes': 11737, 'sha256': '005ffd00370af1cdf4723cf30d8ab8cc6e961bc4c8f084a99b96739df49f39ad'}, 'prior_genuine_serial_adoption': {'path': 'C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\all_sources_next_execution_oct07\\qvl32_native_execution_01a10f94\\ROOT_H3_TEXT_SERIAL_PENDING_ACE_AUDIO_ADOPTION_V1.json', 'bytes': 28989, 'sha256': 'f776fc901232fed121f2ba62b690834659a63e0c426f8c290e9f92685e5162af'}, 'completed_speech_component_model_adoption': {'path': 'C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\all_sources_next_execution_oct07\\qvl32_native_execution_01a10f94\\ROOT_COMPLETED_SPEECH_COMPONENT_MODEL_FUTURE_ADOPTION_V1.json', 'bytes': 8214, 'sha256': '811a18d1bdfab686315b83883442540f9af4963a369adbb983f88409fd2b6d35'}, 'current_NAS_ROOT_adoption': {'path': 'C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\all_sources_next_execution_oct07\\qvl32_native_execution_01a10f94\\ROOT_CURRENT_K3_49_60_NAS_ACCOUNTING_ADOPTION_V3.json', 'bytes': 12180, 'sha256': 'b059dea3e8fa8bc51a7dfb8a77dae7b3e0ed3cd46a4885e6956a52e91780e8fc'}, 'actual_previous_own_text_normal_release': {'path': 'C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\all_sources_next_execution_oct07\\qvl32_native_execution_01a10f94\\H3_text_layer50_CPU_runtime_route_v2_completed_speech_only\\control_remote_f379f49106184abe8ba6bab56d2add6c.json', 'bytes': 128726, 'sha256': 'aff3143d0f2a4d55ae1854b88b26663774ec1e012de25b7ea49aacac7880911e'}, 'actual_previous_own_text_independent_acceptance': {'path': 'C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\all_sources_next_execution_oct07\\qvl32_native_execution_01a10f94\\H3_text_actual_saved_output_independent_review_v1\\INDEPENDENT_ACTUAL_H3_TEXT_NATIVE_SAVED_CONDITIONING_ACCEPTANCE_V1.json', 'bytes': 154960, 'sha256': 'c570c0481da47a9bdabf6d66c4c0a3e245d4682c1aba4f639fe88e17f73f9c07'}}
    require(set(proof['members'])==set(expected),'exact complete public ROOT joint50 metadata roles required')
    docs={}
    for role,ref in expected.items():
        row=proof['members'][role];raw=row['raw_utf8'].encode()
        require(all(row[k]==ref[k] for k in ('path','sha256','bytes')) and len(raw)==ref['bytes']
            and hashlib.sha256(raw).hexdigest()==ref['sha256'],'full current ROOT joint50 metadata raw pin required')
        docs[role]=json.loads(raw)
    a=docs['ROOT_joint50_adoption']
    require(a['schema']=='chimera.root.H3_joint50_additive_serial_future_scope_adoption.v1'
        and a['native']==a['rootnative']=='01a10f94-259b-7623-9679-ce4e68f82b8c'
        and a['binding']==docs['prior_genuine_serial_adoption']['binding']
        and a['root_adopted'] is True and a['execution_authority'] is False
        and a['resources_claimed'] is False and a['foreign_rows_caps_jobs_or_promises_changed'] is False
        and a['cleanup_or_disk_or_nonreclaimable_credit_bytes']==0,
        'genuine same-owner ROOT joint50 adopted future scope required')
    require(set(a['source_pins'])==set(expected)-{'ROOT_joint50_adoption'}
        and all(a['source_pins'][role]==ref for role,ref in expected.items() if role!='ROOT_joint50_adoption'),
        'exact adoption selected metadata/source pins required')
    require(a['source_checkpoint_sha256']==expected['final_lifecycle_source_checkpoint']['sha256']
        and a['harness_source_checkpoint_sha256']==expected['final_harness_source_checkpoint']['sha256']
        and a['harness_source_acceptance_sha256']==expected['final_harness_source_acceptance']['sha256']
        and a['prior_genuine_serial_adoption_sha256']==expected['prior_genuine_serial_adoption']['sha256']
        and a['completed_speech_component_model_adoption_sha256']==expected['completed_speech_component_model_adoption']['sha256']
        and a['current_NAS_ROOT_adoption_sha256']==expected['current_NAS_ROOT_adoption']['sha256'],
        'current source/lifecycle and prior scope fullSHA ancestry required')
    require(a['existing_ROOT_typed_future_bytes']==46*GIB and a['existing_ROOT_conserved_metadata_margin_bytes']==GIB
        and a['existing_ROOT_future_floor_bytes']==47*GIB and a['NEW_joint_capture_ROOT_future_bytes']==2*GIB
        and a['NEW_fixed3584_fold_Lexar_future_bytes']==28*GIB and a['total_ROOT_future_floor_bytes']==49*GIB
        and a['speech_ROOT_future_bytes']==18*GIB and a['other_source_futures_discharged'] is False
        and a['current_retained_physical_bytes_still_df_charged'] is True,
        'current new ROOT2 and Lexar28 future writes exactly once required')
    require(a['serial_planning_memory_bytes']==16*GIB and a['serial_planning_cpu_threads']==4
        and root_serial['RAM_bytes']==16*GIB and root_serial['CPU_threads']==4
        and root_serial['new_ROOT_write_bytes']==10*GIB
        and a['actual_next_serial_START_requires_previous_own_terminal_release'] is True
        and a['fresh_complete_locked_normal_admissions_required'] is True
        and a['unknown_jobs_or_same_owner_wildcards_allowed'] is False,
        'existing ROOT serial MAX envelope and active actor guards preserved')
    require(a['phase_proposals']=={'capture':{'memory_bytes':8*GIB,'cpu_threads':4,'no_swap':True,'GPU_count':0,'ROOT_output_bytes':2*GIB},
        'fold':{'memory_bytes':16*GIB,'cpu_threads':4,'no_swap':True,'GPU_count':0,'Lexar_output_bytes':28*GIB}}
        and a['joint50_or_fold_or_host_execution_complete'] is False,
        'pending native capture/fold proposals are not current actor or completion proof')
    return {'source_pin':request['sources']['ROOT_current_joint50_serial_future'],
        'owner_adoption_sha256':expected['ROOT_joint50_adoption']['sha256'],
        'RAM_serial_MAX_bytes_once':16*GIB,'CPU_serial_MAX_threads_once':4,'additional_RAM_bytes':0,'additional_CPU_threads':0,
        'additional_ROOT_write_bytes_once':2*GIB,'additional_Lexar_write_bytes_once':28*GIB,
        'owner_inclusive_ROOT_floor_bytes':49*GIB,'conservative_ROOT_speech_padding_preserved_bytes':2*GIB,
        'conservative_whole_ROOT_required_bytes':51*GIB,'current_actor_adopted':False,
        'compatibility_fullhash_phase_adopted':False,'P40_proposal_adopted':False,'prior_future_or_retained_disk_credit_bytes':0}
