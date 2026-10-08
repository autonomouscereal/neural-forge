"""Pure exact public-metadata join, embedded into isolated NEXT preflights.

The current holder census is checked by classify(), outside this pure join.
No PostgreSQL query, provider action, model payload read or mutable feed.
"""

def qimg_native_compute_closure(sources, request):
    import base64
    import zlib
    proof = sources['QIMG_native_compute_closure']
    require(request['sources']['QIMG_native_compute_closure']['sha256'] ==
            '0f287f18050665afd92ee5d35cbf5f3cee516efff185af2e9d643f0721d54b20',
            'exact public QIMG native closure index required')
    require(proof['schema'] == 'neural_forge.k3_QIMG_exact_native_compute_closure_public_proof.v1'
            and proof['raw_terminal_member_count'] == 62
            and proof['raw_terminal_bytes'] == 37925115
            and proof['all_decoded_metadata_and_source_bytes'] == 38111471
            and proof['per_raw_member_limit_bytes'] == 16 << 20
            and proof['aggregate_decoded_limit_bytes'] == 64 << 20
            and proof['retained_disk_credit_bytes'] == 0
            and proof['foreign_provider_scope_discharged'] is False,
            'bounded full QIMG metadata proof geometry differs')
    count = 0

    def raw(row):
        nonlocal count
        size = integer(row['bytes'], 'QIMG public raw metadata bytes')
        require(size <= 16 << 20 and row['encoding'] == 'zlib-base64',
                'unchanged per-raw QIMG metadata bound required')
        packed = base64.b64decode(row['raw_zlib_base64'], validate=True)
        decoder = zlib.decompressobj()
        value = decoder.decompress(packed, size + 1)
        require(len(value) == size and decoder.eof and not decoder.unused_data
                and not decoder.unconsumed_tail
                and hashlib.sha256(value).hexdigest() == row['sha256'],
                'complete raw QIMG metadata fullSHA/bytes required')
        count += size
        require(count <= 64 << 20, 'unchanged decoded QIMG metadata aggregate bound required')
        return value

    def doc(row):
        return json.loads(raw(row))

    outer = proof['outer']
    require(set(outer) == {'terminal', 'release', 'owner_closure', 'owner_closure_source'},
            'complete exact QIMG outer/source proof roles required')
    expected = {
        'terminal': ('5289b9b8869f8bdf86316b5721293e773c9303d907e8895b1c22df0a442650a6', 62825),
        'release': ('6ddcd4815040a345b12e979df272633e09593ad9561dec5a90df95c4036e0b63', 11302),
        'owner_closure': ('54c19ac7a14e0849578cf85afe4dc7798c6cf72b9fc3eb1d6169a67e1897af63', 3594),
        'owner_closure_source': ('f2d040fb042765bac1636ab1dcb685e26f45c884b9b521263e3b712e1b8f2a3c', 5744),
    }
    for role, (sha256, size) in expected.items():
        require((outer[role]['sha256'], outer[role]['bytes']) == (sha256, size),
                'exact QIMG authentic outer/source pin differs')
    terminal = doc(outer['terminal'])
    release = doc(outer['release'])
    owner = doc(outer['owner_closure'])
    raw(outer['owner_closure_source'])  # source bytes only; never execute owner publisher
    cid = '7fb0e676b730d2dded9294d5318a18ba347e67c7d096a759a65dbc5946fd610f'
    allocation = 'f8a611e8-6c1b-4ff2-8a4a-5149bf9bb077'
    job = 'f33f18c8-fca9-4d5f-b53f-ad7c58bb7a76'
    run = 'a4ab5f77-1743-4d3a-b4d4-41ada7102de7'
    for body in (terminal, release):
        require(body['owner'] == QIMG_OWNER and body['container_id'] == cid
                and body['allocation_id'] == allocation and body['job_uuid'] == job
                and body['run_uuid'] == run and body['phase'] == 'native',
                'same actual QIMG owner/native run/job/CID/allocation required')
    require(terminal['schema'] == 'chimera.qimg_first32.actual_fold_native_terminal_local_readback.v1'
            and terminal['actual_completed'] is True
            and terminal['actual_terminal_state_verified_prepost'] is True
            and terminal['actual_native60_full32_load_forward_readback_sourcepost_proven'] is True
            and terminal['all_raw_diagnostic_JSON_archive_members_full_byte_verified'] is True,
            'complete actual QIMG native terminal source/readback proof required')
    require(release['schema'] == 'chimera.qimg_first32.actual_exact_own_preserving_release.v1'
            and release['actual_original_release_group_verified'] is True
            and release['actual_original_PG_transaction_committed'] is True
            and release['actual_physical_exited_object_preserved'] is True
            and release['proof_pins']['gpu_terminal'] ==
                {key: outer['terminal'][key] for key in ('path', 'sha256', 'bytes')}
            and all(release[key] is False for key in ('global_reconcile_called',
                'stop_remove_or_artifact_cleanup_called', 'foreign_allocation_release_called')),
            'authentic exact-own normal committed preserving release required')
    row = release['fresh_committed_released_row']
    require(row['allocation_id'] == row['group_id'] == allocation and row['container_id'] == cid
            and row['owner'] == QIMG_OWNER and row['server'] == 'ai' and row['state'] == 'released'
            and row['details']['image'] == owner['identity']['image'],
            'authentic complete committed normal released row identity required')
    require(owner['schema'] == 'chimera.qimg_first32.same_native_terminal_closure.v37'
            and owner['identity']['run_uuid'] == run
            and owner['source_sha256'] == outer['owner_closure_source']['sha256']
            and owner['native_terminal_retention_pin'] ==
                {key: outer['terminal'][key] for key in ('path', 'sha256', 'bytes')}
            and owner['native_exact_committed_release_pin'] ==
                {key: outer['release'][key] for key in ('path', 'sha256', 'bytes')}
            and owner['original_native_resource_phase_costs_zero'] is True
            and owner['occupied_artifact_disk_still_charged'] is True
            and owner['foreign_restoration_holds_cleared'] is False
            and all(owner[key] is False for key in ('engineering_goal_complete',
                'original128_completion', 'behavioral_or_image_quality_proof')),
            'sole-owner exact native compute phase closure required')
    refs = terminal['full_readbacks']
    require(len(refs) == 62 and len({ref['role'] for ref in refs}) == 62,
            'complete unique QIMG terminal62 raw roster required')
    expected_parts = [
        ('QIMG_native_terminal_raw_part_1', '8e2ada9351e913b593c1b0d851e40725aabdbf11ce1e14b6e9cd3355eec68e28', 9338811),
        ('QIMG_native_terminal_raw_part_2', 'db9fd521dc10e5524f7cd8260d9e079a65647b9cb64ad33426cb1def51fae4ad', 9393937),
    ]
    members = []
    require(len(proof['terminal_raw_parts']) == 2, 'exact two bounded QIMG raw parts required')
    for number, (role, digest, size) in enumerate(expected_parts, 1):
        ref = request['sources'][role]
        part = sources[role]
        declared = proof['terminal_raw_parts'][number - 1]
        require(ref['sha256'] == digest and ref['name'] == declared['name']
                and declared['sha256'] == digest and declared['bytes'] == size
                and part['schema'] == 'neural_forge.k3_QIMG_exact_native_terminal_raw_part.v1'
                and part['part'] == number,
                'exact staged QIMG raw-part source pin required')
        members.extend(part['members'])
    require(len(members) == 62 and [item['role'] for item in members] == [ref['role'] for ref in refs],
            'all QIMG raw terminal members ordered and unique required')
    docs = {}
    for member, ref in zip(members, refs):
        require(all(member[key] == ref[key] for key in ('path', 'sha256', 'bytes'))
                and ref['full_byte_readback'] is True,
                'actual public QIMG raw member pin identity differs')
        value = raw(member)
        if member['role'] in ('report', 'current_container_inspect_before.json', 'current_container_inspect_after.json'):
            docs[member['role']] = json.loads(value)
    report = docs['report']
    require(report['schema'] == 'chimera.qimg_first32.fold_native_terminal_metadata.v1'
            and report['owner'] == QIMG_OWNER and report['run_uuid'] == run
            and report['job_uuid'] == job and report['allocation_id'] == allocation
            and report['container_id'] == cid and report['status'] == 'complete'
            and report['exit_code'] == 0 and report['OOMKilled'] is False
            and report['actual_native_forwards'] == 32 and report['native_blocks_per_forward'] == 60
            and report['native60_full32_load_forward_output_readback_and_sourcepost_metadata_verified'] is True
            and report['same_actual_native_container_runtime_postscan_verified'] is True,
            'complete authentic native60/32 terminal report required')
    physical = proof['release_physical_members']
    require(len(physical) == len(release['fresh_physical_readbacks']) == 3,
            'full before/after/committed release physical proofs required')
    states = [docs['current_container_inspect_before.json'], docs['current_container_inspect_after.json']]
    for member, ref in zip(physical, release['fresh_physical_readbacks']):
        require(member['role'] == ref['label'] and all(member[key] == ref[key] for key in ('path', 'sha256', 'bytes'))
                and ref['full_byte_readback'] is True, 'exact authentic release physical raw pin required')
        states.append(doc(member))
    for inspect in states:
        if isinstance(inspect, list):
            require(len(inspect) == 1, 'one exact QIMG terminal object required')
            inspect = inspect[0]
        state = inspect['State']
        require(inspect['Id'] == cid and state['Status'] == 'exited' and state['Running'] is False
                and state['Pid'] == 0 and state['ExitCode'] == 0 and state['OOMKilled'] is False,
                'authentic terminal normal-released QIMG physical object required')
    require(count == proof['all_decoded_metadata_and_source_bytes'],
            'complete decoded raw QIMG metadata extent differs')
    other = request['source_metadata_bytes_except_QIMG_closure']
    require(type(other) is int and other >= 0 and count + other <= 64 << 20,
            'existing other metadata plus complete decoded QIMG proof exceeds unchanged64MiB guard')
    return {'schema': 'neural_forge.k3_QIMG_exact_native_closed_compute_scope.v1',
            'owner': QIMG_OWNER, 'run_uuid': run, 'job_uuid': job,
            'allocation_id': allocation, 'container_id': cid,
            'RAM_bytes': 0, 'CPU_threads': 0, 'raw_member_count': 62,
            'raw_terminal_bytes': 37925115, 'decoded_all_proof_bytes': count,
            'normal_release_sha256': outer['release']['sha256'],
            'owner_phase_closure_sha256': outer['owner_closure']['sha256'],
            'historical_committed_row_is_current_observation': False,
            'fresh_holder_absence_checked_by_classify': False,
            'retained_disk_credit_bytes': 0, 'foreign_provider_scope_discharged': False,
            'whole128_or_behavioral_scope_discharged': False}
