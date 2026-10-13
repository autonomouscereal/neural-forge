"""2026 synthetic contract fixtures; actual original writer used for batch checks."""
import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

from experiments.chimera_recovery import student_prefix_teacher as P


def write(path, value):
    path.write_bytes(P.canonical(value)+b'\n')
    return P.pin(path)


def make_plan(tmp_path, writer=None):
    writer, engine = writer or tmp_path/'writer.py', tmp_path/'engine.py'
    if writer == tmp_path/'writer.py':
        writer.write_text('# synthetic original-writer pin fixture, not a capture\n')
    engine.write_text('# synthetic engine pin fixture, not a model\n')
    paths = {'writer': writer, 'engine': engine}
    runtime = {'image_id': 'sha256:'+'e'*64, 'python': '/fixture/python',
        'packages': {'torch':'fixture-torch','transformers':'fixture-transformers'},
        'teacher_sha256': P.sha(writer), 'venv_container': '/fixture/runtime',
        'parent_config_sha256':'c'*64}
    for key in ('compiler_image_proof','actual_fla_kernel_proof'):
        runtime[key] = write(tmp_path/(key+'.json'), {'fixture_only':True})
    docs = {
        'prompt_manifest': {'schema':'chimera_recovery.onpolicy_prompts.v1','train_only':True,
            'rows':2,'maximum_sampled_tokens':512,'actual_sampled_tokens':0},
        'shape_proof': {'complete':True,'parent_integrity': {'unchanged':True,
            'before':{'mode':'full','tree_sha256':'a'*64,'files':[
                {'path':'tokenizer.json','sha256':'b'*64},{'path':'config.json','sha256':'c'*64},
                {'path':'chat_template.jinja','sha256':'d'*64}]},'after':{'tree_sha256':'a'*64}}},
        'distribution_review': {'fixture_only':True}, 'paired_proof': {'passed':True},
        'native_parity': {'passed':True,'top_k':64,'vocab_rows':248320,
            'tool':P.pin(writer),'engine':P.pin(engine),
            'runtime':{'python':'/fixture/python','versions':runtime['packages']},
            'linear_attention_kernel':{'requested':'auto','chunk_gated_delta_rule':'fixture-kernel'}},
        'source_plan': {'runtime':runtime}}
    for key, doc in docs.items():
        paths[key] = tmp_path/(key+'.json'); write(paths[key],doc)
    docs['portable_proof'] = {'plan_sha256':P.sha(paths['source_plan']),
        'actual_writer_engine_imported':True,'python':'/fixture/python'}
    docs['execution_policy'] = {'schema':'chimera_recovery.text_teacher_execution_policy.v1',
        'selected_for_preparation':True,'preserve_original_paired_gate':True,'claim_batch_shape_invariance':False,
        **{key+'_sha256':P.sha(paths[value]) for key, value in
            (('shape_proof','shape_proof'),('distribution_review','distribution_review'),
             ('paired_parity_proof','paired_proof'),('source_plan','source_plan'))}}
    for key in ('portable_proof','execution_policy'):
        paths[key] = tmp_path/(key+'.json'); write(paths[key], docs[key])
    output = tmp_path/'teacher-plan.json'
    plan = P.prepare_plan(**paths,output=output)
    return output,plan


@pytest.fixture
def teacher_plan(tmp_path):
    return make_plan(tmp_path)


def metadata(plan):
    config = {k:plan['profile'][k] for k in
        ('dtype','head_precision','attention_implementation','top_k','vocab_rows','rows_per_shard')}
    config.update(corpus_rows=2,linear_attention_kernel=plan['runtime_profile']['linear_attention_kernel'],
        corpus_read={'ids_key':'input_ids','lengths_key':'real_lengths',
                     'mask_source':'target_mask shifted by one inside the tool (stored t iff target t+1)'})
    teacher = {'equal_before_and_after_every_session':True,
               'tree_sha256_before':'a'*64,'tree_sha256_after':'a'*64}
    manifest = {'schema':'omni-distill-teacher-cache-manifest-v1','all_finite':True,'all_structural_ok':True,
        'teacher':teacher,'tool':plan['sources']['writer'],'engine':plan['sources']['engine'],
        'cache_config':config,'forward_tokens_total':14,'positions_total':5}
    session = {'batching':{'micro_batch':1,'max_batch_tokens':0,'order':'corpus'},
        'runtime':copy.deepcopy(plan['runtime_profile']), 'tool':plan['sources']['writer'],'engine':plan['sources']['engine'],
        'teacher_unchanged_in_session':True,'teacher_digest_before':{'tree_sha256':'a'*64},
        'teacher_digest_after':{'tree_sha256':'a'*64},'host_resident':{'mode':'pageable'},
        'head_matmul':{'head_precision':'fp32','float32_matmul_precision':'highest','cuda_matmul_allow_tf32':False}}
    record = {'rows':[0,2],'row_table':[{'row':0,'length':9,'count':3},{'row':1,'length':5,'count':2}],
        'batches':2,'micro_batch':1,'max_batch_tokens':0,'batch_order':'corpus','head_chunk_rows':256,
        'forward_tokens':14,'padded_tokens':14,'session':'session-A'}
    return manifest,[record],{'session-A':session},[9,5],[3,2]


def test_prep_has_no_generated_cache_runtime_claim_and_profile_is_derived(teacher_plan):
    path,plan = teacher_plan
    assert P.validate_plan(path) == plan
    assert plan['actual_sampled_tokens'] == 0 and plan['production_runtime_receipt'] is None
    assert not plan['production_training_ready'] and not plan['offline_paired_profile_changed']
    altered = copy.deepcopy(plan); altered['profile']['micro_batch']=4
    write(path,altered)
    with pytest.raises(P.Error,match='profile differs'): P.validate_plan(path)
    altered = copy.deepcopy(plan); altered['teacher_sha256']='f'*64
    write(path,altered)
    with pytest.raises(P.Error,match='derived'): P.validate_plan(path)
    write(path,plan)
    Path(plan['sources']['writer']['path']).write_text('# changed writer same path\n')
    with pytest.raises(P.Error,match='evidence changed'): P.validate_plan(path)


def test_two_unequal_sequences_each_complete_once(teacher_plan):
    _,plan = teacher_plan
    actual = P.verify_records(plan,*metadata(plan))
    assert actual == {'rows':2,'real_tokens':14,'supervised_positions':5}


@pytest.mark.parametrize('mutation', ['microbatch','padding','crop','batch_count','session_batch',
    'runtime','source','head_precision','kernel','mask','duplicate','missing','row_mask','wrong_parent'])
def test_cache_main_identity_alone_cannot_hide_wrong_shape_or_profile(teacher_plan,mutation):
    _,plan = teacher_plan
    manifest,records,sessions,lengths,counts = copy.deepcopy(metadata(plan))
    r,s = records[0],sessions['session-A']
    if mutation == 'microbatch': r['micro_batch']=4
    elif mutation == 'padding': r['padded_tokens']=18
    elif mutation == 'crop': r['row_table'][0]['length']=8
    elif mutation == 'batch_count': r['batches']=1
    elif mutation == 'session_batch': s['batching']['micro_batch']=8
    elif mutation == 'runtime': s['runtime']['versions']['torch']='other'
    elif mutation == 'source': s['tool']['sha256']='f'*64
    elif mutation == 'head_precision': s['head_matmul']['cuda_matmul_allow_tf32']=True
    elif mutation == 'kernel': manifest['cache_config']['linear_attention_kernel']={'requested':'torch'}
    elif mutation == 'mask': manifest['cache_config']['corpus_read']['mask_source']='none'
    elif mutation == 'duplicate': records.append(copy.deepcopy(r))
    elif mutation == 'missing': r.update(rows=[0,1],row_table=r['row_table'][:1],batches=1,forward_tokens=9,padded_tokens=9)
    elif mutation == 'row_mask': r['row_table'][1]['count']=3
    elif mutation == 'wrong_parent': manifest['teacher']['tree_sha256_after']='f'*64
    with pytest.raises(P.Error): P.verify_records(plan,manifest,records,sessions,lengths,counts)


def test_original_writer_singleton_removes_only_real_padding_and_shifts_target_mask(tmp_path):
    torch = pytest.importorskip('torch')
    from safetensors.torch import save_file
    path = Path(os.environ.get('CHIMERA_QTEXT_WRITER',
        'C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/distill/tools/qtext_teacher_cache.py'))
    if not path.is_file(): pytest.skip('original qtext writer source not staged')
    spec = importlib.util.spec_from_file_location('original_singleton_batch_fixture',path)
    writer = importlib.util.module_from_spec(spec); spec.loader.exec_module(writer)
    ids = torch.tensor([[11,12,13,14,15,16,17],[21,22,23,24,0,0,0]])
    mask = torch.tensor([[False,False,False,False,True,True,True],[False,False,True,True,False,False,False]])
    corpus = tmp_path/'corpus.safetensors'
    save_file({'input_ids':ids,'real_lengths':torch.tensor([7,4]),'assistant_mask':mask,'target_mask':mask.clone()},str(corpus))
    native = writer.load_text_corpus(corpus)
    batches = list(writer.text_batches(native,0,2,1,0,'corpus'))
    assert [b['input_ids'].tolist() for b in batches] == [[[11,12,13,14,15,16,17]],[[21,22,23,24]]]
    assert [b['store_mask'].tolist() for b in batches] == [[[False,False,False,True,True,True,False]],[[False,True,True,False]]]
    assert [b['rows'] for b in batches] == [[0],[1]]
    # The original multirow mode visibly changes the padded width profile.
    paired = list(writer.text_batches(native,0,2,4,0,'corpus'))
    assert paired[0]['input_ids'].shape == (2,7)


def test_real_original_writer_raw_capture_guard_rejects_padding_runtime_and_source_ids(tmp_path):
    """Real native files and original raw verifier; teacher values are fixture-only."""
    torch = pytest.importorskip('torch')
    from safetensors.torch import save_file
    path = Path(os.environ.get('CHIMERA_QTEXT_WRITER',
        'C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/distill/tools/qtext_teacher_cache.py'))
    if not path.is_file(): pytest.skip('original qtext writer source not staged')
    spec = importlib.util.spec_from_file_location('original_capture_fixture',path)
    writer = importlib.util.module_from_spec(spec); spec.loader.exec_module(writer)
    plan_path,plan = make_plan(tmp_path,path)
    cache = tmp_path/'cache'; (cache/'sessions').mkdir(parents=True)
    ids = torch.tensor([[11,12,13,14,15,16,17,18,19],[21,22,23,24,25,0,0,0,0]])
    lengths = torch.tensor([9,5]); mask = torch.tensor([[False]*6+[True]*3,[False]*3+[True]*2+[False]*4])
    native = tmp_path/'corpus.safetensors'
    save_file({'input_ids':ids,'real_lengths':lengths,'assistant_mask':mask,'target_mask':mask.clone()},str(native))
    traces = tmp_path/'traces.jsonl'
    traces.write_bytes(b''.join(P.canonical({'id':str(i),'prompt_ids':ids[i,:n-c].tolist(),
        'generated_ids':ids[i,n-c:n].tolist()})+b'\n' for i,n,c in ((0,9,3),(1,5,2))))
    generation = tmp_path/'generation.json'; write(generation,{'synthetic_fixture_only':True})
    prompts = tmp_path/'prompts.jsonl'; prompts.write_text('synthetic prompt fixture only\n')
    corpus = {'schema':'chimera_recovery.student_prefix_corpus.v1','corpus':P.pin(native),
        'source_prompt_manifest':plan['sources']['prompt_manifest'],'generation_receipt':P.pin(generation),
        'source_prompts':P.pin(prompts),'traces':P.pin(traces),'row_ids':['0','1'],
        'identity':{'tokenizer_sha256':'b'*64,'template_sha256':'d'*64}}
    corpus_path = tmp_path/'corpus.json'; write(corpus_path,corpus)
    manifest,records,sessions,_,_ = metadata(plan)
    sessions['session-A']['runtime']['devices']=[{'uuid':'GPU-00000000-0000-0000-0000-000000000001'}]
    record = records[0]
    record['row_table'][0]['offset']=0; record['row_table'][1]['offset']=3
    arrays = {'ids':torch.tensor([16,17,18,23,24]),'pos':torch.tensor([5,6,7,2,3]),
        'topk_ids':torch.arange(64,dtype=torch.int32).repeat(5,1),
        'topk_logprobs':torch.full((5,64),-10.,dtype=torch.bfloat16),'lse':torch.zeros(5)}
    shard = writer.write_shard(cache,0,arrays,record,64,248320)
    shard_path = cache/shard['shard']/'shard.json'
    spin = write(cache/'sessions'/'session-session-A.json',sessions['session-A'])
    manifest.update(corpus={'sha256':P.sha(native)},shards=[{'shard':shard['shard'],'shard_json_sha256':P.sha(shard_path)}],
                    sessions=[{'session':'session-A','record_sha256':spin['sha256']}])
    manifest_path = cache/'manifest.json'; write(manifest_path,manifest)
    forward = tmp_path/'forward.json'
    observed = {'schema':'chimera_recovery.student_prefix_teacher_runtime.v1','passed':True,
        'teacher_plan_sha256':P.sha(plan_path),'cache_manifest_sha256':P.sha(manifest_path),
        'runtime_profile':plan['runtime_profile'],'sessions':{}}
    runtime_path = tmp_path/'runtime.json'
    def reseal_metadata():
        # Fixture evidence refreshed to isolate the failed semantic shape guard.
        manifest['shards'][0]['shard_json_sha256']=P.sha(shard_path)
        manifest['sessions'][0]['record_sha256']=P.sha(Path(spin['path']))
        write(manifest_path,manifest)
        write(forward,{'passed':True,'native_singleton_forward_verified':True,'fixture_only':True,
            'cache_session_sha256':P.sha(Path(spin['path'])),'runtime_profile':plan['runtime_profile'],
            'allocation_id':'00000000-0000-0000-0000-000000000002','container_id':'c'*64,
            'gpu_uuid':'GPU-00000000-0000-0000-0000-000000000001'})
        observed['sessions']['session-A']={'cache_session_sha256':P.sha(Path(spin['path'])),'allocation_forward_proof':P.pin(forward)}
        observed['cache_manifest_sha256']=P.sha(manifest_path); write(runtime_path,observed)
    reseal_metadata()
    result = P.verify_capture(plan_path,corpus_path,cache,runtime_path)
    assert result['passed'] and result['rows']==2 and result['supervised_positions']==5
    assert not result['production_training_ready']
    original = json.loads(shard_path.read_text())
    write(shard_path,{**original,'padded_tokens':18}); reseal_metadata()
    with pytest.raises(P.Error,match='padded'): P.verify_capture(plan_path,corpus_path,cache,runtime_path)
    write(shard_path,original); reseal_metadata()
    bad_session = copy.deepcopy(sessions['session-A']); bad_session['runtime']['versions']['torch']='other-runtime'
    write(Path(spin['path']),bad_session); reseal_metadata()
    with pytest.raises(P.Error,match='session/runtime'): P.verify_capture(plan_path,corpus_path,cache,runtime_path)
    write(Path(spin['path']),sessions['session-A']); reseal_metadata()
    # Hash-valid, finite raw rows from a different token prefix must still fail.
    target = cache/shard['shard']/'ids.u32'
    import struct
    payload = bytearray(target.read_bytes()); struct.pack_into('<I',payload,0,99); target.write_bytes(payload)
    tampered = copy.deepcopy(original); tampered['files']['ids.u32']['sha256']=P.sha(target)
    write(shard_path,tampered); reseal_metadata()
    with pytest.raises(P.Error,match='stored IDs/positions'): P.verify_capture(plan_path,corpus_path,cache,runtime_path)

