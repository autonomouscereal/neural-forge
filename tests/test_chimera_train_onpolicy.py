"""Train-only admission/real native objective tests; synthetic fixtures are labeled."""
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.chimera_recovery import train_onpolicy as O, train_text as T
from test_chimera_recovery_train_text_integration import production_fixture


def write(path, value):
    path.write_bytes(O.canonical(value) + b"\n")
    return O.pin(path)


def row(identifier, content, *, split="train", group=None):
    return {"schema": "recovery_corpus.v1", "id": identifier, "split": split, "route": "text_arithmetic",
            "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": "Private source answer must not enter prompt inputs."}],
            "quality": {"train_ready": True, "blockers": []}, "media": [], "tools": [],
            "provenance": {"source_verified": True, "recovery_group_id": group or identifier, "split": split}}


def test_prompt_source_group_holdout_and_global_exclusion_not_answer_leakage(tmp_path):
    clean = row("clean", "Compute the discounted total for nine crates each priced seventeen dollars after a five percent discount.")
    held = row("held", "Find the sum of all prime integers from forty through ninety excluding those congruent to three modulo five.")
    same_group = row("held-group-alias", "Determine compound interest on a deposit of fifteen thousand units at a rate of seven percent.", group="held")
    excluded = row("excluded", "Calculate the average speed of a train traveling ninety kilometers during two hours and thirty minutes.")
    train = tmp_path / "train.jsonl"
    train.write_bytes(b"".join(O.canonical(r)+b"\n" for r in (clean, held, same_group, excluded, copy.deepcopy(clean))))
    holdout = tmp_path / "holdout.jsonl"
    holdout.write_bytes(O.canonical(dict(held, split="test"))+b"\n")
    exclusions = tmp_path / "exclude.jsonl"
    exclusions.write_bytes(O.canonical({"id": "excluded", "source_group_id": "excluded"})+b"\n")
    result = O.prepare_prompts([train], [holdout], [exclusions], tmp_path/"prompts", per_route=4, split_inventory=holdout)
    prompts = list(O.C.iter_jsonl(tmp_path/"prompts"/"prompts.jsonl"))
    assert result["rows"] == 1 and prompts[0]["source_row_id"] == "clean"
    assert prompts[0]["input"]["messages"] == clean["messages"][:1]
    assert "Private source answer" not in (tmp_path/"prompts"/"prompts.jsonl").read_text()
    assert result["actual_sampled_tokens"] == 0 and result["maximum_sampled_tokens"] == 256
    assert not result["production_training_ready"] and not result["student_generation_completed"]
    assert result["screened_out"]["heldout_text_or_group_overlap"] == 1
    assert result["screened_out"]["duplicate_source_group_or_prompt"] == 2
    with pytest.raises(O.Error, match="new output"):
        O.prepare_prompts([train], [holdout], [exclusions], tmp_path/"prompts", split_inventory=holdout)


def trace_fixture():
    seed = {"id": "seed", "split": "train", "group_id": "train-family", "input_messages_sha256": "d"*64}
    expected = {"student_checkpoint_sha256": "a"*64, "tokenizer_sha256": "b"*64, "template_sha256": "c"*64,
                "generation": {"do_sample": False, "temperature": 0, "max_new_tokens": 32, "eos_ids": [2]}}
    trace = {"schema": "chimera_recovery.student_trace.v1", "id": seed["id"], "group_id": seed["group_id"],
             "seed_sha256": O.digest(seed), "input_messages_sha256": seed["input_messages_sha256"], **expected,
             "prompt_ids": [9, 8], "generated_ids": [1, 2], "stop_reason": "eos", "answer_text": "Synthetic wrong answer fixture"}
    trace.update(prompt_ids_sha256=O.digest(trace["prompt_ids"]), generated_ids_sha256=O.digest(trace["generated_ids"]),
                 answer_text_sha256=O.digest(trace["answer_text"]))
    return seed, trace, expected


def test_trace_wrong_semantics_remains_context_but_pins_eos_and_holdout_fail_closed():
    seed, trace, expected = trace_fixture()
    ids, mask = O.validate_trace(seed, trace, expected)
    assert ids == [9,8,1,2] and mask == [False,False,True,True]
    for field, wrong in (("template_sha256", "f"*64), ("group_id", "held-family"), ("stop_reason", "cap"),
                         ("generated_ids", [1,2,3]), ("prompt_ids", [248320]), ("answer_text", "tampered")):
        with pytest.raises(O.Error):
            O.validate_trace(seed, dict(trace, **{field:wrong}), expected)
    with pytest.raises(O.Error):
        O.validate_trace(dict(seed, split="test"), trace, expected)
    # A real capped repeated rollout is retained and separately measurable;
    # it must not be mislabeled EOS or assigned a hard-target CE objective.
    repeated = dict(trace, generated_ids=[1]*32, generated_ids_sha256=O.digest([1]*32), stop_reason="cap")
    assert len(O.validate_trace(seed, repeated, expected)[0]) == 34
    assert O.OBJECTIVE["w_ce"] == O.OBJECTIVE["w_eos"] == O.OBJECTIVE["w_ul"] == 0


def tool_fixture(split='train', family='train-uppercase'):
    """Operator-authored 2026 mechanism fixture; no benchmark/model score."""
    tools=[{'type':'function','function':{'name':'local_text','parameters':{
        'type':'object','properties':{'text':{'type':'string','maxLength':12}},
        'required':['text'],'additionalProperties':False}}}]
    result={'text':'ABC'}
    observed={'call_id':'call-1','tool_name':'local_text','arguments':{'text':'abc'},
              'result':result,'result_sha256':O.digest(result),'dispatch_group':0,'concurrent_dispatch':False}
    identifier=split+'-episode'
    record={'id':identifier,'split':split,'template_family':family,'calls':[observed]}
    episode={'schema':'recovery_corpus.v1','id':identifier,'route':'tool_calling','tools':tools,'media':[],
        'messages':[{'role':'system','content':'Use only the authentic local schema.'},
                    {'role':'user','content':'Uppercase the literal abc through the local text function.'},
                    {'role':'assistant','content':None,'tool_calls':[{'id':'call-1','type':'function',
                        'function':{'name':'local_text','arguments':'{"text":"abc"}'}}]},
                    {'role':'tool','tool_call_id':'call-1','content':json.dumps(result)},
                    {'role':'assistant','content':'ABC is the recorded source outcome.'}],
        'quality':{'train_ready':True,'tool_routes':['tool_call','tool_result']},
        'provenance':{'split':split,'group_id':'family-'+family,'template_family':family,'case_id':identifier,
            'source_ids':[identifier],'actual_tool_execution':True,'source_verified':True,
            'execution_receipt_sha256':O.digest(record['calls']),'source_code_sha256':'c'*64,'tool_catalog_sha256':'d'*64}}
    return episode,record


def test_tool_source_history_requires_observed_results_and_valid_authentic_schema():
    episode,record=tool_fixture()
    assert len(O.source_tool_calls(episode,record))==1  # maxLength is an authentic supported constraint.
    invalid=copy.deepcopy(episode)
    invalid['messages'][2]['tool_calls'][0]['function']['arguments']='{"text":123}'
    with pytest.raises(O.Error,match='schema/arguments'): O.source_tool_calls(invalid,record)
    invalid=copy.deepcopy(episode); invalid['messages'][3]['content']='{"text":"WRONG"}'
    with pytest.raises(O.Error,match='observed result'): O.source_tool_calls(invalid,record)
    invalid=copy.deepcopy(episode); invalid['messages'].pop(3)
    with pytest.raises(O.Error,match='incomplete'): O.source_tool_calls(invalid,record)
    invalid=copy.deepcopy(episode); invalid['messages'].append(copy.deepcopy(invalid['messages'][3]))
    with pytest.raises(O.Error,match='incomplete'): O.source_tool_calls(invalid,record)
    invalid=copy.deepcopy(record); invalid['calls'].append(copy.deepcopy(invalid['calls'][0]))
    with pytest.raises(O.Error,match='receipt mismatch'): O.source_tool_calls(episode,invalid)


def test_tool_trace_cannot_inherit_historical_execution_or_different_schema():
    seed,trace,expected=trace_fixture()
    episode,_=tool_fixture()
    seed['input']={'tools':episode['tools']}; trace['seed_sha256']=O.digest(seed)
    with pytest.raises(O.Error,match='unobserved outcomes'): O.validate_trace(seed,trace,expected)
    trace.update(native_tools_sha256=O.digest(episode['tools']),execution_scope='initial_assistant_only_no_tool_dispatch',tool_outcomes=[])
    assert O.validate_trace(seed,trace,expected)[1]==[False,False,True,True]
    for field,value in (('tool_outcomes',[{'result':'ABC'}]),('native_tools_sha256','e'*64),('execution_scope','unexecuted_multiturn')):
        with pytest.raises(O.Error,match='unobserved outcomes'): O.validate_trace(seed,dict(trace,**{field:value}),expected)


def test_tool_augmentation_preserves_input_schema_and_excludes_source_assistant_results(tmp_path):
    train=tmp_path/'train.jsonl'; train.write_bytes(O.canonical(row('text','Compute the discounted total for nine crates each priced seventeen dollars after a five percent discount.'))+b'\n')
    held=tmp_path/'held.jsonl'; held.write_bytes(O.canonical(row('held','Describe how a compiler allocates stack slots for local variables during register spilling.',split='test'))+b'\n')
    exclusions=tmp_path/'excluded.jsonl'; exclusions.write_bytes(O.canonical({'id':'excluded','source_group_id':'excluded'})+b'\n')
    base=tmp_path/'base'; O.prepare_prompts([train],[held],[exclusions],base,split_inventory=held)
    episode,record=tool_fixture()
    ready=tmp_path/'ready.jsonl'; ready.write_bytes(O.canonical(episode)+b'\n')
    executions=tmp_path/'executions.jsonl'; executions.write_bytes(O.canonical(record)+b'\n')
    report=tmp_path/'report.json'
    report_data={'artifacts':{'ready.jsonl':O.pin(ready),'execution_receipts.jsonl':O.pin(executions)},
                 'source_code_sha256':'c'*64,'tool_catalog_sha256':'d'*64}
    write(report,report_data)
    result=O.add_tool_prompts(base,ready,executions,report,tmp_path/'out',per_family=1)
    seed=next(r for r in O.C.iter_jsonl(tmp_path/'out'/'prompts.jsonl') if r['route']=='tool_call')
    assert seed['input']['tools']==episode['tools'] and seed['input']['messages']==episode['messages'][:2]
    assert 'recorded source outcome' not in (tmp_path/'out'/'prompts.jsonl').read_text()
    assert result['rows']==2 and result['maximum_sampled_tokens']==512 and result['actual_sampled_tokens']==0
    held_episode,held_record=tool_fixture('test',episode['provenance']['template_family'])
    ready.write_bytes(O.canonical(episode)+b'\n'+O.canonical(held_episode)+b'\n')
    executions.write_bytes(O.canonical(record)+b'\n'+O.canonical(held_record)+b'\n')
    report_data['artifacts']={'ready.jsonl':O.pin(ready),'execution_receipts.jsonl':O.pin(executions)}; write(report,report_data)
    with pytest.raises(O.Error,match='crosses splits'):
        O.add_tool_prompts(base,ready,executions,report,tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()


def test_onpolicy_objective_and_exact_sampler_lineage_not_offpolicy_alias():
    args = SimpleNamespace(structured_recovery=True, **O.OBJECTIVE)
    O.verify_objective(args)
    for key,value in (("structured_recovery",False), ("w_ce",.1), ("onpolicy_divergence","fkl"), ("jsd_beta",.1)):
        with pytest.raises(O.Error): O.verify_objective(SimpleNamespace(**{**vars(args),key:value}))
    sources=[SimpleNamespace(name="snapshot",row_ids=list(range(7)))]
    sampler = O.OnePassSampler(sources, 7)
    first=sampler.next_rows(4); state=sampler.state(); tail=sampler.next_rows(4)
    assert len(first+tail)==7 and len(set(first+tail))==7 and state['policy']==O.POLICY
    restored=O.OnePassSampler(sources,7); restored.load_state(state); assert restored.next_rows(4)==tail
    with pytest.raises(O.Error,match="policy"): restored.load_state(T.OnePassSampler(sources,7).state())


def test_actual_candidate_guard_runs_before_any_source_loading():
    path=Path(os.environ.get('CHIMERA_STREAMED_BASELINE', 'C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/distill/tools/streamed_kd_trainer.py'))
    if not path.is_file(): pytest.skip('exact audited streamed baseline not staged')
    candidate,receipt=O.patch_trainer(path)
    node=next(n for n in ast.parse(candidate).body if isinstance(n,ast.FunctionDef) and n.name=='command_train')
    source=candidate.splitlines()[node.lineno-1:node.end_lineno]
    start=next(i for i,v in enumerate(source) if 'structured = bool(' in v)
    end=next(i for i,v in enumerate(source[start:],start) if 'kernels = select_' in v)
    fragment='\n'.join(v[4:] for v in source[start:end])
    namespace={'args':SimpleNamespace(structured_recovery=True,allow_config_change=False,allow_identity_change=False,no_preregistration=False,**{**O.OBJECTIVE,'w_ce':.1}),
               'world':1,'sys':__import__('sys'),'Path':Path,'__file__':__file__,'TrainerError':O.Error}
    with pytest.raises(O.Error,match='objective'):
        exec(compile(fragment,'actual-production-pre-source-guard','exec'),namespace)
    namespace['args'].structured_recovery=False
    with pytest.raises(O.Error,match='requires --structured'):
        exec(compile(fragment,'actual-production-mode-guard','exec'),namespace)
    assert receipt['objective']==O.OBJECTIVE and not receipt['production_training_ready']


def test_no_invented_generation_receipt_before_torch_or_output(tmp_path):
    seed, trace, _=trace_fixture()
    prompts=tmp_path/'prompts.jsonl'; prompts.write_bytes(O.canonical(seed)+b'\n')
    manifest=tmp_path/'manifest.json'; write(manifest,{'schema':'chimera_recovery.onpolicy_prompts.v1','prompts_sha256':O.sha(prompts)})
    traces=tmp_path/'traces.jsonl'; traces.write_bytes(O.canonical(trace)+b'\n')
    receipt=tmp_path/'receipt.json'; write(receipt,{'completed':True,'accepted_student':False})
    with pytest.raises(O.Error,match='exact complete'):
        O.pack_traces(prompts,manifest,traces,receipt,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_import_exact_accepted_bank_and_original_base_before_optimizer(tmp_path):
    bank = tmp_path/'fixture-bank.bin'; bank.write_bytes(b'synthetic mechanism fixture; not model weights')
    acceptance = tmp_path/'acceptance.json'
    identity={'fixture_source':20261013}
    accepted={'accepted_complete_stage':True,'stage':'D2','checkpoint_sha256':'a'*64,
              'base_source_identity_sha256':O.digest(identity),'bank_scope':'D2.full_bank_replacement','adapters':O.pin(bank)}
    write(acceptance,accepted)
    generation = tmp_path/'generation.json'; write(generation,{'scope':'synthetic fixture only'})
    corpus = tmp_path/'corpus.json'
    write(corpus,{'identity':{'student_checkpoint_sha256':'a'*64},'generation_receipt':O.pin(generation),'objective':O.OBJECTIVE})
    round_path=tmp_path/'round.json'
    evidence={**{k:accepted[k] for k in ('student_checkpoint_sha256',) if k in accepted},
              'schema':'chimera_recovery.student_prefix_round.v1','student_checkpoint_sha256':'a'*64,
              'base_source_identity_sha256':O.digest(identity),'student_acceptance_receipt':O.pin(acceptance),
              'init_adapters':O.pin(bank),'generation_receipt':O.pin(generation)}
    plan=tmp_path/'teacher-plan.json'; write(plan,{'fixture_only':True})
    capture=tmp_path/'rescoring.json'; write(capture,{'fixture_only':True})
    evidence.update(teacher_plan=O.pin(plan),rescoring_receipt=O.pin(capture))
    write(round_path,evidence)
    data=tmp_path/'data.json'; write(data,{'student_prefix_round':O.pin(round_path),'student_prefix_corpus':O.pin(corpus),
        'teacher_plan':O.pin(plan),'rescoring_receipt':O.pin(capture)})
    args=SimpleNamespace(student_prefix_round=str(round_path),init_adapters=str(bank),data_config=str(data))
    O.verify_source(args,SimpleNamespace(identity=lambda:identity))
    with pytest.raises(O.Error,match='already composed'):
        O.verify_source(args,SimpleNamespace(identity=lambda:{**identity,'already_composed':True}))
    bank.write_bytes(b'changed same path')
    with pytest.raises(O.Error,match='full-bank'):
        O.verify_round(args)


def test_actual_cached_student_prefix_jsd_gradient_and_no_hard_wrong_token_imitation(production_fixture):
    """Real patched cache reader/objective; tiny targets are synthetic, never campaign scores."""
    import torch
    module,_,_,_,root,data=production_fixture
    spec=json.loads(data.read_text()); spec['sources'][0]['kind']='onpolicy'; write(data,spec)
    sources,_=module.load_data_config(data)
    objective=module.ObjectiveConfig(w_q38=1,w_ce=0,w_eos=0,w_ul=0,onpolicy_divergence='jsd',jsd_beta=.5,text_vocab_rows=32)
    sampler=O.OnePassSampler(sources,20261013)
    batches,stats=O.accumulation_window(sources,sampler,2,2,objective,module.build_batch)
    ids,targets=batches[0]; units=stats['supervised_positions']
    assert targets['jsd'].any() and not targets['ce_w'].any() and not targets['eos_w'].any()
    assert torch.equal(targets['jsd'],targets['q_w']>0)
    normed=torch.randn(*ids.shape,4,generator=torch.Generator().manual_seed(912)); head=torch.randn(32,4,generator=torch.Generator().manual_seed(913))
    parts,gradient=module.chunked_objective(normed,head,targets,objective,denominator=units)
    # Independently evaluate the three buckets (two cached top IDs + tail).
    q=torch.softmax(normed.reshape(-1,4)@head.T,-1).gather(1,targets['q_ids'].clamp_min(0))
    p=targets['q_lp'].float().exp()
    p=torch.cat((p,(1-p.sum(-1,keepdim=True)).clamp_min(0)),-1)
    q=torch.cat((q,(1-q.sum(-1,keepdim=True)).clamp_min(0)),-1)
    mixture=(p+q)/2
    expected=(.5*(p*(p.clamp_min(1e-30).log()-mixture.clamp_min(1e-30).log())).sum(-1)
              +.5*(q*(q.clamp_min(1e-30).log()-mixture.clamp_min(1e-30).log())).sum(-1))
    assert abs(parts['q38']-float((expected*targets['q_w']).sum()/units))<1e-6
    assert parts['ce']==parts['eos']==parts['ul']==0 and gradient.isfinite().all()
    altered={**targets,'next_ids':torch.ones_like(targets['next_ids'])*17}
    other,other_gradient=module.chunked_objective(normed,head,altered,objective,denominator=units)
    assert other==parts and torch.equal(gradient,other_gradient)
    fkl=module.chunked_objective(normed,head,{**targets,'jsd':torch.zeros_like(targets['jsd'])},objective,denominator=units)[0]
    assert abs(fkl['q38']-parts['q38'])>1e-4
