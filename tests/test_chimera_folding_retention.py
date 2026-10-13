import copy
import pytest
from experiments.chimera_recovery import folding_retention as F


def contract():
    return dict(schema='chimera_recovery.fold_rollout_contract.v1', decoding={'strategy':'greedy','do_sample':False},
                vocab_size=1000,max_new_tokens=40,eos_ids=[999],tokenizer_sha256='1'*64,evaluation_source_sha256='2'*64,
                parent_checkpoint_sha256='3'*64,candidate_checkpoint_sha256='4'*64,training_source_groups=['train-source'])


def row(parent,child,ps='eos',cs='eos'):
    p=[81,82]
    return dict(id='held-one',source_group_id='held-group',split='test',prompt_ids=p,
                parent=dict(mode='free_running',generated_ids=parent,prompt_sha256=F.digest(p),checkpoint_sha256='3'*64,stop_reason=ps),
                candidate=dict(mode='free_running',generated_ids=child,prompt_sha256=F.digest(p),checkpoint_sha256='4'*64,stop_reason=cs))


def test_identical_first_token_does_not_equal_five_correct_tokens():
    parent=list(range(1,11))+[999]
    first=F.score([row(parent,[1,77,78,79,80,999])],contract())
    five=F.score([row(parent,[1,2,3,4,5,77,999])],contract())
    assert first['summary']['consecutive_prefix']['5']['matches']==0
    assert five['summary']['consecutive_prefix']['5']['matches']==1
    assert five['summary']['consecutive_prefix']['8']['matches']==0


def test_eos_only_probe_cannot_pass_long_horizons_or_completion():
    result=F.score([row([999],[999])],contract())
    assert result['summary']['exact_complete_generations']==0
    assert result['summary']['candidate_empty']==1
    assert all(v['eligible']==0 and v['rate'] is None for v in result['summary']['consecutive_prefix'].values())


def test_short_correct_answer_counts_completion_but_not_unobserved_horizon():
    result=F.score([row([1,2,999],[1,2,999])],contract())
    assert result['summary']['exact_complete_generations']==1
    assert result['summary']['consecutive_prefix']['5']['rate'] is None


def test_capped_repetition_is_not_success_even_when_parent_matches():
    ids=[1,2,3,4]*10
    result=F.score([row(ids,ids,'max_tokens','max_tokens')],contract())
    assert result['summary']['exact_complete_generations']==0
    assert result['summary']['candidate_capped']==1
    assert result['rows'][0]['candidate_repetition']['max_occurrences']==10


def test_later_run_does_not_inflate_correct_prefix():
    result=F.score([row([1,2,3,4,5,6,7,999],[1,9,3,4,5,6,7,999])],contract())
    assert result['rows'][0]['longest_correct_prefix']==1
    assert result['rows'][0]['longest_aligned_match_run']==5
    assert result['summary']['consecutive_prefix']['5']['matches']==0


@pytest.mark.parametrize('mutation', ['teacher_forced','wrong_prompt','wrong_checkpoint','training_group','cap_lie','bad_eos'])
def test_reject_misleading_rollout_evidence(mutation):
    r=row([1,2,999],[1,2,999])
    if mutation=='teacher_forced':r['candidate']['mode']='teacher_forced'
    elif mutation=='wrong_prompt':r['candidate']['prompt_sha256']='0'*64
    elif mutation=='wrong_checkpoint':r['candidate']['checkpoint_sha256']='0'*64
    elif mutation=='training_group':r['source_group_id']='train-source'
    elif mutation=='cap_lie':r['candidate']['stop_reason']='max_tokens'
    else:r['candidate']['generated_ids']=[1,999,2]
    with pytest.raises(ValueError):F.score([r],contract())


def test_top_five_rank_field_does_not_change_sequence_result():
    r=row([1,2,3,4,5,999],[1,7,8,9,10,999]);before=F.score([r],contract())
    r['candidate']['top5_rank_hits']=[True]*5
    after=F.score([r],contract())
    assert before['summary']==after['summary']


def test_duplicate_held_ids_and_empty_reports_refused():
    r=row([1,999],[1,999])
    with pytest.raises(ValueError):F.score([r,copy.deepcopy(r)],contract())
    with pytest.raises(ValueError):F.score([],contract())
