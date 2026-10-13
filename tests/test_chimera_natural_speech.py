import copy,json
import pytest
from experiments.chimera_recovery import natural_speech as N

def row():
    asset={'kind':'audio','media':'audio','path':'/source/speech.flac','sha256':'a'*64,'byte_size':45}
    answer={'audio_sha256':'a'*64,'transcript':'A bird flew across the lake.','speaker_id':7,'chapter_id':8}
    return {'schema':'recovery_corpus.v1','id':'original-source','route':'audio_understand','split':'validation',
      'media':[asset],'messages':[{'role':'user','content':'Original task. Expected transcript: '+answer['transcript']},
        {'role':'assistant','content':json.dumps(answer)}],'provenance':{'recovery_group_id':'group','group_id':'oldgroup',
        'source_dates':['2026-10-13T00:39:44Z'],'source_date_basis':'collection; original author date not inferred',
        'registry_origin':{'source_dataset':'openslr/librispeech_asr','task_type':'speech_transcription_train_clean_100',
           'rid':'source:1:8','shard_sha256':'b'*64,'shard_mirror':'/source/original.jsonl','license':'source-license'}},
      'quality':{'train_ready':False,'blockers':['original_blocker']}}

def test_authentic_cue_removed_and_original_provenance_held_split_preserved():
    original=row();before=copy.deepcopy(original);fixed=N.repair(original)
    assert original==before and fixed['messages'][0]['content']==N.PROMPT
    assert fixed['messages'][-1]['content']=='A bird flew across the lake.'
    assert fixed['media'][0]['role']=='input' and fixed['media'][0]['sha256']=='a'*64
    assert fixed['provenance']['parent_source_row_sha256']==N.digest(original)
    assert fixed['provenance']['registry_origin']==original['provenance']['registry_origin']
    assert fixed['split']=='validation' and fixed['provenance']['recovery_group_id']=='group'
    assert fixed['quality']['train_ready'] is False and fixed['quality']['source_train_eligible'] is False
    assert 'original_data_date_2025_2026_unverified' in fixed['quality']['blockers']
    assert fixed['provenance']['original_data_date_verified'] is False

@pytest.mark.parametrize('change',[
 lambda r:r['messages'][0].update(content='Expected transcript: Incorrect answer'),
 lambda r:r['media'][0].update(sha256='c'*64),lambda r:r.update(route='audio_generate'),
 lambda r:r['provenance']['registry_origin'].pop('license'),
 lambda r:r['provenance'].pop('recovery_group_id')])
def test_changed_answer_binding_or_source_contract_refused(change):
    value=row();change(value)
    with pytest.raises(N.Error):N.repair(value)

def test_full_inventory_counts_kind_aliases_and_never_promotes_collection_date(tmp_path):
    rows=[row(),{**row(),'id':'generation','route':'audio_generate'}]
    p=tmp_path/'original.jsonl';p.write_bytes(b'\n'.join(N.canonical(r) for r in rows)+b'\n')
    manifest=N.audit(p,N.sha(p),tmp_path/'audit')
    assert manifest['audio_media_routes']=={'audio_understand':1,'audio_generate':1}
    assert manifest['outputs']['blocked_repaired_asr.jsonl']['rows']==1
    assert manifest['outputs']['original_inventory.jsonl']['rows']==2
    assert manifest['train_admitted']==0 and manifest['native_ready'] is False
    with pytest.raises(N.Error,match='preserve'):N.audit(p,N.sha(p),tmp_path/'audit')
    with pytest.raises(N.Error,match='pin differs'):N.audit(p,'0'*64,tmp_path/'changed')

def test_audio_restoration_roles_are_counted_without_becoming_speech_observations():
    value={'media':[{'kind':'audio_input','media':'audio','role':'degraded_input'},
                    {'kind':'audio_target','media':'audio','role':'clean_ground_truth'}]}
    assert len(N.audio_assets(value))==2
    with pytest.raises(N.Error,match='unsupported'):N.repair(value)
