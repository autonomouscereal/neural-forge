"""Actual unchanged 2026 ACE/QIMG/H3 producer receipt interoperability."""
import copy
import json
import os
from pathlib import Path

import pytest
from experiments.chimera_recovery import consolidation_dispatch as D


def actual_inputs():
    path=Path(os.environ.get('CHIMERA_GUIDED_CATALOG_FIXTURES',
        'C:/Users/cereal/Documents/ai training/reviews/2026-10-13-folding/recovery-preparation/target-preparation-v2/guided-metadata-fixtures/index.json'))
    if not path.is_file():pytest.skip('actual source-pinned producer fixtures not supplied')
    index=json.loads(path.read_text())
    rows=[]
    for entry in index['lanes']:
        data={}
        for name in ('manifest','receipt'):
            ref=entry[name];p=path.parent/ref['path']
            assert D.I.sha(p)==ref['sha256']
            data[name]=json.loads(p.read_text())
        data['lane']=entry['lane'];rows.append(data)
    return rows


def test_actual_producer_metadata_binds_exact_native_membership_without_rewriting_receipt():
    rows=actual_inputs();assert {row['lane'] for row in rows}=={'ace','qimg','h3'}
    for row in rows:
        manifest,receipt=row['manifest'],row['receipt'];item=manifest['batches'][0]
        original=copy.deepcopy(receipt)
        assert 'tensors_sha256' not in receipt and 'recovery_group_id' not in receipt
        metadata=D.guided_receipt_metadata(row['lane'],receipt,item,manifest)
        assert metadata['tensors_sha256']==item['tensors']['sha256']
        assert metadata['recovery_group_id']==item['recovery_group_id']
        assert receipt==original
        for field in ('cache_sha256','route','corpus_sha256'):
            wrong=copy.deepcopy(receipt);wrong[field]='f'*64
            with pytest.raises(D.Error,match='identity differs'):
                D.guided_receipt_metadata(row['lane'],wrong,item,manifest)
        wrong=copy.deepcopy(receipt);wrong['source']['recovery_group_id']='held'
        with pytest.raises(D.Error,match='identity differs'):
            D.guided_receipt_metadata(row['lane'],wrong,item,manifest)
        wrong=copy.deepcopy(receipt);wrong['held']=True
        with pytest.raises(D.Error,match='holdout'):
            D.guided_receipt_metadata(row['lane'],wrong,item,manifest)
        wrong=copy.deepcopy(manifest);wrong['split']='validation'
        with pytest.raises(D.Error,match='identity differs'):
            D.guided_receipt_metadata(row['lane'],receipt,item,wrong)


def test_actual_h3_source_match_and_original_guided_contract_cannot_be_bypassed():
    row=next(row for row in actual_inputs() if row['lane']=='h3')
    receipt,manifest=row['receipt'],row['manifest'];item=manifest['batches'][0]
    for change in ({'source_id':'invented-source'},{'diagnostic_sha256':'f'*64}):
        wrong=copy.deepcopy(receipt);wrong['source'].update(change)
        with pytest.raises(D.Error):D.guided_receipt_metadata('h3',wrong,item,manifest)
    wrong=copy.deepcopy(receipt);wrong['guided_teacher']=False
    with pytest.raises(ValueError,match='sealed guided'):
        D.guided_receipt_metadata('h3',wrong,item,manifest)
    wrong=copy.deepcopy(manifest);wrong['uncached_held_source_ids'].append(item['source_id'])
    with pytest.raises(D.Error,match='held diagnostic'):
        D.guided_receipt_metadata('h3',receipt,item,wrong)
