"""Malformed provenance/member regressions, not teacher or capability scores."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from experiments.chimera_recovery import recovery_readiness as R


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def write(self,name,data):
        path=self.root/name
        path.write_text(json.dumps(data,sort_keys=True),encoding='utf-8')
        return R.pin(path)
    def cache(self,role,count=1):
        stage,obj,schema=R.ROLES[role];items=[]
        for i in range(count):
            tensor=self.root/f'{role}-{i}.safetensors';tensor.write_bytes(b'explicit non-tensor fixture')
            receipt={'sealed':True,'split':'train','held':False,'id':f'{role}-{i}',
                'recovery_group_id':f'{role}-group','tensors_sha256':R.sha(tensor),
                'source_row_sha256':'1'*64,'sigma':.05,'noise_seed':7,'assistant_targets':3}
            if role=='image_edit':receipt['supervised_units']=65536
            rp=self.write(f'{role}-{i}.json',receipt)
            items.append({'id':f'{role}-{i}','source_id':f'{role}-source-{i}',
                'route':R.ROUTES[role][0],'recovery_group_id':f'{role}-group','teacher_receipt':rp,'tensors':R.pin(tensor)})
        data={'schema':schema,'stage':stage,'objective':obj,'sealed':True,'complete_source_traversal':True,
            'split':'train','heldout_groups_excluded':True,'batches':items}
        if role in {'speech_flow','image_edit'}:data['sigma_grid']=[.05]
        path=self.root/f'{role}-manifest.json';path.write_text(json.dumps(data))
        return data,path
    def index(self,caches):
        rows=[];summary={}
        for role,(d,p) in caches.items():
            for item in d['batches']:
                receipt=json.loads(Path(item['teacher_receipt']['path']).read_text())
                rows.append({**item,'objective':R.ROLES[role][1],'source_manifest':R.pin(p),
                    'source_row_sha256':'1'*64,'supervised_units':3 if role=='audio_ce' else 128,
                    'native_audio_rows':4,'latent_frames_per_channel':2,'sigma':.05,'noise_seed':7,
                    'denominator_kind':'assistant_next_token_positions_including_completion_eos' if role=='audio_ce'
                        else 'native_weighted_audio_rows_times32_emission_channels'})
            summary[R.ROLES[role][1]]={'batches':len(d['batches']),'supervised_units':len(d['batches'])*(3 if role=='audio_ce' else 128)}
        return {'schema':'chimera_recovery.audio_speech_replay_index.v1','sealed':True,'passed':True,'verified':True,
            'complete_source_traversal':True,'source_reconstruction_bitwise_equal':True,'entries':rows,'objectives':summary}
    def inventory(self,caches):
        return {k:R.membership(k,d,p,R.Locator()) for k,(d,p) in caches.items()}
    def test_metadata_cannot_promote_non_tensor_fixture_to_reconstruction_or_production(self):
        d,p=self.cache('audio_ce');r=R.membership('audio_ce',d,p,R.Locator())
        self.assertEqual(r['received_receipts'],1);self.assertEqual(r['recorded_supervised_units'],{'assistant_next_token_positions':3})
        self.assertFalse(r['tensor_reconstruction_performed']);self.assertFalse(r['production_ready'])
    def test_changed_existing_file_cannot_be_rescued_by_alias_to_old_valid_bytes(self):
        p=self.root/'existing';p.write_bytes(b'first');ref=R.pin(p)
        old=self.root/'old';old.write_bytes(b'first');p.write_bytes(b'other')
        with self.assertRaisesRegex(R.Error,'bytes changed'):R.Locator({ref['sha256']:str(old)}).file(ref,self.root)
    def test_same_size_receipt_tamper_refuses_even_with_pass_flags(self):
        d,p=self.cache('audio_ce');rp=Path(d['batches'][0]['teacher_receipt']['path'])
        rp.write_text(rp.read_text().replace('"assistant_targets": 3','"assistant_targets": 9'))
        with self.assertRaisesRegex(R.Error,'bytes changed'):R.membership('audio_ce',d,p,R.Locator())
    def test_missing_receipt_is_inventory_dependency_not_passed_membership(self):
        d,p=self.cache('audio_ce');Path(d['batches'][0]['teacher_receipt']['path']).unlink();l=R.Locator()
        result=R.membership('audio_ce',d,p,l)
        self.assertFalse(result['metadata_membership_verified']);self.assertEqual(result['received_receipts'],0)
        self.assertTrue(l.missing);self.assertFalse(result['production_ready'])
    def test_sealed_wrong_route_and_missing_sigma_pair_refuse(self):
        d,p=self.cache('speech_flow');bad=copy.deepcopy(d);bad['batches'][0]['route']='image_generate'
        with self.assertRaisesRegex(R.Error,'different lane'):R.membership('speech_flow',bad,p,R.Locator())
        d['sigma_grid']=[.05,.3]
        with self.assertRaisesRegex(R.Error,'sigma grid'):R.membership('speech_flow',d,p,R.Locator())
    def test_index_changed_denominator_and_omitted_actual_target_refuse(self):
        caches={r:self.cache(r,2) for r in ('audio_ce','speech_flow')};inv=self.inventory(caches)
        index=self.index(caches);ref=self.write('index.json',index)
        result=R.audio_index(ref,self.root,R.Locator(),inv)
        self.assertEqual(result['objectives'][R.D.O.SPEECH]['supervised_units'],256)
        self.assertFalse(result['tensor_reconstruction_repeated'])
        wrong=copy.deepcopy(index);wrong['entries'][-1]['supervised_units']=64
        with self.assertRaisesRegex(R.Error,'scalar denominator'):R.audio_index(self.write('wrong.json',wrong),self.root,R.Locator(),inv)
        wrong=copy.deepcopy(index);wrong['entries'].pop()
        with self.assertRaisesRegex(R.Error,'omitted'):R.audio_index(self.write('omit.json',wrong),self.root,R.Locator(),inv)
    def test_index_tensor_or_held_group_cannot_be_substituted_by_new_passing_summary(self):
        caches={r:self.cache(r) for r in ('audio_ce','speech_flow')};inv=self.inventory(caches)
        index=self.index(caches);index['entries'][0]['recovery_group_id']='held-group'
        with self.assertRaisesRegex(R.Error,'producer membership'):R.audio_index(self.write('held.json',index),self.root,R.Locator(),inv)
    def test_current_actual_receipt_inventory_when_supplied(self):
        path=os.environ.get('CHIMERA_READINESS_REQUEST')
        if not path:self.skipTest('actual immutable request not supplied')
        result=R.inspect(Path(path));index=result['audio_speech_index']['objectives']
        self.assertEqual(index[R.D.O.AUDIO],{'batches':194,'supervised_units':2833})
        self.assertEqual(index[R.D.O.SPEECH],{'batches':744,'supervised_units':8776960})
        self.assertFalse(result['production_ready']);self.assertFalse(result['training_launched'])
    def test_selected_policy_rejects_wrong_producing_gain_and_missing_owner_family(self):
        from decimal import Decimal
        policy={'constants':R.SELECTED_POLICY}
        rates={k:{k+'.actual-leaf':str(Decimal(v)/Decimal('0.3'))}
               for k,v in R.SELECTED_POLICY['expected_d4_rates'].items()}
        self.assertIs(R.verify_derived_rates(rates,policy),rates)
        wrong=copy.deepcopy(rates);wrong['D2.gain']['D2.gain.actual-leaf']='0.0001'
        with self.assertRaisesRegex(R.Error,'producing base rate'):R.verify_derived_rates(wrong,policy)
        wrong.pop('D3.music_head')
        with self.assertRaisesRegex(R.Error,'omit'):R.verify_derived_rates(wrong,policy)
    def test_selected_policy_rejects_repeated_counts_outer_weights_and_other_source(self):
        mapping={'geometry':'hy3d','h3':'h3','image_edit':'image_edit','audio_ce':'audio_ce','speech_flow':'speech_flow'}
        refs={role:{'sha256':str(i+1)*64} for i,role in enumerate(mapping)}
        rows=[{'lane':lane,'objective_weight':1.,'source_manifest':refs[role]} for role,lane in mapping.items()]
        replay={'counts':{lane:R.SELECTED_POLICY['exact_batch_counts'][role] for role,lane in mapping.items()},
            'rows':rows,'measured_supervised_units':R.SELECTED_POLICY['recorded_index_units']}
        self.assertTrue(R.verify_replay_policy({'target_manifests':refs},replay))
        for alter in ('count','weight','source'):
            wrong=copy.deepcopy(replay)
            if alter=='count':wrong['counts']['h3']+=1
            elif alter=='weight':wrong['rows'][0]['objective_weight']=2.
            else:wrong['rows'][0]['source_manifest']['sha256']='f'*64
            with self.assertRaises(R.Error):R.verify_replay_policy({'target_manifests':refs},wrong)
    def test_actual_d4_preflight_requires_selected_policy_without_constructing_model(self):
        path=self.root/'d4.json';path.write_text(json.dumps({'schema':R.D.SCHEMA,'stage':'D4'}))
        value=R.D.preflight(path)
        self.assertFalse(value['candidate_preflight_passed']);self.assertFalse(value['production_ready'])
        self.assertTrue(any('selected_policy' in b for b in value['blockers']))


if __name__=='__main__':unittest.main()
