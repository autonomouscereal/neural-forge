"""Structural/source guards only; unit data are never emitted as teacher targets."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.chimera_recovery import video_teacher as T


class SourcePreflightTests(unittest.TestCase):
    def write(self,root,name,value,sealed=False):
        p=root/name
        if sealed:
            value={**value,'sealed':True};value['manifest_sha256']=T.A.digest(value)
        p.write_bytes(T.A.canonical(value)+b'\n');return p

    def test_canonical_manifest_cannot_be_replaced_by_true_flags(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.write(root,'a.json',{'schema':'unit','sealed':True,'manifest_sha256':'0'*64})
            with self.assertRaisesRegex(ValueError,'canonical'):T.json_pin(p,T.A.file_sha(p),'unit')
            with self.assertRaises(ValueError):T.json_pin(p,'0'*64,'unit')

    def fixture(self,root):
        records=[];receipts=[];decisions=[]
        for i in range(24):
            split=('train','validation','test')[i//8];source={'id':str(i),'route':'video_understand','split':split}
            records.append(source);sha=T.A.digest(source)
            receipts.append({'id':str(i),'source_row_sha256':sha,'asset_sha256':str(i%10)*64,'split':split,
                'all_eight_frames_retained':True,'parent_prefix_full_ids_reproduced':True,
                'metadata':{'frames_indices':list(range(8))},'grid_thw':[4,6,6]})
            decisions.append({'id':str(i),'source_row_sha256':sha,'asset_sha256':str(i%10)*64,'split':split,
                'train_admitted':split=='train','blockers':[] if split=='train' else ['held'], 'recovery_group_id':split})
        def jsonl(name,values):
            p=root/name;p.write_bytes(b''.join(T.A.canonical(r)+b'\n' for r in values));return p
        r=jsonl('receipts.jsonl',receipts);source=jsonl('source.jsonl',records);a=jsonl('admission.jsonl',decisions)
        ingress=self.write(root,'ingress.json',{'schema':T.V.SCHEMA,'rows':24,'all_frames_retained':True,
            'model_weights_loaded':False,'teacher_targets_generated':False,'outputs':{'receipts':T.V.pin(r)}},True)
        admission=self.write(root,'global.json',{'schema':'recovery_audio_admission.v1','complete_source_traversal':True,
            'frozen_exclusions_verified':True,'connected_groups_verified':True,
            'sources':{'source':T.V.pin(source)},'outputs':{'admission':T.V.pin(a)}},True)
        return ingress,r,admission,decisions

    def test_join_keeps_held_rows_and_exact_source_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);i,r,a,_=self.fixture(root)
            result=T.join_global(ingress=i,ingress_sha256=T.A.file_sha(i),receipts=r,
                admission=a,admission_sha256=T.A.file_sha(a),output=root/'join.json')
            self.assertEqual((result['source_train_eligible'],result['held']),(8,16))
            self.assertFalse(result['native_train_ready'])
            self.assertEqual([x['split'] for x in result['records']],['train']*8+['validation']*8+['test']*8)

    def test_sealed_join_refuses_changed_asset_or_lifted_held(self):
        for defect in ('asset','held','source'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as d:
                root=Path(d);i,r,a,decisions=self.fixture(root)
                if defect=='asset':decisions[0]['asset_sha256']='f'*64
                elif defect=='held':decisions[8].update(train_admitted=True,blockers=[])
                else:decisions[0]['source_row_sha256']='a'*64
                p=root/'admission.jsonl';p.write_bytes(b''.join(T.A.canonical(x)+b'\n' for x in decisions))
                value=json.loads(a.read_text());value.pop('manifest_sha256');value['outputs']['admission']=T.V.pin(p)
                self.write(root,'global.json',value,True)
                with self.assertRaisesRegex(ValueError,'same video'):
                    T.join_global(ingress=i,ingress_sha256=T.A.file_sha(i),receipts=r,
                        admission=a,admission_sha256=T.A.file_sha(a),output=root/'join.json')

    def test_receipt_hash_or_duplicate_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);i,r,a,_=self.fixture(root);r.write_bytes(r.read_bytes()+r.read_bytes().splitlines()[0]+b'\n')
            with self.assertRaisesRegex(ValueError,'complete CPU'):T.load_inputs(i,T.A.file_sha(i),r)

    def test_mirror_is_explicit_byte_checked_not_a_hash_waiver(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'a.npz';p.write_bytes(b'unit source bytes')
            spec={'path':'/original/output/shards/a.npz','sha256':T.A.file_sha(p)}
            self.assertEqual(T.shard_path(spec,root),p.resolve())
            p.write_bytes(b'tampered')
            with self.assertRaises(ValueError):T.shard_path(spec,root)

    def test_source_catalog_only_accepts_actual_architecture_and_visual_namespace(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=root/'config.json';idx=root/'model.safetensors.index.json'
            cfg={'vision_config':dict(zip(('model_type','depth','hidden_size','num_heads','patch_size',
                'temporal_patch_size','spatial_merge_size','out_hidden_size'),('qwen3_5_vision',27,1152,16,16,2,2,5120)))}
            c.write_text(json.dumps(cfg));idx.write_text(json.dumps({'weight_map':{
                'model.visual.unit_'+str(i):'model-00012-of-00012.safetensors' for i in range(333)}}))
            with patch.object(T,'CONFIG_SHA',T.A.file_sha(c)),patch.object(T,'INDEX_SHA',T.A.file_sha(idx)):
                self.assertEqual(len(T.source_catalog(root)),333)
                idx.write_text(idx.read_text().replace('model.visual.unit_0','model.language_model.unit_0'))
                with self.assertRaisesRegex(ValueError,'config/index'):T.source_catalog(root)
            cfg['vision_config']['temporal_patch_size']=1;c.write_text(json.dumps(cfg))
            with patch.object(T,'CONFIG_SHA',T.A.file_sha(c)),patch.object(T,'INDEX_SHA',T.A.file_sha(idx)):
                with self.assertRaisesRegex(ValueError,'architecture'):T.source_catalog(root)


if __name__=='__main__':unittest.main()
