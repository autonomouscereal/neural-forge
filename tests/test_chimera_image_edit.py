"""Editing ownership/conditioning tests; fixtures do not assert parent parity."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/chimera_recovery'))
spec=importlib.util.spec_from_file_location('test_native_image_edit',ROOT/'experiments/chimera_recovery/image_edit.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
try:import torch
except ImportError:torch=None

def record():
    return {'schema':'recovery_corpus.v1','id':'editing-contract-fixture','route':'image_edit','split':'train',
        'provenance':{'source_year':2026,'split':'train','group_id':'editing-train'},
        'media':[{'type':'image','role':'input','source_verified':True,'sha256':'a'*64},
                 {'type':'image','role':'target','source_verified':True,'sha256':'b'*64}],
        'supervision':{'message_mask':[False,True]},'messages':[{'role':'user','content':'Change red to green.'},
            {'role':'assistant','content':'target metadata never becomes an instruction'}],'tools':[]}

class SourceTests(unittest.TestCase):
    def test_source_only_prompt_and_reference_after_roles_are_fixed(self):
        source=record();self.assertEqual(E.source_messages(source),source['messages'][:1])
        changed=copy.deepcopy(source);changed['messages'][-1]['content']='different target metadata'
        self.assertEqual(E.source_messages(source),E.source_messages(changed))
        for bad in [copy.deepcopy(source) for _ in range(3)]:
            bad['media'][1]['role']='input'
            with self.assertRaises(E.Error):E.source_messages(bad)
        bad=copy.deepcopy(source);bad['media'][0]['sha256']=bad['media'][1]['sha256']
        with self.assertRaises(E.Error):E.source_messages(bad)

    def test_seed_preserves_declared_positive_sigma_and_source_identity(self):
        source=record();self.assertEqual(E.noise_seed(source,.3),E.noise_seed(source,.3))
        self.assertNotEqual(E.noise_seed(source,.3),E.noise_seed(source,.7))
        for sigma in [0.,True,float('nan')]:
            with self.assertRaises(E.Error):E.noise_seed(source,sigma)

@unittest.skipIf(torch is None,'actual Torch runtime required')
class TensorTests(unittest.TestCase):
    def setUp(self):
        self.record=record();self.prefix={'messages':self.record['messages'][:1],'parent_text_ids':[1,2],
            'input_ids':[1,2]+[E.M.MEDIA_PLACEHOLDER]*2048,'prompt_length':2,
            'source_row_sha256':E.R.digest(self.record),'policy':E.POLICY,'assistant_metadata_in_prefix':False}
        self.before=torch.randn(1,1024,64,generator=torch.Generator().manual_seed(3))
        self.after=torch.randn(1,1024,64,generator=torch.Generator().manual_seed(5))

    def test_target_free_sigma1_input_reference_order_and_zero_reference_loss(self):
        batch=E.build_batch(self.record,self.prefix,self.before,self.after,sigma=1.)
        other=E.build_batch(self.record,self.prefix,self.before,self.after+20,sigma=1.)
        self.assertTrue(torch.equal(batch['native_media_features'],other['native_media_features']))
        self.assertFalse(torch.equal(batch['native_media_targets'],other['native_media_targets']))
        self.assertTrue(torch.equal(batch['native_media_features'][:,1024:,512:576],self.before))
        self.assertTrue(torch.equal(batch['native_media_noise_levels'][:,1024:],torch.zeros(1,1024)))
        self.assertTrue(torch.equal(batch['native_media_positions'][:,1024:,0],torch.ones(1,1024)))
        prediction=torch.zeros(1,2048,6144,requires_grad=True)
        loss=E.velocity_loss(prediction,batch);loss.backward()
        self.assertGreater(float(prediction.grad[:,:1024,256:320].abs().sum()),0)
        self.assertEqual(float(prediction.grad[:,1024:].abs().sum()),0)
        bad=copy.deepcopy(batch);bad['native_media_loss_mask'][:,1024:]=1
        with self.assertRaises(E.Error):E.velocity_loss(prediction,bad)

    def test_compact_storage_keeps_clean_reference_and_noisy_output_exact(self):
        batch=E.build_batch(self.record,self.prefix,self.before,self.after,sigma=.3)
        stored,storage=E.M.compact_native_batch(batch,E.BANDS);actual=E.M.inflate_native_batch(stored,storage)
        self.assertEqual(set(actual),set(batch))
        self.assertTrue(all(torch.equal(actual[k],v) for k,v in batch.items()))
        self.assertLess(sum(v.numel() for v in stored.values()),sum(v.numel() for v in batch.values())//20)

if __name__=='__main__':unittest.main()
