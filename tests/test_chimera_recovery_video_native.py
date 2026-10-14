import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from experiments.chimera_recovery import video_native as N
from experiments.chimera_recovery import video_quantized_answer as Q


def small():
    ids=np.array([[10,11,12,13]],np.int64)
    mask=np.array([False,False,True,True])
    positions=np.array([1,2],np.uint32)
    cached=np.array([11,12],np.uint32)
    top=np.tile(np.arange(64,dtype=np.int32),(2,1))
    lp=np.full((2,64),-8,np.float32)
    bits=(lp.view(np.uint32)>>16).astype(np.uint16)
    return ids,mask,positions,cached,top,bits,np.full(2,20,np.float32)


class QuantizedAnswerTests(unittest.TestCase):
    def test_transport_alias_requires_same_bytes_and_original_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'actual.json';path.write_bytes(b'{"actual":true}\n')
            original='C:\\Users\\cereal\\Documents\\ai training\\fixture.json'
            spec={'path':str(path),'sha256':N.A.file_sha(path)}
            with N.source_aliases({original:spec}):
                self.assertEqual(N.pinned({'path':original,'sha256':spec['sha256']},'transport'),path)
                with self.assertRaisesRegex(ValueError,'seal'):N.pinned({'path':original,'sha256':'0'*64},'bad')
                with self.assertRaisesRegex(ValueError,'nested'):
                    with N.source_aliases({original:spec}):pass
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                with N.source_aliases({original:spec}):pass

    def test_once_shift_bits_and_no_normalization(self):
        args=small();answer,proof=Q.dense_answer(*args)
        self.assertEqual(answer['q_w'].tolist(),[0,1,1,0])
        self.assertEqual(answer['next_ids'].tolist(),[0,12,13,0])
        self.assertTrue(np.array_equal(answer['q_lp'][1:3].view(np.uint32)>>16,args[5]))
        self.assertFalse(proof['target_probabilities_rescaled'])
        self.assertFalse(proof['exact_normalization_proven'])

    def test_source_mask_ids_and_call_targets_rejected(self):
        for field in (1,2,3,4,5,6):
            with self.subTest(field=field):
                a=list(small());a[field]=a[field].copy()
                if field==1:a[field][1]=True
                elif field==2:a[field][0]=0
                elif field==3:a[field][0]=99
                elif field==4:a[field][0,1]=a[field][0,0]
                elif field==5:a[field][:]=0
                else:a[field][0]=np.nan
                with self.assertRaises(ValueError):Q.dense_answer(*a)
        a=list(small());a[4][0,0]=N.V.TEXT_VOCAB
        with self.assertRaises(ValueError):Q.dense_answer(*a)

    def test_bf16_representability_and_impossible_mass(self):
        lp=np.full((1,64),-8,np.float32);lp[0,0]=-8.00001
        with self.assertRaisesRegex(ValueError,'representability'):Q.rounding_compatible(lp,np.array([20],np.float32))
        with self.assertRaisesRegex(ValueError,'impossible'):Q.rounding_compatible(np.zeros((1,64),np.float32),np.array([20],np.float32))
        with self.assertRaises(ValueError):Q.rounding_compatible(np.full((1,64),-8,np.float32),np.array([np.inf],np.float32))

    def test_rebase_rejects_source_target_and_mapping_changes(self):
        args=small();answer,_=Q.dense_answer(*args)
        ingress={'parent_input_ids':args[0],'parent_assistant_token_mask':args[1],
                 'input_ids':args[0],'assistant_token_mask':args[1],
                 'parent_to_native_token_index':np.arange(4),
                 'assistant_prediction_mask':np.array([False,True,True,False])}
        result=Q.rebase_answer(answer,ingress,args[-1]);self.assertEqual(result['q_w'].sum(),2)
        altered=copy.deepcopy(answer);altered['next_ids'][1]=99
        with self.assertRaises(ValueError):Q.rebase_answer(altered,ingress,args[-1])
        altered=copy.deepcopy(ingress);altered['parent_to_native_token_index'][2]=1
        with self.assertRaises(ValueError):Q.rebase_answer(answer,altered,args[-1])

    def test_actual_execution_binds_request_entry_code_and_completion(self):
        request={'sources':{'video_capture_job.py':{'sha256':'a'*64}}}
        value={'schema':'recovery_video_teacher_execution.v1','job':'capture','exit_code':0,
               'failure':None,'request':{'sha256':'b'*64},'code_sha256':'a'*64,'training_started':False}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'execution.json';p.write_text(json.dumps(value))
            N.execution_binding(p,'b'*64,request,'capture')
            for key,new in [('exit_code',1),('code_sha256','c'*64),('job','control'),('training_started',True)]:
                with self.subTest(key=key):
                    changed={**value,key:new};p.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):N.execution_binding(p,'b'*64,request,'capture')


@unittest.skipUnless(os.environ.get('CHIMERA_VIDEO_ACTUAL_CAPTURE'),'actual sealed capture path not provided')
class ActualCaptureTests(unittest.TestCase):
    def test_all792_rounding_compatible_and_altered_lse_rejected(self):
        root=Path(os.environ['CHIMERA_VIDEO_ACTUAL_CAPTURE']);manifest=next(root.glob('cache/*/*/manifest.json'))
        r=json.loads(manifest.read_text());entry=r['shards'][0];base=manifest.parent/entry['shard']
        bits=np.fromfile(base/'topk_logprobs.bf16',dtype='<u2').reshape(792,64)
        lse=np.fromfile(base/'lse.f32',dtype='<f4');lp=(bits.astype(np.uint32)<<16).view(np.float32)
        for name,spec in entry['files'].items():self.assertEqual(N.A.file_sha(base/name),spec['sha256'])
        proof=Q.rounding_compatible(lp,lse)
        self.assertEqual(proof['stored_mass_over_one_rows'],191)
        self.assertEqual(proof['stored_mass_over_1_001_rows'],20)
        self.assertAlmostEqual(proof['stored_mass_max'],1.0033815256367713,places=12)
        self.assertTrue(np.array_equal(lp.view(np.uint32)>>16,bits))
        with self.assertRaisesRegex(ValueError,'impossible'):Q.rounding_compatible(lp,np.zeros_like(lse))


@unittest.skipUnless(os.environ.get('CHIMERA_VIDEO_NATIVE_MANIFEST'),'actual native conversion path not provided')
class ActualReceiverTests(unittest.TestCase):
    def test_all8_receiver_and_target_pixel_mask_tamper(self):
        from safetensors.numpy import load_file
        p=Path(os.environ['CHIMERA_VIDEO_NATIVE_MANIFEST']);manifest=json.loads(p.read_text())
        total=0
        for entry in manifest['shards']:
            batch=load_file(entry['path']);proof=N.validate_native_row(batch=batch,entry=entry,manifest=manifest,manifest_path=p)
            total+=proof['assistant_positions']
        self.assertEqual(total,792)
        entry=manifest['shards'][0];original=load_file(entry['path'])
        for key in ('input_ids','native_media_features','teacher_features','answer.q_lp','assistant_token_mask'):
            with self.subTest(key=key):
                changed={k:v.copy() for k,v in original.items()}
                changed[key].flat[0]=not changed[key].flat[0] if changed[key].dtype==np.bool_ else changed[key].flat[0]+1
                with self.assertRaisesRegex(ValueError,'differ'):N.validate_native_row(batch=changed,entry=entry,manifest=manifest,manifest_path=p)
        changed={**entry,'source_row_sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'identity'):N.validate_native_row(batch=original,entry=changed,manifest=manifest,manifest_path=p)


if __name__=='__main__':unittest.main()
