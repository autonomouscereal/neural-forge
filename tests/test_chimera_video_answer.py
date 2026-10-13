"""Unit-only conversion guards; no model/teacher evidence is invented here."""
import unittest
import os
import tempfile
from pathlib import Path
import numpy as np
from experiments.chimera_recovery import video_answer as B


class RawAnswerConversionTests(unittest.TestCase):
    def test_singleton_argv_keeps_full_sequence_and_refuses_relative_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);plan={'teacher':'unit-only','sources':[{'split':'train','path':'train.jsonl'}]}
            argv=B.writer_argv(plan,base/'manifest.json','train',base/'model',base/'out',base/'allocator.json')
            self.assertEqual(argv[argv.index('--micro-batch')+1],'1')
            self.assertEqual(argv[argv.index('--batch-order')+1],'corpus')
            self.assertEqual(argv[argv.index('--rows-per-shard')+1],'8')
            self.assertEqual(argv[argv.index('--head-precision')+1],'fp32')
            self.assertEqual(argv[argv.index('--top-k')+1],'64')
            self.assertNotIn('--max-length',argv);self.assertNotIn('--truncate',argv)
            for path in ('../outside.jsonl',str(base.parent/'outside.jsonl')):
                with self.assertRaisesRegex(ValueError,'leaves'):B.contained(base,path)

    def args(self):
        ids=np.array([[10,11,12,13,14,15]],np.int64);mask=np.array([0,0,0,1,1,1],np.bool_)
        return [ids,mask,np.array([2,3,4],np.uint32),np.array([12,13,14],np.uint32),
            np.tile(np.arange(64,dtype=np.int32),(3,1)),np.full((3,64),0xc100,np.uint16)]  # BF16 -8, unit data only

    def test_raw_bf16_bits_and_assistant_next_ids_are_preserved_exactly(self):
        result=B.dense_answer(*self.args())
        self.assertEqual(result['q_w'].tolist(),[0,0,1,1,1,0])
        self.assertEqual(result['next_ids'].tolist(),[0,0,13,14,15,0])
        np.testing.assert_array_equal(result['q_lp'][2:5],np.full((3,64),-8,np.float32))
        self.assertFalse(result['ce_w'].any());self.assertFalse(result['eos_w'].any())

    def test_unshifted_or_duplicate_positions_and_next_ids_in_raw_input_field_fail(self):
        for change in ('unshifted','duplicate','nextid','missing'):
            with self.subTest(change=change):
                a=self.args()
                if change=='unshifted':a[2]+=1
                elif change=='duplicate':a[2][1]=a[2][0]
                elif change=='nextid':a[3]+=1
                else:a[2]=a[2][:-1]
                with self.assertRaisesRegex(ValueError,'once-shifted'):B.dense_answer(*a)

    def test_invalid_true_logits_or_foreign_vocab_cannot_become_ready(self):
        for change in ('positive','nan','duplicate','foreign','excessmass'):
            with self.subTest(change=change):
                a=self.args()
                if change=='positive':a[5][0,0]=0x3f80
                elif change=='nan':a[5][0,0]=0x7fc0
                elif change=='duplicate':a[4][0,1]=a[4][0,0]
                elif change=='foreign':a[4][0,0]=B.V.TEXT_VOCAB
                else:a[5][:]=0
                with self.assertRaisesRegex(ValueError,'distributions'):B.dense_answer(*a)

    def integrity(self):
        files=[{'path':'unit.safetensors','bytes':4,'sha256':'a'*64}];tree=B.A.digest(files)
        return {'teacher':{'digest_mode':'full','equal_before_and_after_every_session':True,
            'tree_sha256_before':tree,'tree_sha256_after':tree,'files':files},
            'sessions':[{'unchanged':True,'before':{'mode':'full','tree_sha256':tree},'after':{'mode':'full','tree_sha256':tree}}]}

    def test_after_file_list_is_derived_from_actual_full_tree_not_a_true_flag(self):
        r=B.full_integrity(self.integrity())
        self.assertEqual(r['before'],r['after']);self.assertIn('canonical',r['derivation'])
        for defect in ('stat','moved','missing','tree'):
            with self.subTest(defect=defect):
                r=self.integrity()
                if defect=='stat':r['sessions'][0]['after']['mode']='stat'
                elif defect=='moved':r['sessions'][0]['after']['tree_sha256']='f'*64
                elif defect=='missing':r['sessions']=[]
                else:r['teacher']['files'][0]['sha256']='b'*64
                with self.assertRaisesRegex(ValueError,'full teacher'):B.full_integrity(r)


class ActualOriginalWriterInteropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=os.environ.get('CHIMERA_QTEXT_WRITER','')
        if not source:raise unittest.SkipTest('actual pinned original writer must be supplied explicitly')
        try:import torch
        except ImportError:raise unittest.SkipTest('actual Torch unavailable')
        cls.torch=torch;cls.W=B.V.module(Path(source),B.V.WRITER_SHA,'video_answer_actual_writer_test')

    def test_actual_parser_and_five_raw_seals_preserve_bf16_bits(self):
        # Unit arrays exercise the genuine writer API; these are never teacher targets.
        torch=self.torch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan={'teacher':'unit-only','sources':[{'split':'train','path':'train.jsonl'}]}
            args=self.W.parser().parse_args(B.writer_argv(plan,root/'manifest.json','train',root/'model',root/'out',root/'allocator.json'))
            self.assertEqual((args.micro_batch,args.batch_order,args.rows_per_shard),(1,'corpus',8))
            a=RawAnswerConversionTests().args();ids,mask,pos,input_ids,topk,raw=a
            arrays={'ids':torch.from_numpy(input_ids.astype(np.int64)),'pos':torch.from_numpy(pos.astype(np.int64)),
                'topk_ids':torch.from_numpy(topk.astype(np.int64)),'topk_logprobs':torch.full((3,64),-8,dtype=torch.bfloat16),
                'lse':torch.full((3,),1.5,dtype=torch.float32)}
            self.W.write_shard(root,0,arrays,{},B.V.TEXT_VOCAB,64)
            shard=root/self.W.shard_name(0);self.assertTrue(self.W.verify_shard(shard)['ok'])
            answer=B.dense_answer(ids,mask,pos,input_ids,topk,np.fromfile(shard/'topk_logprobs.bf16',dtype='<u2').reshape(3,64))
            np.testing.assert_array_equal(answer['q_lp'][2:5],np.full((3,64),-8,np.float32))
            p=shard/'lse.f32';payload=bytearray(p.read_bytes());payload[0]^=1;p.write_bytes(payload)
            self.assertFalse(self.W.verify_shard(shard)['ok'])


if __name__=='__main__':unittest.main()
