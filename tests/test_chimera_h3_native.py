"""Fail-closed source/clock/parent-layout contracts; no synthetic teacher claim."""
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('test_h3_native_module',ROOT/'experiments/chimera_recovery/h3_native.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch=None

def source(root):
    video,audio,oracle=root/'source.mp4',root/'sound.wav',root/'oracle.json'
    video.write_bytes(b'contract-fixture video bytes');audio.write_bytes(b'contract-fixture audio bytes')
    mux=N.H.sha(video);answer={'count':1,'events':[{'frame':1,'start_seconds':.125,'frequency_hz':400}]}
    oracle.write_text(json.dumps({'groundtruth':{'audio_video_muxed':True,'same_origin_sha256':mux,**answer}}))
    return {'schema':'recovery_corpus.v1','id':'test-av-2026','route':'joint_av','split':'train',
        'provenance':{'source_year':2026,'group_id':'test-2026-train-family','split':'train',
            'groundtruth_path':str(oracle),'groundtruth_sha256':N.H.sha(oracle)},
        'messages':[{'role':'user','content':'Create a flash with a tone.'},{'role':'assistant','content':json.dumps(answer)}],
        'av_alignment':{'same_origin':True},'media':[
            {'type':'video','role':'target','source_verified':True,'source_media_sha256':mux,'start_seconds':0.,
             'path':str(video),'sha256':mux,'probe':{'streams':[
                {'codec_type':'video','avg_frame_rate':'8/1','width':96,'height':96,'duration':'1.000000','start_time':'0.000000'},
                {'codec_type':'audio','duration':'1.000000','start_time':'0.000000'}]}},
            {'type':'audio','role':'target','source_verified':True,'source_media_sha256':mux,'start_seconds':0.,
             'path':str(audio),'sha256':N.H.sha(audio)}]}

class SourceTests(unittest.TestCase):
    def test_raw_utf8_teacher_hash_and_canonical_admission_hash_bind_same_prompt(self):
        import hashlib
        prompt='Source event at 0.125 seconds.';record={'messages':[{'content':prompt}]}
        item={'prompt':prompt+N.SUFFIX,'source_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
        admission={'prompt_sha256':N.R.digest(prompt)}
        self.assertNotEqual(item['source_prompt_sha256'],admission['prompt_sha256'])
        N.validate_conditioning(item,record,admission)
        bad=copy.deepcopy(item);bad['prompt']=prompt+N.SUFFIX+' changed'
        with self.assertRaises(N.Error):N.validate_conditioning(bad,record,admission)
        bad=copy.deepcopy(item);bad['source_prompt_sha256']=admission['prompt_sha256']
        with self.assertRaises(N.Error):N.validate_conditioning(bad,record,admission)

    def test_parent_identity_checks_bytes_without_equating_hash_timings(self):
        before={'parent':{'sha256':'a'*64,'bytes':123,'inode':1,'mtime_ns':2,'hash_seconds':25.8}}
        after=copy.deepcopy(before);after['parent']['hash_seconds']=16.4
        integrity={'mode':'full','unchanged':True,'before':before,'after':after}
        N.immutable_parent_integrity(integrity)
        bad=copy.deepcopy(integrity);bad['after']['parent']['sha256']='b'*64
        with self.assertRaises(N.Error):N.immutable_parent_integrity(bad)

    def test_preserves_full_source_and_target_only_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=source(Path(tmp));before=copy.deepcopy(r);out=N.source_candidate(r)
            self.assertEqual(r,before);self.assertEqual(out['asset_sha256'],r['media'][0]['sha256'])
            for section,key,value in [('media','role','input'),('media','source_media_sha256','a'*64),
                                      ('media','start_seconds',.125)]:
                bad=copy.deepcopy(r);bad[section][1][key]=value
                with self.subTest(key=key),self.assertRaises(N.Error):N.source_candidate(bad)

    def test_rejects_changed_groundtruth_bytes_clock_and_source_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=source(Path(tmp))
            bad=copy.deepcopy(r);bad['messages'][-1]['content']='{"count":0,"events":[]}'
            with self.assertRaises(N.Error):N.source_candidate(bad)
            bad=copy.deepcopy(r);bad['media'][0]['probe']['streams'][0]['avg_frame_rate']='24/1'
            with self.assertRaises(N.Error):N.source_candidate(bad)
            bad=copy.deepcopy(r);bad['provenance']['split']='test'
            with self.assertRaises(N.Error):N.source_candidate(bad)
            Path(r['media'][1]['path']).write_bytes(b'changed actual source')
            with self.assertRaises(N.Error):N.source_candidate(r)

    def test_readonly_alias_requires_exact_original_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'sealed';f.write_bytes(b'actual immutable source bytes');h=N.H.sha(f)
            self.assertEqual(N.pinned({'path':'missing-original','sha256':h},Path(tmp),{h:str(f)}),f)
            f.write_bytes(b'changed')
            with self.assertRaises(N.Error):N.pinned({'path':'missing-original','sha256':h},Path(tmp),{h:str(f)})

    def test_dominance_is_not_receptive_field_and_requires_complete_parent_proof(self):
        groups=[[0],[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17],[18,19,20,21]]
        doc={'schema':'chimera.h3-native-geometry-probe.v1','complete':True,'weights_unchanged':True,
            'shared_group_map_stable':True,'groups_nonempty_contiguous':True,'shared_groups':groups,
            'group_semantics':'dominant per-frame finite-perturbation influence, not exclusive receptive fields',
            'capture_preflight':{'cache_manifest_sha256':'a'*64,'capture_contract_ok':True},
            'video_weight_before':'b'*64,'video_weight_after':'b'*64,'model_state_before':'c'*64,'model_state_after':'c'*64,
            'measurements':[{'id':'source','groups':groups,'base_bitwise_equal':True}],
            'numerical_targets':[{'id':'source','target':0,'native_PackedLayout_bitwise_equal':True,'native_packers_bitwise_equal':True}]}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'proof.json'
            def check(d):
                p.write_text(json.dumps(d));return N.geometry_proof(p,N.H.sha(p),'a'*64,['source'])
            self.assertEqual(check(doc)[1][0]['id'],'source')
            for key,value in [('group_semantics','exact receptive fields'),('video_weight_after','d'*64),
                              ('shared_group_map_stable',False),('measurements',[])]:
                bad=copy.deepcopy(doc);bad[key]=value
                with self.subTest(key=key),self.assertRaises(N.Error):check(bad)
            bad=copy.deepcopy(doc);bad['numerical_targets'][0]['native_PackedLayout_bitwise_equal']=False
            with self.assertRaises(N.Error):check(bad)

@unittest.skipIf(torch is None,'actual Torch runtime is required')
class TensorContractTests(unittest.TestCase):
    def test_runtime_mutated_tensor_or_relabelled_receipt_is_rejected_before_loss(self):
        # Contract fixture only, not teacher evidence. The positive full-source
        # route is separately executed against every genuine captured target.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);types=torch.tensor([[2,3]])
            batch={'native_media_type_ids':types,'native_media_features':torch.zeros(1,2,6144),
                   'native_media_targets':torch.zeros(1,2,6144)}
            stored,storage=N.M.compact_native_batch(batch,N.BANDS)
            save_file(stored,str(root/'input.safetensors'))
            receipt={'cache_sha256':N.H.sha(root/'input.safetensors'),'storage':storage,'source':{'source_id':'fixture'}}
            (root/'receipt.json').write_bytes(N.R.canonical(receipt)+b'\n')
            manifest={'schema':'chimera_recovery.native_batches.v1','sealed':True,
                'adapter_kind':'exact_h3_joint_native_dualclock_guided_velocity','complete_source_traversal':True,
                'heldout_groups_excluded':True,'batches':[{'storage':storage,'tensors':N.pin(root/'input.safetensors'),
                    'teacher_receipt':N.pin(root/'receipt.json')}]}
            (root/'manifest.json').write_text(json.dumps(manifest))
            bad={k:v.clone() for k,v in batch.items()};bad['native_media_features'][0,0,384]=1
            with self.assertRaisesRegex(N.Error,'runtime batch changed'):N.validate_batch(bad,receipt,root)
            changed=copy.deepcopy(receipt);changed['source']['source_id']='other-source'
            with self.assertRaisesRegex(N.Error,'exactly one immutable'):N.validate_batch(batch,changed,root)

if __name__=='__main__':unittest.main()
