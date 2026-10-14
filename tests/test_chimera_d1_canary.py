"""Private-canary safety/actual source capsule regressions; no CUDA/model runs."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.chimera_recovery import d1_canary as C


class CanaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        (self.root/'canary_profile.json').write_bytes(C.R.canonical(C.PROFILE)+b'\n')
        self.candidate={'actual_counts':{'total_rows':979,'width':4096,'held_rows':62},
            'files':{'canary_profile.json':C.R.sha(self.root/'canary_profile.json'),'tools/d1_canary.py':C.R.sha(C.__file__)}}
        self.candidate['canary_profile']={'path':'canary_profile.json','sha256':self.candidate['files']['canary_profile.json']}
    def test_original_source_selector_preserved_and_row_subsample_refused(self):
        original=['--ids','complete.safetensors','--held-rows','62','--layers','all','--epochs','2']
        after=C.arguments(original,self.candidate,self.root)
        self.assertEqual(after,['--ids','complete.safetensors','--held-rows','62','--layers','0,3','--epochs','2',
            '--max-steps','1','--stop-after-layer','3'])
        self.assertEqual(original[5],'all')
        for flag in ('--max-rows','--max-steps','--stop-after-layer'):
            with self.assertRaisesRegex(C.R.Error,'unexpected'):C.arguments(original+[flag,'1'],self.candidate,self.root)
        wrong=copy.deepcopy(self.candidate);wrong['actual_counts']['held_rows']=0
        with self.assertRaisesRegex(C.R.Error,'complete'):C.arguments(original,wrong,self.root)
    def test_resealed_broader_profile_cannot_expand_optimizer_window(self):
        changed=dict(C.PROFILE,max_steps=2)
        (self.root/'canary_profile.json').write_bytes(C.R.canonical(changed)+b'\n')
        self.candidate['files']['canary_profile.json']=C.R.sha(self.root/'canary_profile.json')
        self.candidate['canary_profile']['sha256']=self.candidate['files']['canary_profile.json']
        with self.assertRaisesRegex(C.R.Error,'exact selected'):C.overrides(self.candidate,self.root)
    def test_genuine_clone_config_proof_and_wrong_parent_first_gqa_refuse(self):
        path=os.environ.get('CHIMERA_D1_CANARY_CLONE_CONFIG')
        if not path:self.skipTest('actual received clone config not supplied')
        raw=Path(path).read_bytes();self.assertEqual(C.R.sha(path),C.CONFIG_SHA)
        childroot=self.root/'child';childroot.mkdir();(childroot/'config.json').write_bytes(raw)
        kinds=json.loads(raw)['config']['block_pattern'];indices=[i for i,k in enumerate(kinds) if k!='charted']
        child=SimpleNamespace(root=childroot,text_blocks=indices,cfg=SimpleNamespace(block_kind=lambda i:kinds[i]))
        C.actual_geometry(self.candidate,child,child,self.root)
        wrong=SimpleNamespace(text_blocks=[0,1,2,3],cfg=SimpleNamespace(block_kind=lambda i:'gated_gqa' if i==1 else 'kda'))
        with self.assertRaisesRegex(C.R.Error,'boundaries differ'):C.actual_geometry(self.candidate,child,wrong,self.root)
    def test_any_existing_canary_delta_or_before_receipt_blocks_resume(self):
        run=self.root/'run';delta=self.root/'delta';run.mkdir();delta.mkdir()
        C.fresh_outputs(run,delta)
        (delta/'trainer').mkdir();(delta/'trainer'/'private_delta').write_bytes(b'previous attempt')
        with self.assertRaisesRegex(C.R.Error,'cannot resume'):C.fresh_outputs(run,delta)
    def test_actual_selected_v8_derivative_keeps_optimizer_bytes_and_source_seals(self):
        path=os.environ.get('CHIMERA_D1_PRODUCTION_V8')
        if not path:self.skipTest('actual selected v8 capsule not supplied')
        original=Path(path);out=self.root/'candidate'
        C.prepare(original,out,'/home/cereal/candidate','/home/cereal/lease','/home/cereal/run','/mnt/lexar/delta')
        parent=C.R.read(original/'candidate_receipt.json',C.V8_SHA);child=C.R.read(out/'candidate_receipt.json',C.R.sha(out/'candidate_receipt.json'))
        self.assertEqual(child['successor'],parent['successor']);self.assertEqual(child['actual_counts'],parent['actual_counts'])
        self.assertEqual(child['files']['tools/d1_trunk_recovery.py'],parent['files']['tools/d1_trunk_recovery.py'])
        self.assertEqual((out/'tools/d1_trunk_recovery.py').read_bytes(),(original/'tools/d1_trunk_recovery.py').read_bytes())
        self.assertEqual(child['files']['tools/d1_source_contract.py'],parent['files']['tools/d1_source_contract.py'])
        self.assertEqual(child['files']['resource_policy.json'],parent['files']['resource_policy.json'])
        self.assertEqual(C.profile(child,out),C.PROFILE)
        self.assertTrue(child['canary_only']);self.assertFalse(child['production_ready'])
        plan=C.R.read(out/'allocation_plan.json',C.R.sha(out/'allocation_plan.json'))
        self.assertEqual(plan['root_controller_sha256'],child['files']['tools/d1_allocated_root.py'])
        self.assertIn(b"'/candidate/tools/d1_canary.py','watch'",(out/'tools/d1_allocated_root.py').read_bytes())
        for item in plan['mounts']:
            if item['role'] in {'child','parent','successor','chimera'}:self.assertTrue(item['read_only'])


if __name__=='__main__':unittest.main()
