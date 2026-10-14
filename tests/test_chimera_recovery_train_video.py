"""Policy refusals plus actual native 144-row tiny full-depth/backward proof.

The tiny parameters/targets are authored fixtures, never relabeled actual
teacher features. Genuine eight-source tensor/teacher joins have separate tests.
"""
import copy
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from experiments.chimera_recovery import train_video as D


class PolicyTests(unittest.TestCase):
    def test_model_cannot_create_from_unpreflighted_or_duck_adapter(self):
        with self.assertRaisesRegex(ValueError,'preflight'):
            D.create_engine(object(),None,None,{},total_steps=1)

    def test_unselected_amendment_or_different_optimizer_refused(self):
        source=Path(__file__).resolve().parents[1]/'experiments/chimera_recovery/TEMPORAL_VIDEO_AMENDMENT_2026-10-13.v1.json'
        if not source.exists():self.skipTest('exact proposed amendment not mounted')
        amendment=json.loads(source.read_text());manifest=Path(amendment['native_binding']['path'])
        if not manifest.exists():self.skipTest('actual native artifact not mounted')
        base={'schema':'recovery_temporal_video_selection.v1','selected':False,
              'amendment':{'path':str(source),'sha256':D.A.file_sha(source)},'native_binding':amendment['native_binding'],
              'optimizer':D.OPTIMIZER,'optimizer_ownership':D.OWNERSHIP}
        with tempfile.TemporaryDirectory() as directory:
            selection=Path(directory)/'test-only-selection.json';selection.write_bytes(D.A.canonical(base))
            with self.assertRaisesRegex(ValueError,'selected'):
                D.selected_policy(manifest,D.A.file_sha(manifest),source,D.A.file_sha(source),selection,D.A.file_sha(selection))
            bad=copy.deepcopy(base);bad['selected']=True;bad['optimizer']['unit_lr']=1.
            selection.write_bytes(D.A.canonical(bad))
            with self.assertRaisesRegex(ValueError,'selected'):
                D.selected_policy(manifest,D.A.file_sha(manifest),source,D.A.file_sha(source),selection,D.A.file_sha(selection))


@unittest.skipUnless(importlib.util.find_spec('torch') and os.environ.get('CHIMERA_D3_BASELINE'),
                     'actual Torch/native source fixture required')
class TorchNativeVideoTests(unittest.TestCase):
    def test_full_depth144_temporal_rows_all_gradient_groups_and_resume(self):
        import torch
        source=os.environ['CHIMERA_D3_BASELINE'];foundation=os.environ.get('CHIMERA_OMNI_SRC')
        if foundation:sys.path.insert(0,str(Path(foundation)/'src'))
        driver_root=Path(os.environ['CHIMERA_VIDEO_DRIVER_DIR'])
        spec=importlib.util.spec_from_file_location('actual_temporal_vision_fixture',driver_root/'train_vision.py')
        V=importlib.util.module_from_spec(spec);sys.modules[spec.name]=V;spec.loader.exec_module(V)
        module=V.T.load_module(Path(source),'actual_hybrid_temporal_video_fixture')
        original=module.tiny_unit_specs
        module.tiny_unit_specs=lambda depth:original(depth)+[dataclasses.replace(original(depth)[-1],
                       layer_index=depth+4,source='qvl32',component='tiny.qvl32',type_ids=(6,))]
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);built=module.build_tiny_child(path/'base',seed=17)
            def construct():
                plan=module.trainer_module().SitePlanConfig(rank=4,alpha=4.,k3_rank=0,trunk_ffn_width=module.TINY.TRUNK,gain=False)
                h=module.HybridD3(built['dir'],built['parent_config'],module.HybridOptions(dtype=torch.float32,device='cpu',
                        kernels='torch',lora=plan,lora_seed=3,pin_boundaries=False,prefetch=False))
                optimizer={**D.OPTIMIZER,'warmup_steps':0,'min_lr_ratio':1.}
                return V.VisionEngine(module,h,'tiny_qvl32',optimizer,{'fixture':True,'stage':'D3d_video'},total_steps=3)
            engine=construct()
            segment=module.tiny_segment(engine.hybrid.cfg,'ocr',prompt=6,rows=144,seed=8,suffix=4)
            tyx=torch.cartesian_prod(torch.arange(4),torch.arange(6),torch.arange(6))
            segment['positions'][0,:,0]=(tyx[:,0].float()*2+.5)/8
            segment['positions'][0,:,1:3]=tyx[:,1:3]
            segment['positions'][0,:,3]=0
            length=segment['ids'].shape[1];positions=torch.arange(length-4,length-1)
            lp=torch.log_softmax(torch.randn(3,engine.hybrid.base_vocab,generator=torch.Generator().manual_seed(12)),-1)
            qlp,qids=lp.topk(5,-1)
            targets=module.answer_kd_targets(length,positions,torch.tensor([2,3,4]),qids,qlp)
            teacher=torch.randn(1,144,engine.placement.spec.width,generator=torch.Generator().manual_seed(13))
            captures=[];h=engine.hybrid;context=h.prepare(segment,media_grad=True);x=context['x'];contexts=h.contexts(x)
            with V.capture_unit_block(h.units[engine.placement.container],captures):
                for i in range(h.num_layers):
                    layer=copy.deepcopy(h.engine.layer(i,i+1 if i+1<h.num_layers else None));h.bank.bind(layer,i)
                    x=h.engine._run(layer,i,x,contexts);x=h.apply_units(x,i,context)
            text=module.d3d_head_loss(h,targets,denominator=3.)(h.final_norm(x),context)
            eager=text+.1*(captures[0].float()-teacher).square().mean();eager.backward()
            expected={name:t.grad.clone() for name,t in engine.census.items()};engine.optimizer.zero_grad(set_to_none=True)
            result=V.feature_forward_backward(module,h,segment,module.d3d_head_loss(h,targets,denominator=3.),
                        engine.placement.container,teacher)
            self.assertAlmostEqual(result['loss'],float(eager.detach()),places=5)
            for name,t in engine.census.items():torch.testing.assert_close(t.grad,expected[name],rtol=2e-4,atol=2e-5)
            for group in ('unit.','bridge.','lora.'):
                self.assertTrue(any(t.grad.norm()>0 for name,t in engine.census.items() if name.startswith(group)))
            first=engine.train_step(segment,targets,teacher,answer_positions=3)
            self.assertFalse(first['skipped']);self.assertEqual(first['feature_captures'],1)
            engine.save(path/'checkpoint');expected_step=engine.train_step(segment,targets,teacher,answer_positions=3)
            resumed=construct();resumed.resume(path/'checkpoint')
            actual=resumed.train_step(segment,targets,teacher,answer_positions=3)
            self.assertEqual(actual,expected_step)
            self.assertTrue(all(torch.equal(t,resumed.census[name]) for name,t in engine.census.items()))


if __name__=='__main__':unittest.main()
