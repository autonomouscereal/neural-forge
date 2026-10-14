"""Temporal-video receiver and distinct native full-depth training adapter.

The tensor artifact is not itself an optimizer approval. An exact selected
dated amendment and all-row CPU preflight are required before model loading.
Existing VisionEngine computes unchanged text bucket_terms plus feature MSE.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

try:
    from . import video_native as N
except ImportError:
    import video_native as N
A=N.A
OBJECTIVE='temporal_video_answer_bucket_kd_plus_qwen38_vit13_feature_mse'
OPTIMIZER={'unit_lr':2e-5,'bridge_lr':2e-5,'lora_lr':1e-4,'betas':[.9,.95],
           'eps':1e-8,'weight_decay':0.,'clip':1.,'warmup_steps':20,'min_lr_ratio':.1}
OWNERSHIP=['qvl32_vit_L13','type6_bridge','shared_r16_lora']


def selected_policy(manifest_path,manifest_sha,amendment_path,amendment_sha,selection_path,selection_sha):
    manifest=N.T.json_pin(manifest_path,manifest_sha,'chimera_recovery.temporal_video_native_binding.v1')
    amendment=N.json_pin(amendment_path,amendment_sha)
    selection=N.json_pin(selection_path,selection_sha)
    if (amendment.get('schema')!='recovery_temporal_video_amendment.v1' or amendment.get('source_year')!=2026 or
            amendment.get('stage')!='D3d_video' or amendment.get('objective')!=OBJECTIVE or
            amendment.get('native_binding',{}).get('sha256')!=manifest_sha or
            amendment.get('optimizer')!=OPTIMIZER or amendment.get('optimizer_ownership')!=OWNERSHIP or
            amendment.get('feature_weight')!=.1 or amendment.get('text_weight')!=1. or
            amendment.get('rounding_policy')!=N.Q.POLICY or amendment.get('bucket_terms_modified') is not False or
            amendment.get('target_normalization')!='preserve original BF16 bits; no rescaling' or
            amendment.get('parent_xyz_or_full_trunk_parity_claimed') is not False or
            amendment.get('scope',{}).get('train_rows')!=8 or amendment['scope'].get('held_rows')!=16 or
            selection.get('schema')!='recovery_temporal_video_selection.v1' or selection.get('selected') is not True or
            selection.get('amendment',{}).get('sha256')!=amendment_sha or
            selection.get('native_binding',{}).get('sha256')!=manifest_sha or
            selection.get('optimizer')!=OPTIMIZER or selection.get('optimizer_ownership')!=OWNERSHIP):
        raise ValueError('exact selected temporal-video objective/source/optimizer amendment required')
    for name,module in [('video_native.py',N),('video_quantized_answer.py',N.Q)]:
        if amendment['sources'][name]['sha256']!=A.file_sha(Path(module.__file__)):
            raise ValueError('selected temporal-video receiver source differs')
    if amendment['sources']['train_video.py']['sha256']!=A.file_sha(Path(__file__)):
        raise ValueError('selected temporal-video driver source differs')
    return manifest,amendment,selection


class VideoAdapter:
    def __init__(self,*,manifest_path,manifest_sha256,amendment_path,amendment_sha256,selection_path,selection_sha256):
        self.manifest_path=Path(manifest_path)
        self.manifest_sha256=manifest_sha256
        self.amendment_sha256=amendment_sha256
        self.manifest,self.amendment,self.selection=selected_policy(self.manifest_path,manifest_sha256,
            Path(amendment_path),amendment_sha256,Path(selection_path),selection_sha256)
        self.entries={row['source_id']:row for row in self.manifest['shards']}
        if len(self.entries)!=8 or self.manifest.get('heldout_groups_excluded') is not True:
            raise ValueError('complete eight-row globally admitted video traversal required')
        self.preflight_passed=False

    def preflight(self):
        """Authenticate all immutable tensors and actual controls before a model."""
        total=0
        for entry in self.entries.values():
            batch=N.safetensors_numpy(A.pinned(entry,'native temporal video tensor'))
            total+=N.validate_native_row(batch=batch,entry=entry,manifest=self.manifest,
                                        manifest_path=self.manifest_path)['assistant_positions']
        if total!=792:raise ValueError('all792 actual assistant targets required')
        self.preflight_passed=True
        return {'rows':8,'assistant_positions':total,'verified':True,'model_loaded':False}

    def bind(self,batch,source_id,driver):
        if not self.preflight_passed:raise ValueError('complete pre-model temporal-video preflight required')
        if source_id not in self.entries:raise ValueError('unknown/held video source')
        N.validate_native_row(batch=batch,entry=self.entries[source_id],manifest=self.manifest,manifest_path=self.manifest_path)
        import torch
        values={k:(v if isinstance(v,torch.Tensor) else torch.from_numpy(v.copy())) for k,v in batch.items()}
        segment=driver.T.make_segment(values)
        targets={key:values['answer.'+key] for key in ('next_ids','ce_w','eos_w','q_w','q_ids','q_lp')}
        return {'segment':segment,'answer_targets':targets,'teacher_features':values['teacher_features'],
                'answer_positions':self.entries[source_id]['assistant_positions']}


def vision_driver(amendment):
    root=Path(os.environ.get('CHIMERA_VIDEO_DRIVER_DIR',Path(__file__).resolve().parent))
    for name in ('train_vision.py','train_media.py','training.py'):
        expected=amendment['sources'][name]['sha256']
        A.pinned({'path':str(root/name),'sha256':expected},'exact existing full-depth video dependency '+name)
    spec=importlib.util.spec_from_file_location('chimera_recovery_temporal_vision_driver',root/'train_vision.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def create_engine(adapter,module,hybrid,identity,*,total_steps):
    """Called only after preflight; child/base loading remains the native driver."""
    if not isinstance(adapter,VideoAdapter) or not adapter.preflight_passed:
        raise ValueError('temporal-video tensors must preflight before model loading')
    if (identity.get('stage')!='D3d_video' or identity.get('teacher_cache_sha256')!=adapter.manifest_sha256 or
            identity.get('amendment_sha256')!=adapter.amendment_sha256 or identity.get('teacher_feature_block')!=13 or
            identity.get('heldout_groups_excluded') is not True or
            identity.get('hybrid_module_sha256')!=A.file_sha(Path(module.__file__)) or
            adapter.amendment['sources']['d3_hybrid_module.py']['sha256']!=A.file_sha(Path(module.__file__))):
        raise ValueError('native video engine identity/source lineage differs')
    driver=vision_driver(adapter.amendment)
    class TemporalVideoEngine(driver.VisionEngine):
        checkpoint_schema='chimera_recovery.temporal_video_checkpoint.v1'

        def train_video_step(self,batch,source_id):
            bound=adapter.bind(batch,source_id,driver)
            device=self.hybrid.device
            bound['answer_targets']={k:v.to(device) for k,v in bound['answer_targets'].items()}
            bound['teacher_features']=bound['teacher_features'].to(device)
            return self.train_step(**bound)
    return TemporalVideoEngine(module,hybrid,'qvl32_vit_L13',OPTIMIZER,identity,total_steps=total_steps)
