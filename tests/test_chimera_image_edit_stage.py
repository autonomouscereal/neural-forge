"""Genuine imported tiny QIMG family, streamed gradient/Adam/resume checks."""
import copy,dataclasses,importlib.util,json,os,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.chimera_recovery import train_image_edit as D,image_edit as E

def module():
    torch=pytest.importorskip('torch')
    # The streamed trainer has no tiny native harness. Keep its production
    # source pin separate from the actual HybridD3 fixture module.
    base=os.environ.get('CHIMERA_D3_HYBRID',os.environ.get('CHIMERA_D3_BASELINE'))
    source=os.environ.get('CHIMERA_OMNI_SRC')
    if not base or not source:pytest.skip('authentic baseline/foundation source environment required')
    source=Path(source)
    sys.path.insert(0,str(source if source.name=='src' else source/'src'))
    return torch,D.T.load_module(Path(base),'native_image_edit_fixture')

def fixture(tmp_path):
    torch,M=module()
    from chimera_omni.charted import ChartedBlockSpec
    # The original tiny harness has no QIMG. Add a correctly typed QIMG family
    # with its actual adaln6/layer/GELU-tanh/no-RoPE/time-factor contract.
    # Its random miniature weights are a mechanics fixture, never teacher data.
    M.TINY.FEATURE=6144
    M.TINY_BANDS={1:((512,576),(256,320))}
    M.tiny_unit_specs=lambda after:[ChartedBlockSpec(layer_index=after+1,source='qimg',component='tiny.qimg',
        type_ids=(1,),width=32,heads=2,kv_heads=2,head_dim=16,mlp_width=36,
        mlp_kind='gelu_tanh',modulation='adaln6',norm_kind='layer',norm_affine=False,
        norm_eps=1e-6,qk_norm=True,attn_bias=True,rope_theta=0.,sliding_window=0,
        time_freq_dim=16,time_hidden=32,time_factor=1000.,time_embed_r=False,chart_in_bias=True)]
    built=M.build_tiny_child(tmp_path/'native-image-edit',seed=705)
    T=M.trainer_module()
    def construct():
        return M.HybridD3(built['dir'],built['parent_config'],M.HybridOptions(dtype=torch.float32,device='cpu',kernels='torch',
            mode='streamed',pin_boundaries=False,prefetch=False,
            lora=T.SitePlanConfig(rank=3,alpha=6.,k3_rank=0,gain=False,trunk_ffn_width=M.TINY.TRUNK),lora_seed=706))
    h=construct();rows=8
    ids=torch.tensor([[1,2]+[h.media_placeholder]*rows]);features=torch.zeros(1,rows,6144)
    features[:,:4,512:576]=torch.randn(1,4,64,generator=torch.Generator().manual_seed(707))
    features[:,4:,512:576]=torch.randn(1,4,64,generator=torch.Generator().manual_seed(708))
    target=torch.zeros_like(features);target[:,:4,256:320]=.1
    batch=dict(input_ids=ids,native_media_features=features,native_media_targets=target,
        native_media_type_ids=torch.ones(1,rows,dtype=torch.int64),native_media_indices=torch.arange(2,2+rows).view(1,-1),
        native_media_positions=torch.zeros(1,rows,4),native_media_noise_levels=torch.tensor([[.3]*4+[0.]*4]),
        native_media_valid_mask=torch.ones(1,rows,dtype=torch.bool),native_media_loss_mask=torch.tensor([[1.]*4+[0.]*4]))
    batch['native_media_positions'][:,4:,0]=1.
    def compute(prediction):return (prediction[:,:4,256:320].float()-target[:,:4,256:320]).square().mean()
    return torch,M,construct,batch,compute

def test_wrong_stage_and_missing_sources_are_fail_closed(tmp_path):
    p=tmp_path/'wrong.json';p.write_text(json.dumps({'schema':D.SCHEMA,'stage':'D3b','objective':D.OBJECTIVE}))
    result=D.preflight(p)
    assert not result['candidate_preflight_passed'] and not result['production_ready']
    assert 'explicit distinct' in result['blockers'][0]

def test_actual_qimg_clean_reference_changes_output_and_gradients_scale(tmp_path):
    torch,M,construct,batch,compute=fixture(tmp_path)
    h1,h2=construct(),construct();unit=h1.layout[0].container
    engines=[D.ImageEditEngine(M,h,unit,D.OPTIMIZER,{'fixture':True},total_steps=2) for h in (h1,h2)]
    original=h1.forward(D.T.make_segment(batch,media_placeholder=h1.media_placeholder))['emission']
    changed={k:v.clone() for k,v in batch.items()};changed['native_media_features'][:,4:,512:576]+=1.
    reference_changed=h1.forward(D.T.make_segment(changed,media_placeholder=h1.media_placeholder))['emission']
    assert not torch.equal(original[:,:4,256:320],reference_changed[:,:4,256:320])
    expected=float(compute(original));actual=E.full_depth_backward(M,h1,batch,compute)
    scaled=E.full_depth_backward(M,h2,batch,compute,loss_scale=.3)
    assert abs(actual-expected)<1e-5 and abs(actual-scaled)<1e-6
    for name,p in engines[0].census.items():
        assert p.grad is not None,name
        torch.testing.assert_close(engines[1].census[name].grad,p.grad*.3,rtol=3e-4,atol=2e-5)
    # An actual observed reference influences output despite zero direct reference MSE.
    tracked=batch['native_media_features'].clone().requires_grad_()
    value={**batch,'native_media_features':tracked}
    context=h1.prepare(D.T.make_segment(value,media_placeholder=h1.media_placeholder),media_grad=True)
    # The native QIMG unit's self attention makes the reference readable before
    # the streamed trunk; direct local differentiation checks input influence.
    x=context['x'];write=h1.apply_units(x,1,context)
    write[:,2:6].square().sum().backward()
    assert tracked.grad is not None and float(tracked.grad[:,4:,512:576].abs().sum())>0

def test_actual_optimizer_authority_family_updates_and_exact_resume(tmp_path):
    torch,M,construct,batch,compute=fixture(tmp_path)
    def engine():
        h=construct();return D.ImageEditEngine(M,h,h.layout[0].container,D.OPTIMIZER,{'fixture':True},total_steps=2)
    first=engine();before={name:p.detach().clone() for name,p in first.census.items()}
    def revoked():raise D.Error('fixture authority revoked')
    with pytest.raises(D.Error,match='authority revoked'):first.train_step(batch,compute,gate=revoked)
    assert first.step==first.sampler_counter==0 and not first.optimizer.state
    assert all(torch.equal(p,before[name]) for name,p in first.census.items())
    result=first.train_step(batch,compute,gate=lambda:None)
    assert result['supervised_units']==4*64 and first.step==first.sampler_counter==1
    for family in ('unit','bridge','emission','lora'):
        assert any(not torch.equal(p,before[name]) for name,p in first.census.items() if name.startswith(family+'.')),family
    first.save(tmp_path/'checkpoint');resumed=engine();resumed.resume(tmp_path/'checkpoint')
    first.train_step(batch,compute,gate=lambda:None);resumed.train_step(batch,compute,gate=lambda:None)
    assert all(torch.equal(p,first.census[name]) for name,p in resumed.census.items())
    assert len({id(p) for g in resumed.optimizer.param_groups for p in g['params']})==len(resumed.census)
    assert resumed.step==resumed.sampler_counter==2
    with pytest.raises(D.Error,match='remaining one-pass'):resumed.train_step(batch,compute,gate=lambda:None)
