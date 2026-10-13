"""Scoped stock VL weight-placement control for one 24GiB GPU.

The stock model is loaded completely on CPU. Its actual visual tower and FP32
head execute on CUDA; Accelerate offloads whole stock decoder layers. Forward,
masking, M-RoPE, scoring and the original exact top-k gate stay source-owned.
This prepares a control only and does not launch or allocate a GPU.
"""
from __future__ import annotations
from contextlib import contextmanager
import gc
import importlib
from pathlib import Path

try:
    from . import video_teacher as T
except ImportError:
    import video_teacher as T

TPS_SHA='449eecc4acf985556a2454f4af25ead6379d2d515a7d77879f27d9c9d80f9a1a'
REVIEWED_TEXT_CONTROL_SHA='33a38cd9a1b8c04fa9b49df12c90639eac671cc9b12e80936cf5f0c22c963fa6'


def memory_available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:')))*1024


def require_budget(available,free_gpu,*,reference=False):
    minimum=(96 if reference else 128)*1024**3
    if available<minimum or (not reference and free_gpu<20*1024**3):
        raise ValueError('stock VL control requires128GiB initial/96GiB reference host headroom and20GiB initial GPU free')
    return {'minimum_available_host_bytes':minimum,'observed_available_host_bytes':available,
        'observed_free_gpu_bytes':free_gpu,'reference':reference}


@contextmanager
def reference_placement(writer):
    import torch
    import transformers
    import accelerate
    if (torch.__version__!='2.11.0+cu128' or transformers.__version__!='5.12.1' or
            accelerate.__version__!='1.14.0' or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported()):
        raise ValueError('source-audited runtime and exactly one remapped BF16 GPU required')
    if writer.engine_identity()['sha256']!=TPS_SHA:raise ValueError('actual reviewed tps_probe source changed')
    for name,expected in {**T.V.SOURCE_PINS,**T.MODEL_SOURCE_PINS}.items():
        if T.A.file_sha(importlib.import_module(name).__file__)!=expected:
            raise ValueError('actual source-audited stock parent implementation changed')
    initial=require_budget(memory_available(),torch.cuda.mem_get_info()[0]);factory=writer.build_full_load_teacher
    observations=[]

    def offloaded(model_dir,device,dtype,attn,vocab_rows=248320,head_precision='model'):
        gc.collect();torch.cuda.empty_cache()
        budget=require_budget(memory_available(),torch.cuda.mem_get_info()[0],reference=True)
        T.source_catalog(Path(model_dir))
        tps=writer.engine_module()
        model,facts=tps.load_resident_model(Path(model_dir),dtype,'cpu',attn,None,None)
        if facts['non_finite_parameter_tensors_after_cast']!=0:raise ValueError('stock parent cast contains nonfinite parameters')
        model.config.text_config.use_cache=False;trunk,head=tps.text_trunk_and_head(model)
        reference_head=writer.prepare_head(head.weight,vocab_rows,head_precision).to(device)
        visual=model.model.visual
        buffers={name:t.detach().cpu().clone() for name,t in visual.named_buffers()}
        visual.to(device=device).eval()  # Move only; never reconstruct/recompute the CPU rotary buffer.
        if any(not torch.equal(before,t.detach().cpu()) for name,t in visual.named_buffers() for before in [buffers[name]]):
            raise ValueError('stock visual buffer changed during explicit device placement')
        layer_classes=sorted({type(layer).__name__ for layer in trunk.layers})
        accelerate.cpu_offload(trunk,execution_device=torch.device(device),offload_buffers=True,
            preload_module_classes=layer_classes)
        facts.update(reference_placement='stock complete CPU checkpoint; stock visual CUDA; whole stock decoder-layer CUDA offload',
            offload_preload_classes=layer_classes,offload_buffers=True,accelerate_version=accelerate.__version__,
            gpu_head_dtype=str(reference_head.dtype),gpu_head_shape=list(reference_head.shape),
            visual_parameter_bytes=sum(p.numel()*p.element_size() for p in visual.parameters()),
            visual_buffers_cpu_construction_preserved_bitwise=True,source_tps_sha256=TPS_SHA,
            source_reference_ancestry_sha256=REVIEWED_TEXT_CONTROL_SHA,host_budget=budget)
        observations.append(facts)
        return writer.Teacher('stock-vl-cpu-offload',Path(model_dir),model.config,reference_head,
            text_trunk=trunk,vl_model=model.model,facts=facts,owner=model)

    writer.build_full_load_teacher=offloaded
    record={'schema':'recovery_video_stock_placement.v1','source_year':2026,'initial_budget':initial,
        'reference_facts':observations,'reference_placement_only_changed':True,'original_forward_and_exact_id_gate_unchanged':True,
        'code_sha256':T.A.file_sha(Path(__file__)),'reviewed_text_control_ancestry_sha256':REVIEWED_TEXT_CONTROL_SHA}
    try:yield record
    finally:writer.build_full_load_teacher=factory
