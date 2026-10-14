"""Actual-entrypoint seven-row CPU preflight; no optimizer/model construction."""
from pathlib import Path
import ast,hashlib,importlib,json,os,sys

SCHEMA='chimera.actual_entrypoint_registered_native_preflight.v29'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read_pin(pin,cap=1024*1024):
 p=Path(pin['path'])
 if not p.is_absolute() or str(p).startswith(('\\\\','//')) or not 0<pin['bytes']<=cap:raise ValueError('bounded absolute local pin required')
 b=p.read_bytes()
 if len(b)!=pin['bytes'] or sha(b)!=pin['sha256']:raise ValueError('pinned source/plan changed:'+p.name)
 return b
def source_contract(plan):
 trainer=ast.parse(read_pin(plan['original_trainer']).decode('utf-8'))
 hybrid=ast.parse(read_pin(plan['original_hybrid']).decode('utf-8'))
 engine=next(x for x in trainer.body if isinstance(x,ast.ClassDef) and x.name=='LayerwiseEngine')
 forward=next(x for x in engine.body if isinstance(x,ast.FunctionDef) and x.name=='forward_backward')
 loader=next(x for x in trainer.body if isinstance(x,ast.FunctionDef) and x.name=='load_data_config')
 head_rows=None
 for x in trainer.body:
  if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TEXT_VOCAB_ROWS' for t in x.targets):head_rows=ast.literal_eval(x.value)
 h=next(x for x in hybrid.body if isinstance(x,ast.ClassDef) and x.name=='HybridD3')
 prepare=next(x for x in h.body if isinstance(x,ast.FunctionDef) and x.name=='prepare')
 hforward=next(x for x in h.body if isinstance(x,ast.FunctionDef) and x.name=='forward')
 names=lambda x:sorted({n.attr for n in ast.walk(x) if isinstance(n,ast.Attribute)})
 return {'D2_forward_backward_arguments':[a.arg for a in forward.args.args+forward.args.kwonlyargs],
  'D2_accepts_native_keyword_packet':forward.args.kwarg is not None,
  'D2_data_loader_uses_QTextCache':any(isinstance(n,ast.Name) and n.id=='QTextCache' for n in ast.walk(loader)),
  'D2_text_head_rows':head_rows,'Hybrid_forward_arguments':[a.arg for a in hforward.args.args],
  'Hybrid_forward_uses_no_grad':'no_grad' in names(hforward),
  'Hybrid_prepare_has_owned_H3_prefix_guard':'_h3_conditioning_prefix_context' in names(prepare),
  'Hybrid_prepare_uses_native_bridge':'_add_native_media' in names(prepare)}

def command_preflight(args):
 output=Path(args.receipt)
 if output.exists():raise ValueError('fresh receipt path required')
 plan=json.loads(read_pin({'path':args.native_plan,'sha256':args.native_plan_sha256,'bytes':args.native_plan_bytes}))
 if plan.get('schema')!='chimera.actual_training_entrypoint_native_input_plan.v29' or plan.get('optimization_allowed') is not False:raise ValueError('explicit no-optimization input plan required')
 report={'schema':SCHEMA,'PID':os.getpid(),'data_ready':False,'production_training_ready':False,'rows':[],
  'model_or_optimizer_constructed':False,'forward_backward_or_training_launched':False,'encoder_reruns':0,'blockers':[]}
 if args.device!='cpu' or args.dtype!='float32':
  report['blockers'].append('Only the actual CPU/float32 reader profile is qualified. Device/dtype changes require a new explicit binding; no implicit cast or GPU activation.')
 else:
  if plan['device']!=args.device or plan['floating_dtype']!=args.dtype:raise ValueError('plan device/dtype contract mismatch')
  root=Path(plan['seven_package_root'])
  if root!=Path(plan['seven_selection']['path']).parent:raise ValueError('explicit seven-row package root mismatch')
  dispatcher=read_pin(plan['seven_dispatcher'])
  if sha(dispatcher)!=plan['seven_dispatcher']['sha256']:raise ValueError('dispatcher changed')
  sys.path.insert(0,str(root));module=importlib.import_module('prepare_seven_linux_inputs_v28')
  if Path(module.__file__).read_bytes()!=dispatcher:raise ValueError('executing dispatcher source mismatch')
  import torch
  if torch.get_default_device().type!='cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') not in ('',None):raise ValueError('CPU-only preparation environment required')
  contracts=source_contract(plan);report['deployed_backend_contracts']=contracts
  for rid,kwargs,evidence in module.iter_seven_training_inputs(plan['seven_selection']):
   tensors=[]
   def walk(value):
    if isinstance(value,torch.Tensor):tensors.append(value)
    elif isinstance(value,dict):
     for v in value.values():walk(v)
   walk(kwargs)
   if any(t.device.type!='cpu' or (t.is_floating_point() and t.dtype!=torch.float32) for t in tensors):raise ValueError('actual input device/dtype differs from explicit CPU/float32 plan')
   labels=kwargs['labels'];supervised=labels[labels.ne(-100)]
   report['rows'].append({'record_id':rid,'route':evidence['route'],'native_rows':evidence['native_rows'],
    'tokens':evidence['tokens'],'labels':evidence['labels'],'supervised_token_max':int(supervised.max()),
    'supervised_tokens_outside_D2_text_head':int(supervised.ge(contracts['D2_text_head_rows']).sum()),
    'CPU_float32_packet_verified':True,'fresh_native_factory_binding':kwargs.get('h3_conditioning_prefix') is not None})
  if len(report['rows'])!=7:raise ValueError('all seven registered rows required')
  report['data_ready']=True
  report['blockers'] += [
   'Deployed D2 LayerwiseEngine accepts ids/teacher-targets/objective, not the registered native-media keyword packet. Its QTextCache source loader has no native source kind.',
   'Deployed HybridD3.forward takes one segment and returns no-grad hidden/emission outputs. It is not the ChimeraOmniModel keyword/loss-mapping API.',
   'Deployed HybridD3.prepare does not invoke the owned H3 conditioning-prefix guard required by the registered speech/video packets.',
   'The seven-row source supplies original assistant labels and genuine media conditioning; it supplies no authenticated Qwen/Kimi teacher-KD targets for the existing frozen D2 objective. No KD/CE substitution was performed.'
  ]
  outside=sum(r['supervised_tokens_outside_D2_text_head'] for r in report['rows'])
  report['supervised_tokens_outside_D2_text_head']=outside
  if outside:report['blockers'].append(f'{outside} supervised native-protocol tokens exceed the deployed D2 text head range. Do not truncate the vocabulary, mask these labels silently, or relabel the existing teacher objective.')
 report['next_owner_gate']='A reviewed native-aware model/engine and objective/framing binding for these exact packets, then actual checkpoint/device/dtype qualification. This preflight does not authorize or launch training.'
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({'data_ready':report['data_ready'],'rows':len(report['rows']),'production_training_ready':False,'blockers':len(report['blockers']),'training_launched':False}))
 return 2
