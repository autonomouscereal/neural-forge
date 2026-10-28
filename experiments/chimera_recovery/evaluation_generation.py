"""Bounded native text/tool recording, with real allowlisted local outcomes.

No historical command, arbitrary model code, external tool or media sampler is
executed. Actual model loading/allocation belongs to the source-pinned caller.
Source oracles never enter the generation/controller projection.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import time

try:
    from . import audio_admission as A
    from . import tool_seed as S
except ImportError:
    import audio_admission as A
    import tool_seed as S

S8_SHA='cfa9720f29bbfcad0baebd34a0d0a5d0923525a060ca066c9033a884b5f4a7dc'
TOOL_SOURCE_SHA='d7c4ad019b6735924494c8eea0e6828f847a2d3110aa0501879378d459b73f3c'
POLICY='source_native_cached_text_and_actual_local_tool_controller_2026-10-13.v1'
DECODING={'greedy':True,'thinking':False,'ledger_band_logit_mask':True,'max_new_tokens':512,
          'max_prompt_tokens':8192,'max_tool_rounds':8,'max_calls_per_round':8,'max_calls_per_episode':16}
PIN_FIELDS=('holdout_sha256','corpus_sha256','tokenizer_sha256','template_sha256',
            'input_config_sha256','decoding_config_sha256','runtime_sha256')


def strict_json(text):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate model JSON key')
            result[key]=value
        return result
    return json.loads(text,object_pairs_hook=unique,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def project_task(task):
    if (task.get('schema')!='recovery_eval_task.v1' or task.get('split') not in ('test','validation') or
            not task.get('id') or not task.get('group_id') or not isinstance(task.get('input'),dict)):
        raise ValueError('explicit frozen held-out source task required')
    value={'id':task['id'],'group_id':task['group_id'],'split':task['split'],'route':task['route'],
           'input':copy.deepcopy(task['input']),'input_sha256':A.digest(task['input']),
           'environment':copy.deepcopy(task.get('environment',{}))}
    if value['input'].get('media'):raise ValueError('native media generation requires separate real route adapter; S8 text shortcut refused')
    if any(not isinstance(m.get('content'),str) for m in value['input']['messages']):
        raise ValueError('text/tool generation cannot drop typed media content')
    environment=value['environment']
    if environment:
        if (set(environment)!={'followup_user_messages','followup_after_steps','tool_catalog_sha256','tool_source_code_sha256'} or
                environment['tool_source_code_sha256']!=TOOL_SOURCE_SHA or environment['tool_catalog_sha256']!=S.sha(S.TOOLS) or
                A.file_sha(Path(S.__file__))!=TOOL_SOURCE_SHA):
            raise ValueError('authentic deterministic tool controller source/catalog required')
        followups,steps=environment['followup_user_messages'],environment['followup_after_steps']
        if len(followups)!=len(steps) or any(type(n) is not int or not 1<=n<=8 for n in steps):
            raise ValueError('bounded source-user followup schedule required')
        if any(m.get('role')!='user' or not isinstance(m.get('content'),str) for m in followups):
            raise ValueError('only actual source user followups enter controller')
        for tool in value['input'].get('tools',[]):
            name=tool.get('function',{}).get('name')
            if name not in S.TOOL_MAP or tool!=S.TOOL_MAP[name]:
                raise ValueError('captured local schema differs from actual allowlisted implementation')
    value['controller']='actual_local_tools' if environment else 'proposal_or_text_only'
    return value


def load_native_source(path):
    A.pinned({'path':str(path),'sha256':S8_SHA},'reviewed original S8 native decode source')
    spec=importlib.util.spec_from_file_location('recovery_original_native_s8',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def native_decode(decoder,tokenizer,messages,tools,*,stops,config=DECODING):
    """Original native cached text path, authentic tools-aware chat template."""
    import torch
    if config!=DECODING or decoder.mask_band is not True:
        raise ValueError('exact frozen greedy/band-masked decode policy required')
    rendered=tokenizer.apply_chat_template(messages,tools=tools or None,tokenize=False,
                    add_generation_prompt=True,enable_thinking=False)
    ids=tokenizer(rendered,return_tensors=None)['input_ids']
    if not ids or len(ids)>config['max_prompt_tokens']:raise ValueError('native prompt exceeds frozen bound')
    generated=[];termination='length';started=time.monotonic();error=None
    with torch.no_grad():
        try:
            result=decoder.forward(torch.tensor([ids],dtype=torch.long),None)
            for _ in range(config['max_new_tokens']):
                token=int(result['logits'][0,-1].argmax().item())
                if token in stops:termination='eos';break
                generated.append(token)
                result=decoder.forward(torch.tensor([[token]],dtype=torch.long),result['cache'])
        except RuntimeError as exc:
            termination='error';error={'type':type(exc).__name__,'message':str(exc)}
    return {'text':tokenizer.decode(generated,skip_special_tokens=False),'token_ids':generated,
            'stop_reason':termination,'prompt_token_ids_sha256':A.digest(ids),'prompt_tokens':len(ids),
            'prompt_template_text_sha256':hashlib.sha256(rendered.encode()).hexdigest(),
            'elapsed_seconds':time.monotonic()-started,'model_forward_executed':True,
            'charted_rows_present':False,'media_generation_claimed':False,'runtime_error':error}


def verify_checkpoint(root,inventory):
    root=Path(root).resolve()
    if not inventory or len({entry['path'] for entry in inventory})!=len(inventory):
        raise ValueError('complete immutable checkpoint inventory required')
    for entry in inventory:
        original=root/entry['path'];path=original.resolve()
        if not path.is_relative_to(root) or original.is_symlink() or path.stat().st_size!=entry['bytes']:
            raise ValueError('checkpoint inventory escaped root or changed size')
        A.pinned({'path':str(path),'sha256':entry['sha256']},'immutable native evaluation checkpoint')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and
            not any(x in ('.git','.cache','__pycache__') for x in p.relative_to(root).parts)}
    if actual!={entry['path'] for entry in inventory}:raise ValueError('complete checkpoint inventory traversal required')
    return A.digest(inventory)


class NativeGenerator:
    def __init__(self,source,decoder,tokenizer,stops,checkpoint,inventory):
        source_module=sys.modules.get(type(decoder).__module__)
        if source_module is None or A.file_sha(Path(source_module.__file__))!=S8_SHA or type(decoder).__name__!='SplitDecoder':
            raise ValueError('genuine source-pinned native SplitDecoder required')
        self.source,self.decoder,self.tokenizer,self.stops=source,decoder,tokenizer,set(stops)
        self.checkpoint,self.inventory=Path(checkpoint),inventory
        self.checkpoint_sha256=verify_checkpoint(self.checkpoint,self.inventory)

    def __call__(self,messages,tools):
        return native_decode(self.decoder,self.tokenizer,messages,tools,stops=self.stops)

    def verify_unchanged(self):
        if verify_checkpoint(self.checkpoint,self.inventory)!=self.checkpoint_sha256:raise ValueError('native evaluation parent changed')


def load_generator(*,s8_source,native_tree,checkpoint,inventory,tokenizer,stops,split,devices):
    """Executable resident native loader; caller owns actual allocation/budget."""
    import torch
    source=load_native_source(Path(s8_source))
    source.bind_chimera_tree(Path(native_tree),None)
    from chimera_omni.config import ChimeraConfig
    from chimera_omni.importing.safetensors_io import read_checkpoint_config
    verify_checkpoint(checkpoint,inventory)
    cfg=ChimeraConfig.from_manifest(read_checkpoint_config(Path(checkpoint))['config'])
    placement=source.placement_for(cfg,split,devices)
    model,cfg,loaded=source.load_chimera_split(Path(checkpoint),placement=placement,dtype=torch.bfloat16,verify_tensors=True)
    decoder=source.SplitDecoder(model,cfg,placement,ledger_band_logit_mask=True)
    generator=NativeGenerator(source,decoder,tokenizer,stops,checkpoint,inventory)
    generator.load_receipt=loaded
    return generator


def parse_calls(text,task_id,counter):
    if len(text)>65536:raise ValueError('bounded model response required')
    spans=list(re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S))
    if not spans:
        if '<tool_call>' in text or '</tool_call>' in text:raise ValueError('incomplete tool-call markup')
        return text,[],counter
    remainder=re.sub(r'<tool_call>\s*.*?\s*</tool_call>','',text,flags=re.S).strip()
    if '<tool_call>' in remainder or '</tool_call>' in remainder:raise ValueError('malformed tool-call markup')
    calls=[]
    for span in spans:
        value=strict_json(span.group(1))
        if not isinstance(value,dict) or set(value)!={'name','arguments'} or not isinstance(value['name'],str):
            raise ValueError('exact native JSON tool-call fields required')
        arguments=value['arguments']
        if isinstance(arguments,str):arguments=strict_json(arguments)
        if not isinstance(arguments,dict):raise ValueError('tool arguments must be a JSON object')
        counter+=1
        calls.append({'id':f'recovery-{A.digest(task_id)[:12]}-{counter}','type':'function',
                      'function':{'name':value['name'],'arguments':S.canonical(arguments)}})
    return remainder,calls,counter


def record_one(projected,decode):
    """decode is the native generation callable; tests clearly use authored peers."""
    if 'oracle' in projected or projected.get('controller') not in ('actual_local_tools','proposal_or_text_only'):
        raise ValueError('only oracle-free projected model input accepted')
    messages=copy.deepcopy(projected['input']['messages']);prefix=len(messages)
    tools=projected['input'].get('tools',[]);allowed={t['function']['name'] for t in tools}
    rounds=[];outcomes=[];counter=0;dispatched=0;followed=set();stop='error';error=None
    with tempfile.TemporaryDirectory(prefix='recovery-eval-tools-') as directory:
        toolbox=S.Toolbox(directory)
        for number in range(DECODING['max_tool_rounds']):
            try:
                observed=decode(messages,tools)
                if (observed.get('model_forward_executed') is not True or observed.get('stop_reason') not in ('eos','length','error') or
                        not isinstance(observed.get('text'),str) or not isinstance(observed.get('token_ids'),list) or
                        any(type(token) is not int for token in observed['token_ids']) or len(observed['token_ids'])>512):
                    raise ValueError('actual bounded native generation receipt required')
                rounds.append(observed)
                if observed['stop_reason']=='error':
                    messages.append({'role':'assistant','content':observed['text']});error=observed.get('runtime_error');stop='error';break
                if projected['controller']=='proposal_or_text_only':
                    messages.append({'role':'assistant','content':observed['text']});stop=observed['stop_reason'];break
                content,calls,counter=parse_calls(observed['text'],projected['id'],counter)
                if not calls:
                    messages.append({'role':'assistant','content':content});stop=observed['stop_reason'];break
                if (observed['stop_reason']!='eos' or len(calls)>8 or counter>16 or
                        any(call['function']['name'] not in allowed or call['function']['name'] not in S.TOOL_MAP for call in calls)):
                    raise ValueError('unfinished, excessive or nonallowlisted model tool call refused')
                messages.append({'role':'assistant','content':content,'tool_calls':calls})
                def execute(call):
                    arguments=strict_json(call['function']['arguments'])
                    try:result=toolbox.execute(call['function']['name'],arguments)
                    except Exception as exc:
                        from jsonschema.exceptions import ValidationError
                        if not isinstance(exc,ValidationError):raise
                        result={'ok':False,'is_error':True,'error':{'type':'SchemaValidationError','message':str(exc)}}
                    return call,result
                if len(calls)>1:
                    # Shared filenames are prohibited across a concurrent batch;
                    # recorded parallel fixtures use distinct files. No race is
                    # silently accepted as a deterministic tool outcome.
                    names=[strict_json(c['function']['arguments']).get('name') for c in calls if c['function']['name']=='local_file']
                    if len(names)!=len(set(names)):raise ValueError('parallel file race refused')
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        pending=[pool.submit(execute,c) for c in calls];results=[f.result() for f in pending]
                else:results=[execute(calls[0])]
                dispatched+=1
                for call,result in results:
                    message={'role':'tool','tool_call_id':call['id'],'content':S.canonical(result),'is_error':not result['ok']}
                    messages.append(message)
                    receipt={'call_id':call['id'],'tool_name':call['function']['name'],
                        'arguments':strict_json(call['function']['arguments']),'result':result,'concurrent_dispatch':len(calls)>1,
                        'source_code_sha256':TOOL_SOURCE_SHA,'actual_execution':True}
                    outcomes.append({'tool_call_id':call['id'],'status':'success' if result['ok'] else 'error',
                        'source_verified':True,'receipt_sha256':A.digest(receipt),'result_sha256':A.digest(message),'receipt':receipt})
                env=projected['environment']
                for i,(after,user) in enumerate(zip(env['followup_after_steps'],env['followup_user_messages'])):
                    if after==dispatched and i not in followed:messages.append(copy.deepcopy(user));followed.add(i)
            except Exception as exc:
                error={'type':type(exc).__name__,'message':str(exc)};stop='error';break
        else:error={'type':'ControllerBudgetExceeded','message':'eight assistant generation rounds exhausted'}
    return {'schema':'recovery_eval_response.v1','id':projected['id'],'input_sha256':projected['input_sha256'],
            'messages':messages[prefix:],'tool_outcomes':outcomes,
            'generation':{'stop_reason':stop,'max_new_tokens':512,'token_ids':[t for r in rounds for t in r['token_ids']],
                          'rounds':rounds,'controller_policy':POLICY,'runtime_error':error},
            'model_generation_claimed':True,'source_oracle_exposed_to_model':False}


def record_tasks(tasks,decode,*,phase,pins,model,training_group_ids,output):
    if not isinstance(decode,NativeGenerator):raise ValueError('actual source-authenticated native generator required; authored callbacks are controller tests only')
    if output.exists():raise ValueError('fresh baseline/after recording namespace required')
    if phase not in ('baseline','after_recovery') or set(pins)!=set(PIN_FIELDS) or any(not A.SHA.fullmatch(str(v)) for v in pins.values()):
        raise ValueError('matched source/runtime/tokenizer/template/decoding pins required')
    if not A.SHA.fullmatch(str(model.get('checkpoint_sha256'))) or pins['decoding_config_sha256']!=A.digest(DECODING):
        raise ValueError('immutable model and exact bounded decoding policy required')
    if model['checkpoint_sha256']!=decode.checkpoint_sha256:raise ValueError('recorded model differs from genuine loaded native inventory')
    if hashlib.sha256(b''.join(A.canonical(task)+b'\n' for task in tasks)).hexdigest()!=pins['holdout_sha256']:
        raise ValueError('exact complete frozen task bytes/order required')
    decode.verify_unchanged()
    projected=[project_task(t) for t in tasks]
    if len({t['id'] for t in projected})!=len(projected) or len(set(training_group_ids))!=len(training_group_ids) or any(t['group_id'] in training_group_ids for t in projected):
        raise ValueError('held groups must remain disjoint from complete training inventory')
    responses=[record_one(t,decode) for t in projected]
    decode.verify_unchanged()
    receipt={'schema':'recovery_eval_run.v1','phase':phase,'pins':pins,'model':model,
             'training_group_ids':training_group_ids,'recording_complete':True,'full_answers_published':True,
             'responses_sha256':A.digest(responses),'controller_policy':POLICY,
             'source_code_sha256':A.file_sha(Path(__file__)),'native_s8_source_sha256':S8_SHA,
             'tool_controller_source_sha256':TOOL_SOURCE_SHA,'media_tasks_evaluated':0}
    output.mkdir(parents=True)
    (output/'responses.jsonl').write_bytes(b''.join(A.canonical(r)+b'\n' for r in responses))
    (output/'receipt.json').write_bytes(A.canonical(receipt)+b'\n')
    return receipt,responses
