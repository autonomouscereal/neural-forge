"""Metadata/parser tests only; model and optimizer paths are not executed."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,tempfile
import prepare_registered_native_entrypoint_v29 as patcher

ROOT=Path(__file__).absolute().parent
raw=(ROOT/'source_original/streamed_kd_trainer.py').read_bytes();derived=patcher.patch(raw)
before=ast.parse(raw.decode());after=ast.parse(derived.decode())
before_defs={n.name:n for n in before.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
after_defs={n.name:n for n in after.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert set(after_defs)-set(before_defs)=={'command_registered_native_preflight'}
unchanged=[]
for name,old in before_defs.items():
 new=copy.deepcopy(after_defs[name])
 if name=='command_train':
  assert isinstance(new.body[0],ast.If) and 'registered_native_input_plan' in ast.unparse(new.body[0]);new.body=new.body[1:]
 elif name=='parser':
  new.body=[n for n in new.body if not ('ni' in {x.id for x in ast.walk(n) if isinstance(x,ast.Name)} or (isinstance(n,ast.Expr) and '--registered-native-input-plan' in ast.unparse(n)))]
 assert ast.dump(old)==ast.dump(new),name
 unchanged.append(name)
spec=importlib.util.spec_from_file_location('isolated_actual_trainer_v29',ROOT/'candidate/streamed_kd_trainer.py')
module=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=module;spec.loader.exec_module(module)
args=module.parser().parse_args(['native-input-preflight','--native-plan','/plan','--native-plan-sha256','0'*64,'--native-plan-bytes','1',
 '--native-adapter','/adapter','--native-adapter-sha256','0'*64,'--device','cpu','--dtype','float32','--receipt','/receipt'])
assert args.func is module.command_registered_native_preflight
args=module.parser().parse_args(['train','--data-config','/not-read','--seq-len','1','--out','/not-created','--safety-dir','/not-created',
 '--registered-native-input-plan','/not-read'])
def forbidden(*a,**kw):raise AssertionError('model/preregistration/optimizer path was called')
module.preregistration_record=forbidden;module.build_source=forbidden
try:module.command_train(args)
except module.TrainerError as e:assert 'no model or optimizer was loaded' in str(e)
else:raise AssertionError('legacy training path accepted native input plan')
try:patcher.patch(raw+b'\n')
except ValueError:pass
else:raise AssertionError('wrong deployed source accepted')
print(json.dumps({'existing_definitions_preserved_except_explicit_parser_and_early_native_guard':len(unchanged),
 'actual_candidate_parser_registered':True,'native_train_rejected_before_model_or_preregistration':True,'wrong_baseline_refused':True,
 'model_optimizer_or_training_actions':0}))
