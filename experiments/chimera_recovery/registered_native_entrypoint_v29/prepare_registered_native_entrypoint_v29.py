"""Patch only an isolated exact deployed trainer; preserve all default paths."""
from pathlib import Path
import argparse,difflib,hashlib,json

TRAINER_SHA='77091f8ab1581c1edac3e3f1c1e5d727d60f8fb8465745c03a6eb6b12f0567da'
HANDLER='''def command_registered_native_preflight(args):
    import importlib.util
    adapter = Path(args.native_adapter)
    raw = adapter.read_bytes()
    if not adapter.is_absolute() or not 0 < len(raw) <= 128 * 1024 or hashlib.sha256(raw).hexdigest() != args.native_adapter_sha256:
        raise TrainerError("exact bounded native input adapter required")
    spec = importlib.util.spec_from_file_location("reviewed_registered_native_input_adapter_v29", adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.command_preflight(args)


'''
PARSER='''    ni = sub.add_parser("native-input-preflight", help="seven registered native inputs; CPU preparation only, no model/optimizer")
    ni.add_argument("--native-plan", required=True)
    ni.add_argument("--native-plan-sha256", required=True)
    ni.add_argument("--native-plan-bytes", required=True, type=int)
    ni.add_argument("--native-adapter", required=True)
    ni.add_argument("--native-adapter-sha256", required=True)
    ni.add_argument("--device", required=True)
    ni.add_argument("--dtype", required=True)
    ni.add_argument("--receipt", required=True)
    ni.set_defaults(func=command_registered_native_preflight)

'''
def patch(raw):
    if hashlib.sha256(raw).hexdigest()!=TRAINER_SHA:raise ValueError('exact deployed trainer baseline required; no pin waiver')
    source=raw.decode('utf-8');newline='\r\n' if '\r\n' in source else '\n'
    handler=HANDLER.replace('\n',newline);parser=PARSER.replace('\n',newline)
    marker='def parser() -> argparse.ArgumentParser:'
    assert source.count(marker)==1;source=source.replace(marker,handler+marker)
    marker='    gp = sub.add_parser("grad-parity", help="streamed LoRA grads vs a stock full-load autograd")'
    assert source.count(marker)==1;source=source.replace(marker,parser+marker)
    marker='    tr.set_defaults(func=command_train)'
    assert source.count(marker)==1
    source=source.replace(marker,'    tr.add_argument("--registered-native-input-plan", help="recognized but refused by the legacy D2 backend; use native-input-preflight")'+newline+marker)
    marker='def command_train(args: argparse.Namespace) -> int:'
    start=source.index(marker);body_start=source.index(newline,start)+len(newline)
    guard='    if getattr(args, "registered_native_input_plan", None):'+newline+'        raise TrainerError("registered native packets require a native-aware backend/objective; run native-input-preflight; no model or optimizer was loaded")'+newline
    source=source[:body_start]+guard+source[body_start:]
    return source.encode('utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    raw=a.source.read_bytes();derived=patch(raw)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('xb') as f:f.write(derived)
    diff=''.join(difflib.unified_diff(raw.decode().splitlines(keepends=True),derived.decode().splitlines(keepends=True),fromfile='deployed/streamed_kd_trainer.py',tofile='isolated/streamed_kd_trainer.py'))
    with a.out.with_suffix('.patch').open('x',encoding='utf-8',newline='') as f:f.write(diff)
    print(json.dumps({'original_sha256':TRAINER_SHA,'candidate_sha256':hashlib.sha256(derived).hexdigest(),'candidate_bytes':len(derived),'source_modified':False}))

if __name__=='__main__':main()
