import ast,json
from pathlib import Path
doc=json.loads(Path('proof_outputs/speech_official_api_oct13.json').read_text(encoding='utf-8'))
for row in doc['rows']:
    for source in row.get('files',[]):
        if 'text' not in source or not source['path'].endswith('.py'):
            continue
        print('\nSOURCE',source['path'],source['sha256'])
        parsed=ast.parse(source['text'])
        for node in ast.walk(parsed):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                print('IMPORT',ast.unparse(node))
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ['from_pretrained','transcribe','normalize_audios','generate_custom_voice','generate_voice_clone']:
                print('FUNCTION',node.name,ast.unparse(node.args))
        for index,line in enumerate(source['text'].splitlines(),1):
            if any(term in line for term in ['16000','SAMPLE_RATE','sf.read','librosa.resample','sampling_rate','do_sample','max_new_tokens','attn_implementation']):
                print(index,line)
