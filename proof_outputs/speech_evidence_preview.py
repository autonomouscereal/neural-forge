import json
import re
import sys
from pathlib import Path

data=json.loads(Path(sys.argv[1] if len(sys.argv)>1 else 'proof_outputs/speech_memory_qwen_fish_september.json').read_text(encoding='utf-8'))
pattern=re.compile(r'fish.?speech|fish.?audio|qwen3[-_/ ](?:tts|asr)|qwen2[.]5[-_/ ]omni',re.I)
def leaves(value, prefix=''):
    if isinstance(value,str):
        yield prefix,value
    elif isinstance(value,dict):
        for key,item in value.items():
            yield from leaves(item,prefix+'.'+key)
    elif isinstance(value,list):
        for index,item in enumerate(value):
            yield from leaves(item,prefix+'.'+str(index))
for row in data['rows']:
    if row['created_at']>='2026-10-13':
        continue
    print('\nEVENT',row['id'],row['created_at'],row['payload_sha256'])
    found=0
    for key,value in leaves(row.get('captured_evidence',{})):
        for match in pattern.finditer(value):
            print(key,repr(value[max(0,match.start()-120):match.end()+260]))
            found+=1
            if found>=2:
                break
        if found>=2:
            break
