import json
from pathlib import Path
import re
root = Path('C:/Users/cereal/Documents/ai training/reviews/2026-10-13-folding')
for p in root.glob('speech_memory_*.json'):
    data = json.loads(p.read_text(encoding='utf-8'))
    blob = json.dumps(data, ensure_ascii=True)
    matches = [blob[max(0, m.start()-80):m.end()+800] for m in re.finditer(r'\*\*Fish|\*\*Qwen3-TTS|\*\*Qwen3-Omni|LongCat-Audio-Codec', blob)]
    print(json.dumps({'file':p.name,'matches':matches[:5]},ensure_ascii=True))
