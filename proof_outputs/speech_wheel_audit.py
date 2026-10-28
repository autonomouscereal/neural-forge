"""Read official tiny wheel sources in memory; no installation or code execution."""
import datetime,hashlib,io,json,zipfile,urllib.request
from pathlib import Path
doc=json.loads(Path('proof_outputs/speech_official_api_oct13.json').read_text(encoding='utf-8'))
rows=[]
for pkg in doc['rows']:
    if 'package' not in pkg:
        continue
    expected=next(f for f in pkg['files'] if f['filename'].endswith('.whl'))
    with urllib.request.urlopen(expected['url'],timeout=12) as stream:
        blob=stream.read(expected['size']+1)
    if len(blob)!=expected['size'] or hashlib.sha256(blob).hexdigest()!=expected['sha256']:
        raise ValueError('wheel_pin_mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        files=[]
        imports=[]
        for info in archive.infolist():
            if info.filename.endswith('.py'):
                body=archive.read(info.filename)
                text=body.decode('utf-8')
                if '/inference/' in info.filename or info.filename.endswith('__init__.py'):
                    files.append({'path':info.filename,'sha256':hashlib.sha256(body).hexdigest(),'text':text})
                for index,line in enumerate(text.splitlines(),1):
                    if line.lstrip().startswith(('import ','from ')):
                        imports.append({'path':info.filename,'line':index,'statement':line.strip()})
        rows.append({'package':pkg['package'],'version':pkg['version'],'wheel_sha256':expected['sha256'],'files':files,'imports':imports})
output={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only_wheel_audit':True,'rows':rows}
blob=(json.dumps(output,ensure_ascii=False,indent=2)+'\n').encode()
path=Path('proof_outputs/speech_verified_wheel_api_oct13.json')
with path.open('xb') as stream:
    stream.write(blob)
print(json.dumps({'path':str(path.resolve()),'sha256':hashlib.sha256(blob).hexdigest(),'verified_wheels':len(rows)}))
