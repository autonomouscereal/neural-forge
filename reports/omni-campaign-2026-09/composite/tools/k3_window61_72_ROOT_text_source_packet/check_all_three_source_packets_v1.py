"""Run only the three fixed source-only packet checks."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent.parent
for window in ('61_72','73_84','85_92'):
    packet=root/('k3_window'+window+'_ROOT_text_source_packet')
    argv=[sys.executable,'-B',str(packet/'check_future_ROOT_text_promoted_packet_v1.py'),'--candidate-root',str(packet)]
    p=subprocess.run(argv,check=False,timeout=60)
    if p.returncode:raise SystemExit(p.returncode)
