"""Run the fixed three existing source-only checks."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent.parent
for window in ("61_72","73_84","85_92"):
 packet=root/("k3_window"+window+"_ROOT_joint50_future_source_packet")
 p=subprocess.run([sys.executable,"-B",str(packet/"check_NEXT_ROOT_joint50_promoted_packet_v1.py"),"--candidate-root",str(packet)],check=False,timeout=90)
 if p.returncode:raise SystemExit(p.returncode)
