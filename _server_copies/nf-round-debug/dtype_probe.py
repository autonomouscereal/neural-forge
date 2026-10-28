import sys, torch
RUNTIME="/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a"
sys.path.insert(0, RUNTIME+"/src"); sys.path.insert(0, RUNTIME)
from experiments import run_lfm2_residual_attention_fold as residual
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent, _execution_identity
from pathlib import Path
SEALED="96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb"
SEALED_EXEC="3d077b94a3f7a4907f5280acfdc35284ed6e2ba109b48edf7111c45b74266f2c"
PARENT=Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")
dev=torch.device("cuda:0")
for name in ("float32","bfloat16","float16"):
    try:
        m=residual.load_local_lfm2_model(PARENT, device=dev, dtype=residual._torch_dtype(name))
    except Exception as e:
        print(name,"LOAD FAIL",repr(e)[:200]); continue
    fp=fingerprint_lfm2_parent(m).identity_sha256
    pd=str(next(m.parameters()).dtype)
    m.config._attn_implementation="eager"
    ex=_execution_identity(m, dev)
    print(name, "param_dtype=", pd, "identity=", fp, "MATCH=", fp==SEALED)
    print("   exec=", ex["execution_sha256"], "MATCH=", ex["execution_sha256"]==SEALED_EXEC, ex)
    del m; torch.cuda.empty_cache()
