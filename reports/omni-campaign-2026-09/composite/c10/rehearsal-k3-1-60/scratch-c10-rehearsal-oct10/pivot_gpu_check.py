"""Scratch (M4 2026-10-10): c10 1.2.0 pivoted_columns_gram_on(cuda) vs c14 host loop on a random SPD Gram."""
import sys, time, json
sys.path.insert(0, "/home/cereal/omni-campaign/composite/tools")
import torch
import c10_union_install as c10
out = {}
g = torch.Generator().manual_seed(3)
n, k = 6000, 3000
a = torch.randn(2 * n, n, generator=g, dtype=torch.float64)
gram = a.T @ a
del a
t = time.time(); ref = c10.c14.pivoted_columns_gram(gram.clone(), k); out["host_seconds_6000_3000"] = round(time.time() - t, 2)
t = time.time(); got = c10.pivoted_columns_gram_on(gram, k, torch.device("cuda:0")); out["cuda_seconds_6000_3000"] = round(time.time() - t, 2)
out["identical_selection"] = bool(torch.equal(ref, got))
out["symmetric_difference"] = int(len(set(ref.tolist()) ^ set(got.tolist())))
del gram
n, k = 23552, 12288
a = torch.randn(n + 512, n, dtype=torch.float32, device="cuda:0")
gram = (a.T @ a).double().cpu()
del a; torch.cuda.empty_cache()
t = time.time(); got = c10.pivoted_columns_gram_on(gram, k, torch.device("cuda:0")); out["cuda_seconds_23552_12288"] = round(time.time() - t, 2)
out["selected"] = int(got.numel())
out["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated())
print(json.dumps(out), flush=True)
