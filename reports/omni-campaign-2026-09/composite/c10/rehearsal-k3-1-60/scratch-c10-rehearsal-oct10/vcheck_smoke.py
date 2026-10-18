"""Scratch smoke (M4, 2026-10-10): c10_virtual_child_check's streamed engine on the REAL S3 trunk with
child := trunk (identity), 8 R-Q prompts.  Expect: finite, top-1 agreement 1.0, KL 0, residual relL2 0."""
import json, sys, time, argparse
sys.path.insert(0, "/home/cereal/omni-campaign/composite/tools")
import torch
import c10_virtual_child_check as check
from pathlib import Path
from chimera_omni.config import ChimeraConfig
from chimera_omni.importing.safetensors_io import read_checkpoint_config, read_safetensors
T = Path("/mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native")
doc = read_checkpoint_config(T); cfg = ChimeraConfig.from_manifest(doc["config"])
views = read_safetensors(T, verify=False)
ns = argparse.Namespace(prompt_ids=None, tokenizer_dir="/mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07", max_tokens=128)
packs, rec = check.resolve_prompts(ns)
print("tokens", [len(p) for p in packs], flush=True)
layers = {k: {n: views[f"blocks.{k}.{n}"].torch(device="cpu", dtype=torch.bfloat16) for n in check.FFN_NAMES} for k in range(cfg.num_layers)}
t0 = time.time()
lt, lc, facts = check.streamed_logits(cfg, cfg, views, layers, {}, packs, torch.device("cuda:0"), torch.bfloat16)
cmp = check.compare(lt, lc, int(cfg.ledger_band_begin()))
print(json.dumps({"seconds": time.time() - t0, "forward_seconds": facts["seconds"], "top1": cmp["top1_agreement"],
                  "kl": cmp["kl_trunk_child_mean"], "finite": cmp["all_logits_finite"],
                  "max_resid_rel": max(r["residual_relL2_child_vs_trunk"] for r in facts["per_layer"]),
                  "last_tokens": [p["last_position_top1"] for p in cmp["prompts"]],
                  "peak_cuda": torch.cuda.max_memory_allocated()}, indent=0), flush=True)
