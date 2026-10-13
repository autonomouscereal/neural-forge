"""Actual pinned bridge/time/decoder CPU investigation for supervised geometry.

No clean target XYZ enters media features/positions. This does not run the trunk,
grant training admission or claim end-to-end original-parent parity.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

try:
    from . import hy3d_target_session as S
except ImportError:
    import hy3d_target_session as S
A, H = S.A, S.H
NATIVE_PINS = {
    "config.py": "ae93ef1e1bb0160803dc3a5a71243300c3e2b120542dfe3c4cde2ff4ab335957",
    "charted.py": "f3d6153c19bb0a690218fcf6af82c0fb4e6568f35cd018ce2f15f36f7f16eb57",
    "modeling/chimera.py": "8f2971edf76ab751a577d2cd95ec26fff61d0a90dfc1197fed67c1b25a6273fe",
    "modeling/charted.py": "0ee1875895a192cc4ce0dcd469a7cc2e9014b29b68ed3d8c4be978f24828ddc3",
    "modeling/hy3d.py": "1336c85db99b958cfa4097c6e1d9039205adad7782877663e36c4a38576d7b39",
    "modeling/layers.py": "6953425b53084f84b1619e345f9b687d09283ea4d03882a98f952fe47a9bc691",
}
CONFIG_SHA = "3516b587ce80461745b772eb92bf1d3c0ffd40fc5e0d82d83c037ad872c99b02"
INDEX_SHA = "564bd342cc8ddd63a3b28b2c0b37b595a4623020360c97bff87d2842858f6082"
PARENT_EXTRA = {
    "hy3dshape/hy3dshape/schedulers.py": "d3b25b84dac26f0e862cfc866da08df2d6592f01a0aaeda965782ab0fc1d8bba",
    "hy3dshape/hy3dshape/models/denoisers/hunyuandit.py": "27a74c5ba210468847aaf348bfd31ac2781dce094960ede0b2050d94233c531e",
}
SCHEMA = "recovery_geometry_bridge_cpu_probe.v1"
SAMPLE_SEED = 20261013


def media(torch, noisy, image, sigma, policy):
    """Only noisy 64-wide values and raw source image pixels are accepted."""
    if (policy not in ("ordinal_slot", "position_free_slot") or not 0 < sigma <= 1 or
        tuple(noisy.shape) != (1, 4096, 64) or tuple(image["patch_flat"].shape) != (1024, 1536)):
        raise ValueError("actual full source-image/noisy64 layout and explicit policy required")
    features = torch.zeros(1, 5121, 6144, dtype=torch.float32)
    features[0, :1024, 640:2176] = torch.from_numpy(image["patch_flat"].copy())
    features[:, 1025:, 576:640] = noisy
    types = torch.tensor([[6] * 1024 + [7] * 4097])
    positions = torch.zeros(1, 5121, 4, dtype=torch.float32)
    positions[0, :1024] = torch.from_numpy(image["native_media_positions"].copy())
    positions[:, 1024] = -1
    if policy == "ordinal_slot":
        positions[0, 1025:, 0] = (torch.arange(4096, dtype=torch.float32) + .5) / 4096
    levels = torch.zeros(1, 5121, dtype=torch.float32)
    # Old type7 pairing used original progress directly under the name sigma.
    # Canonical corruption sigma is kept separate. t_native=1-sigma preserves
    # that direct original progress→bridge contract without editing the model.
    levels[:, 1024:] = 1 - sigma
    return features, types, positions, torch.ones(1, 5121, dtype=torch.bool), levels


def metric(torch, first, second):
    delta = first.float() - second.float()
    return {"max_abs": float(delta.abs().max()), "relative_l2": float(delta.norm() / first.float().norm().clamp_min(1e-30)),
            "bitwise_equal": bool(torch.equal(first, second)), "allclose_atol2e-4_rtol2e-4": bool(torch.allclose(first, second, atol=2e-4, rtol=2e-4))}


def storage_hash(torch, value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(A.canonical({"dtype": str(value.dtype), "shape": list(value.shape)}))
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def original_time(torch, path, weights):
    if A.file_sha(path) != PARENT_EXTRA["hy3dshape/hy3dshape/models/denoisers/hunyuandit.py"]:
        raise ValueError("reviewed original native timestep source changed")
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("Timesteps", "TimestepEmbedder")]
    if len(nodes) != 2:
        raise ValueError("original native time classes required")
    namespace = {"torch": torch, "nn": torch.nn, "math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    from safetensors import safe_open
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        keys = ("mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias")
        state = {k: handle.get_tensor("model.t_embedder." + k).float() for k in keys}
    obj = namespace["TimestepEmbedder"](state["mlp.0.weight"].shape[1], state["mlp.0.weight"].shape[0], out_size=state["mlp.2.weight"].shape[0])
    obj.load_state_dict(state, strict=True)
    return obj.eval().requires_grad_(False), {k: S.tensor_hash(v) for k, v in state.items()}


def scheduler_check(torch, root):
    path = root / "hy3dshape/hy3dshape/schedulers.py"
    if A.file_sha(path) != PARENT_EXTRA["hy3dshape/hy3dshape/schedulers.py"]:
        raise ValueError("reviewed original native scheduler changed")
    name = "_geometry_original_native_scheduler"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    obj = module.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    obj.set_timesteps(sigmas=np.array([0., .25, .5, .75]), device="cpu")
    # Controlled actual operator execution: source scheduler advances from a
    # declared noise value to declared data under native velocity=data-noise.
    noise, clean = torch.tensor([2.]), torch.tensor([5.])
    sample, observed = noise.clone(), []
    for index, t in enumerate(obj.timesteps):
        sample = obj.step(clean - noise, t, sample).prev_sample
        next_t = obj.sigmas[index + 1]
        expected = (1 - next_t) * noise + next_t * clean
        if not torch.equal(sample, expected):
            raise ValueError("actual original native progress endpoint path differs")
        observed.append({"native_t": float(next_t), "sample": float(sample)})
    if not torch.equal(sample, clean):
        raise ValueError("actual source noise→data endpoint proof failed")
    return {"actual_original_scheduler_executed": True, "native_t0_is_noise_t1_is_data": True,
            "steps": observed, "canonical_sigma": "1-native_t", "native_velocity": "clean-noise",
            "canonical_velocity": "noise-clean", "scale_noise_helper_called_by_flow_pipeline": False}


def run(*, tree, clone, source_root, weights, plan, plan_sha256, output):
    if output.exists():
        raise ValueError("choose a fresh source-bound native investigation namespace")
    if A.file_sha(plan) != plan_sha256:
        raise ValueError("actual source/input investigation plan pin changed")
    p = json.loads(plan.read_text())
    if p.get("sample_seed") != SAMPLE_SEED or len(p["sources"]) != 2 or p.get("target_xyz_input_allowed") is not False:
        raise ValueError("bounded prescribed two-source nonleaking plan required")
    native_root = tree / "src/chimera_omni"
    for name, sha in NATIVE_PINS.items():
        if A.file_sha(native_root / name) != sha:
            raise ValueError("exact current original OMNI1 source changed:" + name)
    H.source_pins(source_root)
    if A.file_sha(clone / "config.json") != CONFIG_SHA or A.file_sha(clone / "model.safetensors.index.json") != INDEX_SHA:
        raise ValueError("actual immutable original OMNI1 clone config/index changed")
    if weights.stat().st_size != H.WEIGHT_BYTES or A.file_sha(weights) != H.WEIGHT_SHA:
        raise ValueError("original whole native geometry parent checkpoint changed")
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("native investigation must be CPU-only")
    torch.set_num_threads(8)
    sys.path.insert(0, str(tree / "src"))
    from chimera_omni.config import ChimeraConfig
    from chimera_omni.modeling.chimera import NativeContinuousMediaBridge
    from safetensors import safe_open
    from safetensors.torch import load_file
    import yaml
    config = ChimeraConfig.from_manifest(json.loads((clone / "config.json").read_text())["config"])
    census = [{"layer": v.layer_index, "source": v.source, "type_ids": list(v.type_ids)} for v in config.charted_blocks]
    bridge = NativeContinuousMediaBridge(config).float().eval().requires_grad_(False)
    index = json.loads((clone / "model.safetensors.index.json").read_text())["weight_map"]
    state, stored = {}, {}
    for name in bridge.state_dict():
        full = "native_media_bridge." + name
        with safe_open(str(clone / index[full]), framework="pt", device="cpu") as handle:
            value = handle.get_tensor(full)
        state[name] = value.float()
        stored[full] = {"shape": list(value.shape), "storage_dtype": str(value.dtype), "tensor_sha256": storage_hash(torch, value), "file": index[full]}
    bridge.load_state_dict(state, strict=True)
    del state
    time_parent, time_pins = original_time(torch, source_root / "hy3dshape/hy3dshape/models/denoisers/hunyuandit.py", weights)
    scheduler = scheduler_check(torch, source_root)
    params = H.validate_config(yaml.safe_load((source_root / "configs/dit_config_2_1.yaml").read_text()))
    vae = H.native_vae(source_root)(**params).float().eval().requires_grad_(False)
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        state = {k: handle.get_tensor("vae." + k).float() for k in vae.state_dict()}
    vae.load_state_dict(state, strict=True)
    del state
    before = {"vae": S.model_hash(vae), "bridge": S.model_hash(bridge), "time_parent": S.model_hash(time_parent)}
    output.mkdir(parents=True)
    generator = torch.Generator().manual_seed(SAMPLE_SEED)
    permutation = torch.randperm(4096, generator=generator)
    inverse = torch.argsort(permutation)
    reports, started = [], time.monotonic()
    with torch.inference_mode():
        for row in p["sources"]:
            tick = time.monotonic()
            for name in ("target", "image_ingress", "target_receipt"):
                A.pinned(row[name], name)
            target = load_file(row["target"]["path"], device="cpu")
            receipt = json.loads(Path(row["target_receipt"]["path"]).read_text())
            if receipt["source_row_sha256"] != row["source_row_sha256"] or receipt["id"] != row["id"]:
                raise ValueError("actual clean target/source identity differs")
            with np.load(row["image_ingress"]["path"], allow_pickle=False) as h:
                image = {k: h[k] for k in h.files}
            raw, model = target["clean_raw_latent"], target["clean_model_space_latent"]
            if raw.shape != (1, 4096, 64) or not torch.isfinite(raw).all() or not torch.equal(raw * float(vae.scale_factor), model):
                raise ValueError("actual full original model-space clean target differs")
            decoded = vae.decode(raw)
            shuffled = vae.decode(raw[:, permutation])
            oracle = target["physical_oracle_queries"]
            logits = vae.geo_decoder(queries=oracle, latents=decoded)
            permuted_logits = vae.geo_decoder(queries=oracle, latents=shuffled)
            decoder = {"decoded_equivariance": metric(torch, decoded, shuffled[:, inverse]),
                       "physical_query_logit_invariance": metric(torch, logits, permuted_logits),
                       "permutation_sha256": S.tensor_hash(permutation), "target_query_xyz_used": False,
                       "original_decoder_executed": True, "query_count": oracle.shape[1]}
            comparisons = []
            for sigma in (.2, .5, .8):
                eps = torch.randn(1, 4096, 64, generator=torch.Generator().manual_seed(SAMPLE_SEED))
                noisy = (1 - sigma) * model + sigma * eps
                ordinal = media(torch, noisy, image, sigma, "ordinal_slot")
                free = media(torch, noisy, image, sigma, "position_free_slot")
                captured = []
                hook = bridge.noise_proj.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
                xo, xf = bridge.embed(*ordinal), bridge.embed(*free)
                hook.remove()
                expected = bridge._fourier(torch.tensor([[[1 - sigma]]]))[0, 0]
                direct = all(torch.equal(v[0, 1024:], expected.expand(4097, -1)) for v in captured)
                if not direct or ordinal[0][0, 1024].any() or ordinal[2][0, 1024].tolist() != [-1.] * 4:
                    raise ValueError("actual native zero-feature time-row/noise_proj mapping failed")
                changed = noisy[:, permutation]
                yo = bridge.embed(*media(torch, changed, image, sigma, "ordinal_slot"))
                yf = bridge.embed(*media(torch, changed, image, sigma, "position_free_slot"))
                original_token = time_parent(torch.tensor([1 - sigma]), None)
                opposite_token = time_parent(torch.tensor([sigma]), None)
                comparisons.append({"canonical_sigma": sigma, "native_bridge_progress": 1 - sigma,
                    "actual_noise_proj_input_exact_direct_progress_fourier": direct,
                    "time_row_feature_all_zero": True, "time_row_position": [-1.] * 4,
                    "ordinal_vs_constantzero_latent_bridge": metric(torch, xo[:, 1025:], xf[:, 1025:]),
                    "ordinal_fixed_slot_permutation_effect": metric(torch, xo[:, 1025:], yo[:, 1025:][:, inverse]),
                    "positionfree_slot_permutation_effect": metric(torch, xf[:, 1025:], yf[:, 1025:][:, inverse]),
                    "image_and_time_policy_comparison_exact": bool(torch.equal(xo[:, :1025], xf[:, :1025])),
                    "original_timestep_parent_1minus_sigma_vs_sigma": metric(torch, original_token, opposite_token),
                    "original_time_token_sha256": S.tensor_hash(original_token),
                    "parent_time_to_trunk_calibration_claimed": False})
            reports.append({"id": row["id"], "split": row["split"], "family": row["family"],
                "source_row_sha256": row["source_row_sha256"], "decoder": decoder, "bridge": comparisons,
                "seconds": time.monotonic() - tick, "native_train_ready": False})
            (output / "status.json").write_bytes(A.canonical({"completed": len(reports), "expected": 2}) + b"\n")
    after = {"vae": S.model_hash(vae), "bridge": S.model_hash(bridge), "time_parent": S.model_hash(time_parent)}
    if after != before:
        raise ValueError("actual original model state changed during comparison")
    result = {"schema": SCHEMA, "source_year": 2026, "sealed": True, "plan": {"path": str(plan), "sha256": plan_sha256},
        "source_code_sha256": A.file_sha(Path(__file__)), "native_source_pins": NATIVE_PINS,
        "original_parent_source_pins": {**H.PINS, **PARENT_EXTRA}, "clone_config_sha256": CONFIG_SHA,
        "clone_index_sha256": INDEX_SHA, "original_whole_parent_sha256": H.WEIGHT_SHA,
        "bridge_actual_tensors": stored, "original_time_actual_tensors": time_pins, "native_charted_census": census,
        "runtime": {"torch": str(torch.__version__), "device": "cpu", "threads": 8, "reference_dtype": "FP32 exact widening of source storage"},
        "model_state_before": before, "model_state_after": after, "weights_unchanged": True,
        "original_scheduler": scheduler, "sources": reports, "elapsed_seconds": time.monotonic() - started,
        "native_train_ready": False, "training_started": False, "trunk_executed": False,
        "clean_target_xyz_in_inputs": False, "parent_end_to_end_parity_claimed": False,
        "positionfree_means": "constantzero latent bridge coordinate signal; actual cos(0) constant retained, time sentinel/image grid retained",
        "blockers": ["Full native tokenizer/insertion/trunk/read/flow objective integration and explicit selected amendment remain required.",
                     "Learned ordinal quality cannot be selected from boundary symmetry or untrained bridge differences alone."]}
    result["receipt_sha256"] = A.digest(result)
    (output / "probe.json").write_bytes(A.canonical(result) + b"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tree", "clone", "source-root", "weights", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    result = run(**vars(parser.parse_args(argv)))
    print(json.dumps({k: result[k] for k in ("schema", "elapsed_seconds", "native_train_ready", "receipt_sha256")}))


if __name__ == "__main__":
    main()
