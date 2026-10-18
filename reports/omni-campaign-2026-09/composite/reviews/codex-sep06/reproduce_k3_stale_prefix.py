"""CPU-only reproduction of downstream done-marker reuse after an upstream refit.

Uses the existing three-layer fixture helpers. All large outputs are temporary
under this script's claimed review directory; only the JSON proof is retained.
No production code or model checkpoints are modified.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import torch
import test_k3_dense_pass as fixture_helpers
from safetensors.torch import load_file

kdp = fixture_helpers.kdp


def difference(new: torch.Tensor, old: torch.Tensor) -> dict:
    delta = new.double() - old.double()
    return {
        "equal": bool(torch.equal(new, old)),
        "max_abs": float(delta.abs().max()),
        "rel_l2": float(delta.norm() / old.double().norm().clamp_min(1e-30)),
    }


def detached_file(path: Path) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in load_file(str(path)).items()}


def main() -> None:
    torch.set_num_threads(2)
    log = io.StringIO()
    proof = {
        "case": "same-rung upstream refit must invalidate downstream done marker",
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu"},
        "production_code_sha256": kdp.sha256_file(str(TOOLS / "k3_dense_pass.py")),
        "tests_sha256": kdp.sha256_file(str(TOOLS / "test_k3_dense_pass.py")),
    }
    try:
        with tempfile.TemporaryDirectory(prefix="k3-stale-prefix-", dir=HERE) as temporary:
            work = Path(temporary)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                model_dir = work / "model"
                synth = fixture_helpers.build_synth(model_dir)
                fixture = {
                    "dir": str(model_dir), "index": fixture_helpers.k3_common.K3Index(str(model_dir)),
                    "synth": synth,
                }
                corpus = fixture_helpers.corpus_files.__wrapped__(work, fixture)
                out, receipts = work / "out", work / "receipts"
                common = {"chimera_src": os.environ.get("CHIMERA_OMNI_SRC", "")}
                argv = lambda **extra: fixture_helpers.base_argv(
                    str(model_dir), corpus, out, receipts, **common, **extra)

                # A complete original pass establishes a valid L2 artifact and marker.
                original = fixture_helpers.run_pass(argv(layers="0-2", k_pool_factor=1))
                old_l1_sha = original["layers"]["1"]["ffn"]["sha256"]
                old_l2_sha = original["layers"]["2"]["ffn"]["sha256"]
                old_snapshot = detached_file(out / "snapshots" / "before_L02_folded.safetensors")
                old_held = detached_file(out / "resid" / "L02" / "held_rows.safetensors")

                # Legitimate CLI-only upstream refit: same K=64, larger selection pool.
                upstream = fixture_helpers.run_pass(argv(layers="0-1", k_pool_factor=2))
                new_l1_sha = upstream["layers"]["1"]["ffn"]["sha256"]
                new_snapshot = detached_file(out / "snapshots" / "before_L02_folded.safetensors")

                # Resume from the new upstream state, while old L2/done.json still exists.
                resumed = fixture_helpers.run_pass(argv(layers="2-2", resume=True, k_pool_factor=1))
                new_held = detached_file(out / "resid" / "L02" / "held_rows.safetensors")

            l2 = resumed["layers"]["2"]
            proof.update({
                "rung": 64,
                "original_pool_factor": 1,
                "upstream_refit_pool_factor": 2,
                "resumed_pool_factor": 1,
                "old_l1_ffn_sha256": old_l1_sha,
                "new_l1_ffn_sha256": new_l1_sha,
                "old_l2_ffn_sha256": old_l2_sha,
                "resumed_l2_ffn_sha256": l2["ffn"]["sha256"],
                "before_l2_folded_prefix": difference(new_snapshot["prefix_sum"], old_snapshot["prefix_sum"]),
                "l2_actual_folded_ffn_input_held": difference(new_held["folded_moe_x"], old_held["folded_moe_x"]),
                "l2_teacher_ffn_input_held": difference(new_held["teacher_moe_x"], old_held["teacher_moe_x"]),
                "l2_fit_skipped": l2["fit_skipped"] is not None,
                "l2_fold_metrics_source": l2["fold_metrics_source"],
                "resume_layer": resumed["start"]["layer"],
                "resume_installed_layers": resumed["start"]["installed_layers"],
                "temporary_outputs_removed": True,
            })
            assert old_l1_sha != new_l1_sha, "Fixture failed to change the upstream fold"
            assert not proof["l2_actual_folded_ffn_input_held"]["equal"], "L2 input did not change"
            assert proof["l2_teacher_ffn_input_held"]["equal"], "Unexpected teacher-stream change"
            assert l2["fit_skipped"] is not None, "Defect not reproduced: L2 was refitted"
            assert l2["ffn"]["sha256"] == old_l2_sha, "Defect not reproduced: L2 artifact changed"
            proof["result"] = "CONFIRMED: old downstream fold reused on changed folded input"
    except BaseException:
        print(log.getvalue()[-14000:], file=sys.stderr)
        raise
    proof_path = HERE / "k3_stale_prefix_proof.json"
    kdp.write_receipt(str(proof_path), proof)
    print(json.dumps(proof, indent=2))
    print(f"Proof: {proof_path}")


if __name__ == "__main__":
    main()
