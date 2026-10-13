# Native D3a/c/d preparation, 2026-10-13

`stage_jobs.py` prepares three independently dated native-stage amendments and
seven configs from the actual current recovery drivers: five selected-unit
D3a refits, one D3c music-band job and one QVL D3d job. No model, tensor target,
GPU allocation or training is loaded. Missing bindings remain JSON `null`.
No native training-ready verdict is produced by this helper.

The frozen 15,339-byte body retains SHA256
`c24d21e08815d7acd2ce14aea989d16f84f167eb24a40597966be19284cb1370`.
Each new LF preregistration preserves the previous amendment tail, including
the D3b guided-velocity contract, before appending its own stage. The original
document and existing D3b v1 cache/amendment seals remain untouched. New
D3a/c/d batches must bind their new stage amendment SHA; an old seal cannot be
relabelled by editing its manifest.

The frozen/current drivers provide no numeric D3a/c/d optimizer defaults.
The generated documents explicitly date these initial recovery choices:

| Stage | Objective and optimizer sites | New explicit initial settings |
|---|---|---|
| D3a | Unit-only chart-in/block relative MSE plus `teacher_delta_host_write` on outward chart; all three term weights1. Exact D2-child host recapture and independent teacher-write target. QVL overall unit weight0.1; other units1. | Constant AdamW LR2e-5, betas0.9/0.95, epsilon1e-8, decay0, clip1. No schedule fields supported by this engine. |
| D3c | Conditional top64+tail KL over64000 written music rows; native prompt and causal next-code alignment. LazyRowAdam embedding/head rows plus all-layer native LoRA. | Band LR2e-5, LoRA LR1e-4; same Adam betas/epsilon/decay/clip;20 warmup steps then cosine to0.1. |
| D3d | Assistant-only answer KD weight1 over248320 text rows plus actual ViT-block13 feature MSE weight0.1. Selected QVL type6 unit, C11bridge delta and all-layer native LoRA. | Unit and bridge LR2e-5, LoRA LR1e-4; same Adam settings and20-step warmup/cosine to0.1. |

Seed20261013; one pass over actual accepted training batches. Native r16,
alpha16, no K3/gain and the actual base's trunk FFN width. These are dated
choices adapted from adjacent D3b, not previous approvals or measured optima.
Explicit dtype is selected at preparation; the example chooses float32 for
contract preparation. Actual execution must preserve the reviewed source and
dtype/config identity through native preflight/resume.

## Executable preparation

From this private worktree:

```powershell
C:/Python314/python.exe -B -m unittest discover -s tests -p test_chimera_stage_jobs.py
C:/Python314/python.exe -B experiments/chimera_recovery/stage_jobs.py prepare --engine-dir 'C:/Users/cereal/Documents/ai training/.agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-training-recovery-oct13-eae1b72e2c3f/experiments/chimera_recovery' --preregistration 'C:/Users/cereal/Documents/ai training/reviews/2026-10-13-folding/recovery-preparation/preregistration-v1/PREREGISTRATION_OMNI_STAGE_D_v1.md' --dtype float32 --out 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_native_stage_jobs_oct13_v1'
```

Preparation pins nine current driver/engine/adapter source files and the exact
input preregistration. `--evidence` may be repeated with actual source-status
receipts, whose bytes are pinned for context without conferring readiness.
Existing output directories are refused. Tests on another host use
`CHIMERA_STAGE_ENGINE_DIR`, `CHIMERA_STAGE_PREREGISTRATION` and, optionally,
`CHIMERA_STAGE_TEST_TMPROOT`. Authentication tests require the real current
engines and real frozen preregistration. Test-only contract fixtures contain
explicit non-tensor bytes and are never described as real teacher captures.

## Bind actual artifacts

After the affected native cache and exact stage base exist, supply a JSON
bindings file containing these actual local paths/values:

| Field | Required actual artifact |
|---|---|
| `hybrid_module` | Reviewed native hybrid module file |
| `immutable_base_receipt` | Registered immutable clone proof, equal base/clone hashes and exact clone path |
| `parity_receipt` | PASS on that exact base |
| `runtime_assignment` | Actual active exclusive assignment matching `device` |
| `batches_manifest` | Complete sealed admitted train native-stage manifest, heldout groups excluded, new amendment SHA |
| `omni` / `parent_config` | Existing immutable clone directory / authentic parent config file |
| `device` / `kernels` / `trunk_ffn_width` | Explicit native execution layout, positive integer actual FFN width |
| `python` | Exact supported runtime executable for production preflight |

Optional `compose_plan` and `decode_review_receipt` preserve the native driver's
interfaces. All other binding keys are refused. Optimizer/objective/unit/dtype
come from the pinned prepared template and cannot be silently overridden.

The exact next command is generated after preparation. Its general form is:

```powershell
C:/Python314/python.exe -B experiments/chimera_recovery/stage_jobs.py bind --plan '<actual manifest.json>' --plan-sha256 '<printed actual manifest SHA256>' --job-id D3a_ace_dit_L12 --bindings '<actual paths JSON>' --out '<new config.json>'
```

`bind` verifies current source bytes, document/tail pins, immutable base/parity/
runtime proofs and every batch/teacher file hash. It derives `steps` from unique
actual accepted batch IDs. It rejects partial traversal, holdouts, old
amendments, duplicate IDs and self-generated D3a outward targets or captures
from a different D2 child. It requires explicit conditional/causal D3c targets
and actual patch/answer features for D3d. It does not load tensor bytes into a
model or reproduce teacher values. The emitted binding receipt therefore
still states `production_ready=false` and contains the actual executable
`train_stages.py preflight --config ... --out ...` argv.
For D3a the manifest-level child, recaptured-host and host-write target pins are
also required and must match each sidecar; the engine builds its resume identity
from those manifest fields. The teacher pin equals the manifest's source teacher
SHA. Missing global target identity is refused before any model loading.

The production preflight performs the fuller authentic source/tensor-adapter
checks, including complete sealed ACE-LM traversal and D3d vocabulary/image/
answer-source alignment. Root runs that preflight on the supported runtime,
then checks its exact driver/engine/config hashes against this preparation.
Source changes require new preparation, not a pin waiver. A planned command
does not reserve a runtime or launch training.
Prepared paths are local and verified. A server job needs a fresh preparation
against the actually transferred server files and their measured hashes;
editing paths or marking an unperformed transfer complete is insufficient.

## Actual remaining data requirements

D3a cannot be run from pre-D2 host observations or a unit's own projected
output. Seal exact D2 child identity, newly recaptured host rows, independent
parent input/output and independent unit-owned host-write targets. Include
native row types, measured positions and the authentic noise/conditioning
contract for each selected unit.

D3c remains blocked by the actual partial ACE-LM traversal (838 captured shards
of6667 planned at the root's2026-10-13T17:32Z read-only status).4766 source-train-eligible
items are not native targets. Complete the source traversal and contamination/
group admission, then materialize the reviewed `music_cache.py` native adapter.

D3d has2294 source-admitted sealed ViT feature candidates under admissionv3,
but feature-only cache has no native `patch_flat` or assistant-answer KD. Those
independent aligned artifacts and exact native vocabulary equality are needed.
This QVL path covers vision/video question sources when its actual frame/input
alignment exists; it does not establish audio-to-text, speech or general3D
understanding. Procedural fixtures remain source evidence, not native targets.

The implemented D3a/c/d engines and dispatch/resume/checkpoint paths have
actual tiny CPU proofs, separate from full native dataset readiness. Root
reported71 native trainer tests passed. No D3a/c/d full recovery launch, trained
quality improvement or registered recovered model is claimed here. Keep the
30-minute/latest2 safety checkpoints, nonfinite skip halt, quarter-point decode
review and final native checkpoint composition/content evaluation.
