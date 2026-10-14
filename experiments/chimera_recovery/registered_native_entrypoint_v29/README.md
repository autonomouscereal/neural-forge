# V29 actual training-entrypoint integration candidate

The seven-row native dispatcher is now wired into an **isolated copy of the actual deployed D2 entrypoint**, through the opt-in `native-input-preflight` command. The candidate also recognizes `train --registered-native-input-plan` and refuses it before preregistration, model-source loading or optimizer creation. The deployed entrypoint and defaults were not replaced. This is working input-entrypoint wiring; it is not a native-aware D2 training implementation.

## Exact sources and owners

The deployed source is `/home/cereal/omni-campaign/distill/tools/streamed_kd_trainer.py`,154,350bytes, SHA256 `77091f8ab1581c1edac3e3f1c1e5d727d60f8fb8465745c03a6eb6b12f0567da`. The deployed native hybrid is `d3_hybrid_module.py`,114,439bytes, SHA256 `540fca2d9d976da336c8d24ae5369c5a303ff37367cae54a9bcfc762a7e95197`.

The recovery-owner source branch is `codex/work/codex/chimera-training-recovery-oct13-eae1b72e2c3f`, commit `81f549884203193efdff6a842054937410ab0ddc`, observed clean. The separate shared Chimera checkout is on `claude/codex-oct08-merge`, commit `11adb320732cfa96863679d9f47c5e4e91c5058b`, with pre-existing edits to curated-data files. Branch names do not prove an attended peer's current identity or acceptance. The user identifies training/folding/GPU authority as thread `01a10f78-c196-73a0-a5ad-717c1fd030e6`.

This candidate uses a new private worktree and branch, `codex/work/codex/chimera-seven-native-entrypoint-v29-owner-base-3e0d60ee1fcc`, based on the exact recovery-owner commit. No shared checkout, validated best branch, live queue, checkpoint, default master or evaluation was modified or promoted.

## What actually ran

In the established Linux Python3.12/image20e01d3a runtime, the patched deployed entrypoint read the pinned combined seven-row configuration and freshly prepared all3code,1speech,2image and1video packets. Every actual input tensor was on CPU; every floating tensor was float32. The preflight reported `data_ready=true` and `production_training_ready=false`, returning2 deliberately. The case controller exited0 because all expected readiness/refusal checks passed.

The same entrypoint refused unqualified bfloat16 and CUDA profiles before reading any rows. Its legacy `train` route refused a registered-native plan before model, optimizer or preregistration loading; the requested output directory was not created. No checkpoint was mounted. No model construction, forward, backward, optimizer or training was executed. The CPU container exited0 after the four cases. The measured peak child RSS was1,094,119,424bytes under an8GiB envelope.

Static/parser tests additionally proved74existing function/class definitions unchanged, except the explicit parser additions and first-statement native-plan refusal. A wrong baseline hash refuses patch generation. The source patch is1577bytes larger than the immutable deployed original; the change does not rewrite the existing objective, sampler, safety checkpoint or resume behavior.

## Precise backend blockers

1. Actual D2 `LayerwiseEngine.forward_backward(self, ids, targets, objective, *, denominator, loss_scale)` has no native-media packet argument. `load_data_config` requires QTextCache text/K3/on-policy sources. The seven-row source does not contain authenticated Qwen/Kimi KD targets.
2. Actual `HybridD3.forward(self, segment)` returns no-grad hidden/context/emission outputs, not `ChimeraOmniModel(**kwargs)`'s loss mapping. Its `prepare` uses the native bridge but does not invoke the owned H3 conditioning-prefix guard required by these speech/video packets.
3. There are **14supervised native-protocol tokens outside D2's248,320row text head**, two per record; observed maximum ID248328. The current seven-row native model configuration supports its own protocol. It is not honest to truncate the head, silently mask these labels or relabel this data as legacy D2 teacher KD.

The smallest remaining owner decision is an explicit backend/objective/framing binding: either a reviewed native-aware full-vocabulary consumer for these exact packets, or an explicitly authorized native framing/objective transformation with fresh packet/source pins. That decision must also preserve H3 prefix causality. Then qualify the actual checkpoint and chosen device/dtype through the owning model path. No such production backend/checkpoint or objective approval is fabricated by this input preflight.

## Run the concrete candidate

Host adapter input root:
`/home/cereal/omni-campaign/receipts/codex-native-entrypoint-01a11bd5-20261014-v29/input`

Mount that read-only at `/work/adapter`, and the verified seven-row input root read-only at `/work/input`. Use the same Python3.12/venv mounts documented in the V28 seven-row README. The verified command is:

```sh
/usr/bin/python3.12 -B /work/adapter/streamed_kd_trainer_candidate_v29.py native-input-preflight \
  --native-plan /work/adapter/NATIVE_ENTRYPOINT_PLAN_V29.json \
  --native-plan-sha256 4d8950e23125f662ae33e26194948cc96e2b241836d1110e36863a50add03936 \
  --native-plan-bytes 2067 \
  --native-adapter /work/adapter/registered_native_inputs_v29.py \
  --native-adapter-sha256 c8e6a00f0175228434b8f0c5454094fb49629927447adb601bcf9afaedc35908 \
  --device cpu --dtype float32 --receipt /work/output/NEW_RECEIPT.json
```

Expected current result is seven valid input packets plus the explicit backend blockers, exit2. This is not a training command. The candidate entrypoint SHA256 is `97af4cf4488896b3879d2e57169b676f07953c3cfa8549929e6f1dc0d1a86d7b`.

## Coordination and handoff

The supported exact-context resolver returned `native_context_unavailable` twice. The installed coordination skill requires missing exact identity to fail closed and publish nothing. No Relay token/session was borrowed, no identity was registered, and no private peer thread was read as a substitute. Claude's attended session/acceptance is therefore unverified. The enclosing parent receives this artifact through the platform's existing delegation handoff; direct Relay delivery remains pending normal exact-context recovery.

Required owner handoff: review this isolated branch/commit and exact deployed-source patch; retain the successful seven-row data/CPU qualification; decide and bind the compatible native model/objective/framing contract before any training adoption. Keep the ten historical candidates quarantined. The candidate has not been installed into the owner's deployed tools or queue.
