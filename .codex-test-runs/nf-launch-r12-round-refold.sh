#!/usr/bin/env bash
# r12 post-publication sequence: one sealed on-policy evidence extension round
# against r12's published child, then one refold that consumes it.
#
# This script is STAGED, NOT EXECUTED.  Run one stage at a time:
#
#   SOURCE_COMMIT=<postpub commit> ./nf-launch-r12-round-refold.sh preflight
#   SOURCE_COMMIT=<postpub commit> ./nf-launch-r12-round-refold.sh round
#   SOURCE_COMMIT=<postpub commit> ./nf-launch-r12-round-refold.sh refold
#
# Why two runtimes are in play: r12 itself was folded by the 7b48e205
# deployment, which predates ab3ec99a (dense/carrier lanes consume evidence
# sets) and 09825ab2 (bridged rounds seal the target-token preimage).  A round
# or refold has to run from a runtime that carries both, so RUNTIME below is
# the postpub checkout and R12 is read only for its published child.  r12's own
# checkout, venv, run directory, and receipts are never written here.
#
# Why the round is not driven by run_lfm2_dynamic_catalog.py: that runner only
# produces rounds for the child its own fold pass just published, and it builds
# the capture configuration from its flags.  This lane's sealed base cache was
# captured with require_top1_agreement AND
# allow_top1_disagreement_within_numeric_error_envelope both true
# (selection f229e5f3...); the runner never sets the second, so no flag set
# reproduces that selection and the round would fail the comparable-capture
# clause.  run_r12_on_policy_round.py rebuilds the selection from the sealed
# base manifest and proves it reproduces the sealed digest before capturing.
#
# W12E (see NOTES-W12D and NOTES-W12E).  Two things changed for the relaunch,
# both because the first round failed closed twice for two unrelated reasons:
#
#   * the round now runs with --screen-cached-decode-admissibility, because the
#     degenerate r12 child emits runs of control tokens whose massive
#     activations make the parent's own bf16 cached decode genuinely disagree
#     with its use_cache=False recompute on 4 of 119 candidate trajectories.
#     The screen excludes and records those instead of killing the round; the
#     floor of 100 keeps a much more degenerate child from sealing quietly.
#   * ROUND_BASE_CACHE below is a recapture of BASE_CACHE's own authenticated
#     corpus under the postpub runtime's implementation identity.  The sealed
#     2026-07-17 cache carries collector 71e99ee9/canonical 5956f909 and the
#     runtime carries 16271526/789c6a6f, so the extension and the base could
#     never agree on implementation_sha256 and no closed-form solve could
#     consume them together.  Corpus, capture selection, parent identity,
#     execution identity and layer topology are all bit-identical to the
#     sealed cache; only the implementation digests move.
#
#   The reviewed relaunch commit is
#   13f711323e923b683347de1dfd315dfe4547f924.
#
# W13D (see NOTES-W13D).  The refold caveat that stood here is resolved, and
# the refold stage now carries the composed r13 flag set.
#
#   * The refold consumes ROUND_BASE_CACHE, not BASE_CACHE.  It has to: the
#     round's extension cache carries implementation 732ca9ce... and
#     Lfm2SealedEvidenceSet.open refuses members that disagree on
#     implementation or layer topology, so the sealed eb3bddeb... base could
#     not be opened alongside the round at all.  REBOUND below holds the two
#     bridge-bound whole-output envelopes re-sealed against ROUND_BASE_CACHE
#     by nf-w13d/rebind_bridge_bound_outputs.py, from the sealed pack's own
#     one-time parent rows: same rows, same bridge evidence, only
#     parent_cache_key and the seals that cover it moved.  Zero parent loads,
#     zero parent forwards, zero generations; the sealed pack is untouched and
#     the corpus inputs (--calibration, --target-tokenizer-calibration) still
#     come from it, because the recapture preserved them bit for bit.
#   * The r13 flag set follows reports/r12-certified-20260804/R13_PLAN.md
#     section 3 with NOTES-W13B's and NOTES-W13C's verdicts:
#       - the moving-gauge reclosure must now prove it improved every boundary
#         it overwrites (require_boundary_improvement, tolerance 1e-6 for
#         W13C's FP32 null-change caveat);
#       - the terminal composition floor and the advisory free-run tripwire
#         are on, so a run headed for r12's wall stops at the wall layer;
#       - per-layer conditioning, gate-operating-point and depth telemetry are
#         on, so the depth profile is reconstructable from the receipt alone;
#       - the EIV input-error covariance and the margin row weights are
#         deliberately OFF.  This arm is the multiplier=0 control, so the
#         reclosure gate's effect on the composed result stays cleanly
#         attributable;
#       - install verification is fingerprint, which trades the install
#         sweep's ~17.7h for per-step replacement-digest authentication plus
#         one terminal capture that still pins the full child state.  A
#         full-recapture certification pass is reserved for a child that
#         reviews well.
#     W13L pins the authenticated W13K basis key below.  This evidence identity
#     is unchanged by b65c81f5, so a mismatch must fail closed instead of
#     silently rebuilding the same 85,984,075-byte basis.
set -euo pipefail

STAGE="${1:-}"
case "$STAGE" in
  preflight|round|refold) ;;
  *)
    echo "usage: SOURCE_COMMIT=<sha> $0 {preflight|round|refold}" >&2
    exit 64
    ;;
esac

: "${SOURCE_COMMIT:?Set SOURCE_COMMIT to the reviewed postpub-deployment commit}"

LAB=${LAB:-/home/cereal/model-surgery-lab}
RUNTIME=${RUNTIME:-$LAB/deployments/neural-forge-6f178f14}
R12=$LAB/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12
RUN=${RUN:-$LAB/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0025-exact4x-h816-mgv3-r12-w13k-cold}
CHILD_NAME=${CHILD_NAME:-lfm2-integrated-carrier-exact4x-h816-mgv3-r12-w13k-cold}
RESUME_FROM_CAPSULE=${RESUME_FROM_CAPSULE:-}
RESUME_ARGS=()
if [ -n "$RESUME_FROM_CAPSULE" ]; then
  RESUME_ARGS=(--dense-resume-from-capsule "$RESUME_FROM_CAPSULE")
fi
ROUND_ROOT=$LAB/artifacts/neural_forge_on_policy_rounds_v1/a0023-exact4x-h816-mgv3-r12
ROUND_MANIFEST=$ROUND_ROOT/round-000/round-manifest.json
CHILD=$R12/published-child/child

DERIVED=$LAB/artifacts/neural_forge_phase_cache_v1/lfm2-global-topology-derived
BALANCED_BASIS=$LAB/artifacts/neural_forge_phase_cache_v1/lfm2-circuit-balanced-residual-basis-v1
BALANCED_BASIS_EVENTS="$RUN/balanced-basis-phase-events.jsonl"
BALANCED_BASIS_HMAC_KEY=$LAB/keys/neural-forge-balanced-basis-phase-v1.key
BALANCED_BASIS_HMAC_KEY_ID=lfm2-circuit-balanced-residual-basis-v1
PACK=$LAB/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5
BASE_CACHE=$PACK/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95
# The W12E recapture of BASE_CACHE, produced by nf-w12e/recapture_r12_base_cache.py
# from BASE_CACHE's own authenticated input tensors under the postpub runtime.
# Same corpus 71884339..., same selection f229e5f3..., same parent 96a6779d...,
# same execution 3d077b94..., same layer topology 92166d4d...; implementation
# eb3bddeb... -> 732ca9ce..., which is the only reason it exists.
ROUND_BASE_CACHE=$ROUND_ROOT/base-cache-recaptured-w12e/c29b67370d5e33be47a11daccb1e6c2144d6e85efb024621458dc14ca5bd5ef7
# W13D: $PACK's two whole-output envelopes re-sealed against ROUND_BASE_CACHE.
# Their rows are byte-identical to the sealed pack's canonical_parent_rows.json
# and mapped_parent_rows.json, which are copied in beside them; the refold's
# preflight below re-proves both facts before anything is loaded.
REBOUND=$ROUND_ROOT/bridge-bound-outputs-recaptured-w13d
REBOUND_PARENT_OUTPUTS_SHA256=fa97d9a8d70f2974832c4971f517eb64d00bd62095946f80ad2f0db2b696cafa
REBOUND_MAPPED_PARENT_OUTPUTS_SHA256=4fea48e1494fa0151f4b58d64824b2e8dac0829373183678537e57de3bf0989e
DONOR=$LAB/artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46
BRIDGE=$LAB/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2
CARRIER=$LAB/artifacts/neural_forge_causal_support_20260730/a9-a11-base-dual-route/4439a2b7-b301539b
SPAN=$LAB/artifacts/neural_forge_phase_cache_v1/lfm2-semantic-step-global-raw-span/3e780d07ac1034ee77756f973b483637cee846e17e6d7191d6b20c9533360167/payload/span-evidence.json
PARENT=$LAB/models/LFM2.5-230M
PROMPTS="$RUNTIME/research/lfm2_semantic_whole_response_prompts_v1.json"
FIT_PROMPTS="$RUNTIME/experiments/lfm2_on_policy_fit_prompts_v1.json"
MANUAL_DIGESTS="$RUNTIME/experiments/lfm2_manual_review_reserved_digests_v1.json"
VENV_PYTHON=$LAB/venv/bin/python

BRIDGE_ARTIFACT_SHA256=5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384
BRIDGE_PROFILE_SHA256=e3fc98c666b70b367962e400435afe7588a9f1a2b13619dffc6f40c6458c5ca1
MANUAL_PROMPTS_SHA256=0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b
FIT_PROMPTS_SHA256=e85000326c5c39ae683506d69bc7e8edf3557845dc3b06e562f5ba6c23fe436f
MANUAL_DIGESTS_SHA256=d9f196956c6da0c7c4bf4d538005d17f2242f4c403f31014525f2318b88cf92c
BASE_CAPTURE_SELECTION_SHA256=f229e5f399858b7a512ec0d7034fe4b3d76cf4dc3bb214ec461a8a6f7c000258

export PYTHONPATH="$RUNTIME/src:$RUNTIME"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Same UUID pinning r12 used, so cuda:0 and cuda:4 are the two RTX 3090s and
# keep the ordinals an operator reads from nvidia-smi.  The round needs one
# 3090; the refold shards the dense closed-loop replay across both.
export CUDA_VISIBLE_DEVICES=GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561,GPU-c95fb5b3-f97c-bfb2-9760-79fe107ffe6d,GPU-a92b0dfe-83f8-11ff-db0d-c65921f365b4,GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8,GPU-2ce6a95c-7b1f-644a-3808-d781732a0582
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

actual_source_commit=$(git -C "$RUNTIME" rev-parse HEAD)
if [ "$actual_source_commit" != "$SOURCE_COMMIT" ]; then
  echo "Runtime source differs from requested postpub commit" >&2
  exit 41
fi
if [ -n "$(git -C "$RUNTIME" status --porcelain)" ]; then
  echo "Postpub runtime checkout is dirty" >&2
  exit 41
fi

preflight() {
  # r12 must have published, and must no longer be running: this sequence
  # reads its child and then competes for the same two 3090s.
  if [ -f "$R12/run.pid" ] && kill -0 "$(cat "$R12/run.pid")" 2>/dev/null; then
    echo "r12 is still running (pid $(cat "$R12/run.pid")); refusing" >&2
    exit 43
  fi
  for required in "$R12/receipt.json" "$CHILD/config.json" \
    "$CHILD/model.safetensors" \
    "$CHILD/neural_forge_tokenizer_bridge_reference.json"; do
    if [ ! -f "$required" ]; then
      echo "r12 publication is incomplete: missing $required" >&2
      exit 44
    fi
  done
  # The bridged round refuses a child published under a different bridge, and
  # the refusal is much cheaper to read here than after a rollout.
  "$VENV_PYTHON" - "$CHILD" "$BRIDGE_ARTIFACT_SHA256" "$BRIDGE_PROFILE_SHA256" <<'CHILD_BRIDGE_PREFLIGHT' || exit 45
import json
import sys

root, artifact, profile = sys.argv[1:4]
reference = json.loads(
    open(f"{root}/neural_forge_tokenizer_bridge_reference.json", encoding="utf-8").read()
)
if reference.get("bridge_artifact_sha256") != artifact:
    sys.exit("Published child names a different bridge artifact.")
if reference.get("bridge_profile_sha256") != profile:
    sys.exit("Published child names a different bridge transport profile.")
print("Child bridge reference matches the sealed bridge.")
CHILD_BRIDGE_PREFLIGHT
  # A staged corpus that drifted from the reviewed one would silently change
  # what the round fits.
  echo "$MANUAL_PROMPTS_SHA256  $PROMPTS" | sha256sum -c -
  echo "$FIT_PROMPTS_SHA256  $FIT_PROMPTS" | sha256sum -c -
  echo "$MANUAL_DIGESTS_SHA256  $MANUAL_DIGESTS" | sha256sum -c -
  # The round is only comparable with the base corpus if it reproduces this
  # selection; the runner re-proves it, and reading it here is free.  Both the
  # sealed cache and the W12E recapture are read: the recapture is what the
  # round captures against, and the sealed cache is what the recapture claims
  # to be a faithful copy of, so a drift in either is worth one refusal here.
  "$VENV_PYTHON" - "$BASE_CAPTURE_SELECTION_SHA256" "$BASE_CACHE" "$ROUND_BASE_CACHE" <<'BASE_SELECTION_PREFLIGHT' || exit 46
import json
import sys

expected = sys.argv[1]
corpora = []
for cache in sys.argv[2:]:
    manifest = json.loads(open(f"{cache}/manifest.json", encoding="utf-8").read())
    selection = manifest["capture"]["selection_sha256"]
    if selection != expected:
        sys.exit(f"{cache} capture selection is {selection}, not {expected}.")
    execution = manifest["execution"]
    print(f"Base capture selection {selection} in {cache}")
    print(f"Base execution {execution['execution_sha256']} "
          f"({execution['attention_backend']} on {execution['input_device_type']}) "
          f"implementation {manifest['implementation']['implementation_sha256']}")
    if execution["input_device_type"] != "cuda":
        sys.exit("Base corpus was not captured on CUDA; the round device is wrong.")
    corpora.append(
        (manifest["corpus"]["corpus_sha256"], manifest["parent"]["identity_sha256"])
    )
if len(set(corpora)) != 1:
    sys.exit(f"The sealed and recaptured base corpora are not the same evidence: {corpora}.")
BASE_SELECTION_PREFLIGHT
  echo "Preflight OK."
}

case "$STAGE" in
preflight)
  preflight
  ;;

round)
  preflight
  if [ -e "$ROUND_ROOT/round-000" ]; then
    echo "Refusing to reuse existing round directory: $ROUND_ROOT/round-000" >&2
    exit 40
  fi
  mkdir -p "$ROUND_ROOT"
  printf '%s\n' "$actual_source_commit" > "$ROUND_ROOT/source-commit.txt"
  # One bridged round, on one 3090, against the child r12 published.  The
  # rollout is child-only; the parent is executed exactly once per surviving
  # batch inside the single sanctioned capture.  max-new-tokens is 8 because
  # the sealed base capture budget is max_decode_steps=8 and a longer
  # continuation would have to be truncated to stay comparable.
  nohup "$VENV_PYTHON" -u \
    "$RUNTIME/experiments/run_r12_on_policy_round.py" \
    --child-artifact-directory "$CHILD" \
    --parent-model "$PARENT" \
    --tokenizer-directory "$DONOR" \
    --target-tokenizer-directory "$BRIDGE" \
    --tokenizer-bridge-artifact-sha256 "$BRIDGE_ARTIFACT_SHA256" \
    --tokenizer-bridge-profile-sha256 "$BRIDGE_PROFILE_SHA256" \
    --base-cache-directory "$ROUND_BASE_CACHE" \
    --fit-prompts "$FIT_PROMPTS" \
    --manual-prompt-manifest "$MANUAL_DIGESTS" \
    --round-root "$ROUND_ROOT" \
    --round-index 0 \
    --max-rounds 2 \
    --max-prompts 128 \
    --max-new-tokens 8 \
    --truncate-forced-decode-to-capture-budget \
    --screen-cached-decode-admissibility \
    --min-cached-decode-admissible-batches 100 \
    --max-context-tokens 512 \
    --max-capture-batches 256 \
    --rollout-seed 0 \
    --device cuda:0 \
    --child-dtype float32 \
    --parent-dtype float32 \
    --parent-name parent \
    --receipt "$ROUND_ROOT/round-000-runner-receipt.json" \
    > "$ROUND_ROOT/round.log" 2>&1 &
  pid=$!
  echo "$pid" > "$ROUND_ROOT/round.pid"
  echo "ROUND_ROOT=$ROUND_ROOT"
  echo "PID=$pid"
  ;;

refold)
  preflight
  if [ ! -f "$ROUND_MANIFEST" ]; then
    echo "No sealed round manifest at $ROUND_MANIFEST; run the round stage" >&2
    exit 47
  fi
  # A round captured parent-canonically has no target-token replay, so the
  # bridged lane refuses it by name.  Reading the sealed block here turns that
  # into a one-line refusal instead of a configuration-time surprise after the
  # parent and the pack are already loaded.
  "$VENV_PYTHON" - "$ROUND_MANIFEST" "$BRIDGE_ARTIFACT_SHA256" <<'ROUND_PREIMAGE_PREFLIGHT' || exit 48
import json
import sys

path, artifact = sys.argv[1:3]
manifest = json.loads(open(path, encoding="utf-8").read())
preimage = manifest.get("target_preimage")
if preimage is None:
    sys.exit("Round manifest seals no target-token preimage.")
if preimage.get("bridge_artifact_sha256") != artifact:
    sys.exit("Round preimage names a different bridge artifact.")
print(f"Round {manifest['round']['index']} carries a target-token preimage "
      f"for {manifest['extension_cache']['batch_count']} batches "
      f"({preimage['parent_cache_input_mode']}).")
ROUND_PREIMAGE_PREFLIGHT
  # W13D: the refold's base evidence is only coherent if the rebound envelopes
  # name ROUND_BASE_CACHE and carry the sealed pack's own parent responses.
  # Both are one file read each and both are much cheaper to refuse here than
  # after the parent and the pack are loaded.
  echo "$REBOUND_PARENT_OUTPUTS_SHA256  $REBOUND/canonical_parent_outputs_bridge_bound.json" | sha256sum -c -
  echo "$REBOUND_MAPPED_PARENT_OUTPUTS_SHA256  $REBOUND/mapped_parent_outputs.json" | sha256sum -c -
  "$VENV_PYTHON" - "$ROUND_BASE_CACHE" "$REBOUND" "$PACK" <<'REBOUND_ENVELOPE_PREFLIGHT' || exit 49
import hashlib
import json
import sys

cache, rebound, pack = sys.argv[1:4]
expected = json.loads(
    open(f"{cache}/manifest.json", encoding="utf-8").read()
)["cache_key"]
for name in (
    "canonical_parent_outputs_bridge_bound.json",
    "mapped_parent_outputs.json",
):
    body = json.loads(open(f"{rebound}/{name}", encoding="utf-8").read())
    if body["parent_cache_key"] != expected:
        sys.exit(
            f"{name} names parent_cache_key {body['parent_cache_key']}, "
            f"not the refold's base cache {expected}."
        )
    print(f"{name} binds parent_cache_key {expected}")
for envelope, rows in (
    ("canonical_parent_outputs_bridge_bound.json", "canonical_parent_rows.json"),
    ("mapped_parent_outputs.json", "mapped_parent_rows.json"),
):
    sealed = json.loads(open(f"{pack}/{envelope}", encoding="utf-8").read())
    rebound_body = json.loads(open(f"{rebound}/{envelope}", encoding="utf-8").read())
    if sealed["rows"] != rebound_body["rows"]:
        sys.exit(f"{envelope} rows are not the sealed pack's parent responses.")
    sealed_rows = hashlib.sha256(open(f"{pack}/{rows}", "rb").read()).hexdigest()
    copied_rows = hashlib.sha256(open(f"{rebound}/{rows}", "rb").read()).hexdigest()
    if sealed_rows != copied_rows:
        sys.exit(f"{rows} differs from the sealed pack copy.")
    print(f"{envelope} carries the sealed pack rows ({rows} {sealed_rows})")
REBOUND_ENVELOPE_PREFLIGHT
  if [ -e "$RUN" ]; then
    echo "Refusing to reuse existing run directory: $RUN" >&2
    exit 40
  fi
  mkdir -p "$RUN"
  printf '%s\n' "$actual_source_commit" > "$RUN/source-commit.txt"
  sha256sum "$PROMPTS" > "$RUN/manual-prompts.sha256"
  sha256sum "$ROUND_MANIFEST" > "$RUN/consumed-round-manifest.sha256"

  # This is r12's flag set with three classes of difference: the destinations
  # are the sibling -round-0-refold run and the sealed round manifest is
  # consumed with its bounded budget stated; the base evidence is the W12E
  # recapture with the W13D rebound envelopes; and the r13 reclosure-acceptance,
  # terminal-floor, telemetry and install-verification flags are composed in
  # (see the header).  Same parent, same corpus, same blank seed, same topology,
  # same counted budget, same carrier package.  Consuming extension evidence
  # enters the derived phase-cache keys, so the global-topology and
  # balanced-basis phases recompute for this evidence identity instead of
  # reusing r12's entries.
  nohup "$VENV_PYTHON" -u \
    "$RUNTIME/experiments/run_lfm2_dynamic_catalog.py" \
    --parent-model "$PARENT" \
    --tokenizer-directory "$DONOR" \
    --target-tokenizer-directory "$BRIDGE" \
    --target-tokenizer-calibration "$PACK/target_tokenized_calibration.json" \
    --tokenizer-bridge-artifact-sha256 "$BRIDGE_ARTIFACT_SHA256" \
    --tokenizer-bridge-profile-sha256 "$BRIDGE_PROFILE_SHA256" \
    --tokenizer-bridge-mapped-parent-outputs "$REBOUND/mapped_parent_outputs.json" \
    --tokenizer-bridge-mapped-parent-outputs-sha256 "$REBOUND_MAPPED_PARENT_OUTPUTS_SHA256" \
    --tokenizer-bridge-minimum-parent-task-retention 0 \
    --calibration "$PACK/mapped_donor_calibration.json" \
    --sealed-parent-cache-directory "$ROUND_BASE_CACHE" \
    --cached-parent-outputs "$REBOUND/canonical_parent_outputs_bridge_bound.json" \
    --cached-parent-outputs-sha256 "$REBOUND_PARENT_OUTPUTS_SHA256" \
    --on-policy-round-manifest "$ROUND_MANIFEST" \
    --on-policy-max-rounds 2 \
    --carrier-route-publication "$CARRIER" \
    --carrier-route-artifact-sha256 860bccfe6be46e543f67c8514f3f3ff7dc0d729eb82267f66a9ce1766e43d7bb \
    --carrier-span-evidence "$SPAN" \
    --carrier-span-evidence-file-sha256 082ef4c782b818f31f72629c71de576cef1e1e5a1f9dcaea28bd2287dec9e6f2 \
    --carrier-span-evidence-sha256 134f1ccb0ac2b2b69829ea483263bb4d92c1783e9b809f5c4a87c4ff10528cf0 \
    --semantic-source-tokenizer-directory "$DONOR" \
    --routed-manual-prompts "$PROMPTS" \
    --routed-manual-prompts-sha256 "$MANUAL_PROMPTS_SHA256" \
    --routed-manual-dossier "$RUN/routed-manual-dossier.json" \
    --routed-manual-checkpoint-directory "$RUN/routed-manual-checkpoints" \
    --counted-target-total-parameters 57420144 \
    --counted-dense-body-parameters 53552368 \
    --carrier-expected-target-rows 4096 \
    --carrier-expected-hidden-size 816 \
    --routed-smoke-maximum-prompt-tokens 512 \
    --routed-smoke-max-new-tokens 2 \
    --routed-manual-maximum-prompt-tokens 8192 \
    --routed-manual-initial-max-new-tokens 1024 \
    --routed-manual-retry-max-new-tokens 2048 \
    --cache-root "$RUN/cache" \
    --derived-global-topology-cache-root "$DERIVED" \
    --balanced-basis-phase-cache-root "$BALANCED_BASIS" \
    --balanced-basis-phase-event-log "$BALANCED_BASIS_EVENTS" \
    --balanced-basis-phase-hmac-key-file "$BALANCED_BASIS_HMAC_KEY" \
    --balanced-basis-phase-hmac-key-id "$BALANCED_BASIS_HMAC_KEY_ID" \
    --balanced-basis-existing-phase-key-sha256 52cbe3fd96650d284b2e2a03395515c3fb467ce846a014712f5d5c1a4e8f683d \
    --snapshot-root "$RUN/snapshots" \
    --publish-destination "$RUN/published-child" \
    --ledger "$RUN/ledger.jsonl" \
    --receipt "$RUN/receipt.json" \
    --diagnostics "$RUN/diagnostics.json" \
    --parent-name parent \
    --child-name "$CHILD_NAME" \
    --device cuda:0 \
    --dtype float32 \
    --compression-ratio 4 \
    --dense-minimum-hidden-size 32 \
    --required-target-hidden-size 816 \
    --dense-heterogeneous-layout \
    --dense-heterogeneous-global-hidden-frontier \
    --dense-heterogeneous-factor-writer-rank-discovery \
    --dense-heterogeneous-global-functional-rows-per-role 2048 \
    --dense-heterogeneous-target-parameters 53552368 \
    --dense-residual-basis-mode circuit_balanced \
    --dense-run-closed-loop \
    --dense-body-reclosure-mode moving_gauge_v3 \
    --dense-moving-gauge-writer-reclosure-acceptance require_boundary_improvement \
    --dense-moving-gauge-boundary-regression-tolerance 1e-6 \
    --dense-moving-gauge-install-verification fingerprint \
    --dense-closed-loop-replay-mode prefix_by_prefix_canonical \
    --dense-closed-loop-performance-metrics-diagnostic-only \
    --dense-closed-loop-merge-adjacent-layer-captures \
    --dense-closed-loop-terminal-composition-floor \
    --dense-closed-loop-terminal-regression-tolerance 0.25 \
    --dense-closed-loop-terminal-absolute-floor-minimum-layer 0 \
    --dense-closed-loop-terminal-free-run-tripwire \
    --dense-closed-loop-input-error-multiplier 0 \
    --dense-closed-loop-solver-conditioning-diagnostics \
    --dense-closed-loop-gate-operating-point-diagnostics \
    --dense-closed-loop-depth-telemetry \
    --dense-replay-devices cuda:0,cuda:4 \
    --dense-restart-capsule-root "$RUN/restart-capsules" \
    --dense-restart-capsule-keep 2 \
    "${RESUME_ARGS[@]}" \
    --dense-attention-consumer-mode coupled-value-output \
    --dense-attention-geometry-risk \
    --dense-vocabulary-margin-observability \
    --dense-streaming-vocabulary-margin-v2 \
    --dense-streaming-vocabulary-margin-pair-chunk-size 256 \
    --sample-last-tokens 256 \
    --max-decode-steps 8 \
    --max-capture-bytes 536870912 \
    --child-decode-trace-mode full_prefix_canonical \
    --capture-logit-rtol 0.2 \
    --capture-logit-atol 1 \
    --max-selection-rows 13728 \
    --max-fit-rows 40656 \
    --max-validation-rows 13199 \
    --max-endpoint-relative-rmse 1000000 \
    --min-endpoint-cosine-similarity -1 \
    --min-teacher-top1 0 \
    --min-free-run-match 0 \
    --min-residual-cosine -1 \
    --whole-output-max-new-tokens 64 \
    --whole-output-maximum-prompt-tokens 512 \
    --whole-output-fragment-minimum-characters 12 \
    --whole-output-maximum-new-empty-responses 28 \
    --whole-output-maximum-new-fragment-responses 28 \
    --whole-output-minimum-parent-nonempty-retention 0 \
    --whole-output-minimum-task-score-retention 0 \
    --whole-output-maximum-parent-task-score-regressions 28 \
    --whole-output-maximum-new-generation-limit-saturations 28 \
    --whole-output-maximum-repeated-ngram-share-regressions 28 \
    --whole-output-maximum-repeated-bigram-share-delta 1 \
    --final-performance-metrics-diagnostic-only \
    > "$RUN/run.log" 2>&1 &

  pid=$!
  echo "$pid" > "$RUN/run.pid"
  echo "RUN=$RUN"
  echo "PID=$pid"
  ;;
esac
