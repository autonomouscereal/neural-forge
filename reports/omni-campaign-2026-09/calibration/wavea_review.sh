#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# wavea_review.sh -- the Wave A REVIEW queue for ONE registered fold child.
#
# G0-4 of CAMPAIGN.md section 2.1, executed rather than described: bank v1 with
# `--thinking off`, the five bank-v2 text lanes, `oc_assess.py` with NO
# `--per-case` (so both columns come out as UPPER BOUNDS), the review record,
# the package, and the dashboards -- after which `REVIEW_QUEUE.md` names the
# child and Cyril has something to read.
#
# THE ONE THING THIS SCRIPT DOES NOT DO IS GRADE.  Every number it moves into
# the ledger is a ceiling produced by `oc_review_upper_bound.py`, and that
# file's docstring is the specification of the translation.  The short version:
# a machine can prove a lane is NOT holding a case (it emitted nothing, looped,
# or rode the cap saying nothing) and can never prove that it is, so
# `canonical_content_miss` is always 0 here -- which means nothing this script
# writes can make a checkpoint `bad`, and therefore nothing it writes can make a
# checkpoint deletable.  The package carries UPPER-BOUND-NOTICE-<ID>.md and the
# review JSON itself, so the ceiling is stated in the package and not only in a
# notes column.
#
# ORDERING IS NOT A STYLE CHOICE.  `ledger/README.md` is explicit: a package's
# digest only exists once the package is written, and a review may not name a
# `package_path` without a `package_sha256`.  So the review is recorded TWICE --
# once bare, then the package, then again carrying the digest -- and the second
# record supersedes the first without editing it, because records are
# append-only.  Steps 4, 5 and 5b below are that sequence.  Both records are
# DRY-RUN first and then `--confirm`ed.
#
# WHAT IT WILL NOT SUPERSEDE.  Records are append-only and the newest record is
# the checkpoint's status.  This script only ever writes its OWN machine bound
# on top of its OWN machine bound (step 5b adds the package digest to the record
# step 4b wrote).  If the head record is a `status` event (Cyril's ruling), or a
# review by any other reviewer, or a review that is not `is_upper_bound`, or a
# bound about a different dossier, the script REFUSES rather than re-computing
# a status from a ceiling on top of a reading.
#
# BOUNDARIES THIS SCRIPT HONOURS
#   * `qwen-fold/` belongs to the build lane.  The child's weights are READ from
#     it and nothing is ever written into it -- the review package goes to
#     `$ROOT/<wave>/human-review/<ID>` (R7: `OC/oc<NN>/human-review/<id>/`),
#     where <wave> is the `wave` field of the sealed register record.
#   * GPU0 is the build queue and is REFUSED outright.  A P40 is refused by name
#     (sm61 is unsupported by this torch build), and so is any card that is not
#     an RTX 3090: the parent lane was measured on `sm86 NVIDIA GeForce RTX
#     3090` and `--require-compute-architecture sm86` is passed to the runner,
#     so an RTX 8000 (sm75) would only fail later with a traceback.  The card
#     must be at or under --min-free-used MiB already in use, READ FROM THE SAME
#     nvidia-smi QUERY AS ITS NAME and validated as a number, or the run refuses
#     rather than measuring contention.
#   * Nothing on `/mnt` is touched, and nothing anywhere is deleted -- including
#     on failure.  A failed step leaves its partial log and stops.  The one move
#     the script makes is `oc_package.py build --overwrite` on a HALF-WRITTEN
#     package (non-empty folder, no MANIFEST.json, and the record names no
#     package digest): that flag MOVES the folder to `<name>.superseded-<utc>`
#     and deletes nothing.
#
# MECHANICS SETTINGS ARE NEVER RUN AGAINST THE PRODUCTION REGISTRY.
#   `--tiny-random`, `--max-cases` and `--max-new-tokens` change the SURFACE
#   (random weights, a truncated bank, a different token cap -- all elements of
#   the lane digest's validity tuple).  A record sealed from such a run is
#   append-only noise on a real child's chain, and the ledger itself does not
#   read `is_random_init`.  So any of the three requires an EXPLICIT `--root`
#   that (a) is not /home/cereal/omni-campaign, (b) does not share the
#   production registry, and (c) carries a `.throwaway-registry` marker file.
#   Their outputs also carry the surface in the file name
#   (`-cap<N>`, `-n<N>`, `-tiny`) and default to `<root>/calibration-out`, so a
#   mechanics dossier can never be picked up by a real run's SKIP path.
#
# IDEMPOTENCE.  Every step names its outputs and is SKIPPED when they already
# exist, are non-empty AND describe the surface this invocation would produce
# (model directory, cap, thinking, random-init flag, case count are read back
# out of the dossier and a mismatch is a refusal, not a skip).  The ledger steps
# check the record CHAIN rather than a file, and the record is re-read after
# every seal so a later guard sees what was actually written.
#
# Usage:
#   ./wavea_review.sh <CHECKPOINT_ID> [options]
#
#   --gpu N              PCI index of the card to run on (default 4; 0 refused)
#   --root DIR           campaign root for the ledger (default $OC_ROOT or
#                        /home/cereal/omni-campaign); REQUIRED, and must be a
#                        throwaway, with any mechanics setting
#   --lane DIR           the calibration lane (default: this script's directory)
#   --out DIR            where the child's dossiers/assessments/logs go
#                        (default $LANE/out; mechanics: $ROOT/calibration-out)
#   --parent-out DIR     where the PARENT artifacts are read from (default
#                        $LANE/out)
#   --parent-label L     parent v1 label, e.g. Q35-4B-NOTHINK-v2 (default: the
#                        `<parent_code>-NOTHINK-v2` files if present, else
#                        `<parent_code>-NOTHINK` with a logged NOTE; the tuple
#                        check below decides, not the file name)
#   --parent-b2-label L  parent bank-v2 label (default `b2-<parent_code>-v2` if
#                        present, else `b2-<parent_code>`)
#   --ledger-lane DIR    where oc_checkpoint.py / oc_package.py / oc_dashboard.py
#                        live (default $ROOT/ledger if it holds them, else the
#                        lane's sibling ../ledger)
#   --reviewer NAME      goes in the review record (default wavea_review.sh)
#   --min-free-used MiB  refuse above this many MiB already used (default 2000)
#   --max-cases N        MECHANICS: truncate both bank runs
#   --max-new-tokens N   MECHANICS: override BOTH token caps (v1 1024, v2 1024)
#   --tiny-random        MECHANICS: shrunken RANDOM-INIT model of the child's
#                        class, decoded on the CPU; every artifact says
#                        is_random_init and grades nothing
#   --skip-bank2         do not run the bank v2 text lanes
#
# PARENT SURFACE.  `bank1.parent_useful_or_partial` is sealed from the parent's
# NOTHINK assessment, so before anything is sealed the child's v1 dossier and
# the parent's are compared element by element on `lane_digest_validity`
# (bank digest, dtype, transformers version, cap, thinking, kernel-report
# schema, decode path, runtime, sm86) and the parent assessment is checked to
# be about that dossier and about this child's parent_code.  A real run DIES on
# any difference, naming the element -- that is G0-2's "run the parent again
# on the v2 tuple", enforced by content rather than by file name.  The bank-v2
# side-by-side is a sheet convenience with no sealed count behind it: on a
# tuple difference it is OMITTED with a NOTE, and the lanes that have no parent
# text in the parent B2 dossier (the lc-8k rungs live in a separate 300-cap
# dossier) are named in the log and in the UPPER-BOUND-NOTICE.
#
# Example (the real thing, once GPU4 is free and a child is registered):
#   ./wavea_review.sh OC01-F-Q35-4B-C2K2304 --gpu 4
# Example (mechanics, against a throwaway registry, no GPU):
#   ./wavea_review.sh OC01-F-Q35-4B-TINY1 --tiny-random --max-cases 2 \
#       --max-new-tokens 32 --root /path/to/throwaway
# ---------------------------------------------------------------------------
set -u

PRODUCTION_ROOT=/home/cereal/omni-campaign

ID="${1:-}"
if [ -z "$ID" ] || [ "$ID" = "-h" ] || [ "$ID" = "--help" ]; then
  sed -n '2,118p' "$0"
  exit 2
fi
shift

GPU=4
LANE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${OC_ROOT:-$PRODUCTION_ROOT}"
ROOT_EXPLICIT=0
OUT=""
PARENT_OUT=""
PARENT_LABEL=""
PARENT_B2_LABEL=""
LEDGER_LANE="${LEDGER_LANE:-}"
REVIEWER="wavea_review.sh"
MIN_FREE_USED=2000
MAX_CASES=""
MAX_NEW_TOKENS=""
TINY=0
SKIP_B2=0

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)             GPU="${2:?--gpu needs an index}"; shift 2 ;;
    --root)            ROOT="${2:?--root needs a directory}"; ROOT_EXPLICIT=1; shift 2 ;;
    --lane)            LANE="${2:?--lane needs a directory}"; shift 2 ;;
    --out)             OUT="${2:?--out needs a directory}"; shift 2 ;;
    --parent-out)      PARENT_OUT="${2:?--parent-out needs a directory}"; shift 2 ;;
    --parent-label)    PARENT_LABEL="${2:?--parent-label needs a label}"; shift 2 ;;
    --parent-b2-label) PARENT_B2_LABEL="${2:?--parent-b2-label needs a label}"; shift 2 ;;
    --ledger-lane)     LEDGER_LANE="${2:?--ledger-lane needs a directory}"; shift 2 ;;
    --reviewer)        REVIEWER="${2:?--reviewer needs a name}"; shift 2 ;;
    --min-free-used)   MIN_FREE_USED="${2:?--min-free-used needs a number}"; shift 2 ;;
    --max-cases)       MAX_CASES="${2:?--max-cases needs a number}"; shift 2 ;;
    --max-new-tokens)  MAX_NEW_TOKENS="${2:?--max-new-tokens needs a number}"; shift 2 ;;
    --tiny-random)     TINY=1; shift ;;
    --skip-bank2)      SKIP_B2=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

now () { date -u +%Y-%m-%dT%H:%M:%SZ; }
# Before the log exists, `say`/`die` go to the terminal only; once LOG is set
# they also append to it.
LOG=""
say () { if [ -n "$LOG" ]; then printf '[%s] %s\n' "$(now)" "$*" | tee -a "$LOG"; else printf '[%s] %s\n' "$(now)" "$*"; fi; }
die () { if [ -n "$LOG" ]; then printf '[%s] FAILED: %s\n' "$(now)" "$1" | tee -a "$LOG" >&2; else printf '[%s] FAILED: %s\n' "$(now)" "$1" >&2; fi; exit "${2:-1}"; }
#: Every line another tool prints gets its own UTC stamp, so the log can be read
#: as a timeline rather than as a pile of interleaved CLI output.
stamped () { while IFS= read -r line; do printf '[%s]   | %s\n' "$(now)" "$line"; done; }
# A step's output is present AND non-empty.  An empty file is a crashed write,
# not a finished step, and skipping one would hide the crash.
have () { for f in "$@"; do [ -s "$f" ] || return 1; done; return 0; }

# ---------------------------------------------------------------------------
# the interpreter.  The campaign's python is the model-surgery venv; anything
# else has no torch/transformers and would only fail at step 1 after step 0
# had already run.  PY= is the explicit override for a CPU-only box.
# ---------------------------------------------------------------------------
PY="${PY:-/home/cereal/model-surgery-lab/venv/bin/python}"
[ -x "$PY" ] || die "$PY is not executable; set PY=<python with torch+transformers> explicitly" 3

# ---------------------------------------------------------------------------
# mechanics guard.  Any surface-changing setting needs an explicit throwaway
# root; see the header.  Checked BEFORE anything is written anywhere.
# ---------------------------------------------------------------------------
MECHANICS=0
if [ "$TINY" -eq 1 ] || [ -n "$MAX_CASES" ] || [ -n "$MAX_NEW_TOKENS" ]; then
  MECHANICS=1
fi
[ -d "$ROOT" ] || die "--root $ROOT is not a directory" 2
ROOT_REAL="$(cd "$ROOT" && pwd -P)"
if [ "$MECHANICS" -eq 1 ]; then
  [ "$ROOT_EXPLICIT" -eq 1 ] \
    || die "mechanics settings (--tiny-random / --max-cases / --max-new-tokens) may not be run against the default registry; pass --root <throwaway>" 2
  case "$ROOT_REAL" in
    "$PRODUCTION_ROOT")
      die "mechanics settings may not be run against the production registry ($ROOT_REAL); pass --root <throwaway>" 2 ;;
  esac
  if [ -e "$PRODUCTION_ROOT/registry" ] && [ "$ROOT_REAL/registry" -ef "$PRODUCTION_ROOT/registry" ]; then
    die "mechanics settings may not be run against the production registry ($ROOT_REAL/registry IS $PRODUCTION_ROOT/registry)" 2
  fi
  [ -f "$ROOT_REAL/.throwaway-registry" ] \
    || die "mechanics settings need a --root that carries a .throwaway-registry marker file (touch $ROOT_REAL/.throwaway-registry if this root really is disposable)" 2
fi
ROOT="$ROOT_REAL"
REGISTRY="$ROOT/registry"
LEDGER_FILE="$ROOT/ledger.jsonl"

if [ -z "$LEDGER_LANE" ]; then
  if [ -f "$ROOT/ledger/oc_checkpoint.py" ]; then
    LEDGER_LANE="$ROOT/ledger"
  else
    LEDGER_LANE="$(cd "$LANE/.." && pwd)/ledger"
  fi
fi
for tool in oc_checkpoint.py oc_package.py oc_dashboard.py; do
  [ -f "$LEDGER_LANE/$tool" ] || die "$LEDGER_LANE/$tool is missing; pass --ledger-lane" 3
done
for tool in oc_run_frozen_bank.py oc_assess.py oc_review_upper_bound.py; do
  [ -f "$LANE/$tool" ] || die "$LANE/$tool is missing; pass --lane" 3
done

if [ -z "$OUT" ]; then
  if [ "$MECHANICS" -eq 1 ]; then OUT="$ROOT/calibration-out"; else OUT="$LANE/out"; fi
fi
[ -n "$PARENT_OUT" ] || PARENT_OUT="$LANE/out"
if [ "$MECHANICS" -eq 1 ]; then LOGS="$OUT/logs"; else LOGS="$LANE/logs"; fi
BANKS="$LANE/banks"
V1="$BANKS/lfm2_semantic_whole_response_prompts_v1.json"
V1_SHA=0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b
V2="$BANKS/omni_review_bank_v2_draft.json"
PROBE="$BANKS/matrix_campaign_probe_prompts_v1.json"
SMOKE="$BANKS/matrix_campaign_smoke_prompts_v1.json"
B2_LANES="code-rle-encode-A,reason-arith-two-step-A,lc-8k-notice-select-A,lc-8k-notice-select-B,mt-dietary-correction-B"

#: The token cap is an element of `lane_digest_validity`, so lowering it makes a
#: DIFFERENT SURFACE rather than a faster version of the same one.  Both caps
#: are 1024 because the parent's B2 dossier (run_r6.sh) was decoded at 1024 and
#: the side-by-side is only a side-by-side on one surface.
NEWTOK_V1=1024
NEWTOK_V2=1024
if [ -n "$MAX_NEW_TOKENS" ]; then
  NEWTOK_V1="$MAX_NEW_TOKENS"
  NEWTOK_V2="$MAX_NEW_TOKENS"
fi
#: The surface travels in the file name, so a mechanics artifact and a real one
#: can never share a path.
SFX=""
[ -n "$MAX_NEW_TOKENS" ] && SFX="$SFX-cap$MAX_NEW_TOKENS"
[ -n "$MAX_CASES" ] && SFX="$SFX-n$MAX_CASES"
[ "$TINY" -eq 1 ] && SFX="$SFX-tiny"

mkdir -p "$OUT" "$LOGS" || exit 1
LOG="$LOGS/review-$ID$SFX.log"

say "=================================================================="
say "wavea_review.sh  checkpoint=$ID  gpu=$GPU"
say "root       $ROOT"
say "registry   $REGISTRY   (every seal in this run lands HERE)"
say "ledger     $LEDGER_FILE"
say "ledger lane $LEDGER_LANE"
say "lane       $LANE   out $OUT   parent artifacts $PARENT_OUT"
say "python     $PY"
say "tiny_random=$TINY  max_cases=${MAX_CASES:-all}  caps v1=$NEWTOK_V1 v2=$NEWTOK_V2  skip_bank2=$SKIP_B2"
if [ "$MECHANICS" -eq 1 ]; then
  say "MECHANICS RUN against throwaway root $ROOT (marker .throwaway-registry present)."
  say "  The surface differs from the parent's (cap / case count / random init),"
  say "  every artifact name carries '$SFX', and NOTHING here is a review anyone"
  say "  is meant to read."
fi
say "=================================================================="

# ---------------------------------------------------------------------------
# step 0 -- the child, FROM THE LEDGER.  The build lane's naming convention says
# OC01-F-Q35-4B-<TAG> lives at qwen-fold/children/QF-<TAG>, but a convention is
# not a record: the sealed register record names the directory that was actually
# digested, and that is the only directory a review may read.  The convention is
# checked against it and a disagreement is REPORTED, not resolved.
#
# `resolve_child` is a function because the head record CHANGES during the run
# (steps 4b and 5b seal new records) and every later guard must look at what
# was actually written, not at what was there when the script started.
# ---------------------------------------------------------------------------
RESOLVE_JSON="$OUT/.resolve-$ID$SFX.json"
RESOLVE_SH="$OUT/.resolve-$ID$SFX.sh"
resolve_child () {
  if ! timeout 300 "$PY" "$LEDGER_LANE/oc_checkpoint.py" --root "$ROOT" show \
          --checkpoint-id "$ID" > "$RESOLVE_JSON" 2>>"$LOG"; then
    die "oc_checkpoint.py show --checkpoint-id $ID refused; see $LOG" 3
  fi
  timeout 120 "$PY" - "$RESOLVE_JSON" "$ROOT" > "$RESOLVE_SH" <<'PYRESOLVE'
import json, shlex, sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
cp = record["checkpoint"]
review = record.get("review") or {}
tag = cp.get("tag") or ""
directory = cp.get("directory") or ""
convention = str(root / "qwen-fold" / "children" / f"QF-{tag}")
out = {
    "CHILD_DIR": directory,
    "CHILD_TAG": tag,
    "CHILD_STAGE": str(cp.get("stage")),
    "CHILD_PARENT_CODE": str(cp.get("parent_code")),
    "CHILD_WAVE": str(cp.get("wave") or ""),
    "CHILD_STATUS": str(record.get("status")),
    "CHILD_RATIO": str(cp.get("net_ratio")),
    "CHILD_CONVENTION": convention,
    "CHILD_CONVENTION_MATCHES": "yes" if directory == convention else "no",
    "CHILD_HEAD_EVENT": str(record.get("event")),
    "CHILD_HEAD_RECORD": str(record.get("record_instance_id")),
    "CHILD_REVIEWED": "yes" if review else "no",
    "CHILD_REVIEWER": str(review.get("reviewer") or ""),
    "CHILD_REVIEW_IS_BOUND": "yes" if review.get("is_upper_bound") else "no",
    "CHILD_REVIEW_IS_RANDOM_INIT": "yes" if review.get("is_random_init") else "no",
    "CHILD_REVIEW_DOSSIER_SHA": str(review.get("dossier_sha256") or ""),
    "CHILD_PACKAGED": "yes" if review.get("package_sha256") else "no",
    "CHILD_PACKAGE_PATH": str(review.get("package_path") or ""),
    "CHILD_PACKAGE_SHA": str(review.get("package_sha256") or ""),
}
for key, value in out.items():
    print(f"{key}={shlex.quote(value)}")
PYRESOLVE
  local rc=$?
  [ $rc -eq 0 ] || die "could not read the sealed record out of $RESOLVE_JSON (exit $rc)" 3
  # shellcheck disable=SC1090
  . "$RESOLVE_SH" || die "the resolved record did not parse as shell assignments" 3
  # The head record is this script's own bound exactly when all four hold; any
  # other head is a reading or a ruling and is never superseded from here.
  HEAD_IS_OUR_BOUND=no
  if [ "$CHILD_REVIEWED" = "yes" ] && [ "$CHILD_HEAD_EVENT" = "review" ] \
     && [ "$CHILD_REVIEW_IS_BOUND" = "yes" ] && [ "$CHILD_REVIEWER" = "$REVIEWER" ]; then
    HEAD_IS_OUR_BOUND=yes
  fi
}

say "--- step 0: resolve $ID from the ledger ---"
resolve_child
say "    directory        $CHILD_DIR"
say "    tag / stage      $CHILD_TAG / $CHILD_STAGE   parent $CHILD_PARENT_CODE   wave $CHILD_WAVE"
say "    status / ratio   $CHILD_STATUS / $CHILD_RATIO"
say "    convention       $CHILD_CONVENTION  (matches: $CHILD_CONVENTION_MATCHES)"
say "    head record      $CHILD_HEAD_RECORD  event=$CHILD_HEAD_EVENT"
say "    already reviewed $CHILD_REVIEWED (reviewer='$CHILD_REVIEWER' bound=$CHILD_REVIEW_IS_BOUND)   packaged $CHILD_PACKAGED"
[ "$CHILD_STAGE" = "fold" ] || die "$ID is stage '$CHILD_STAGE', not 'fold'; this queue reviews fold children" 3
[ -n "$CHILD_WAVE" ] || die "the sealed record carries no 'wave'; the package location \$ROOT/<wave>/human-review/ cannot be derived" 3
[ -d "$CHILD_DIR" ] || die "the sealed record names $CHILD_DIR and it is not a directory" 3
case "$CHILD_DIR" in
  /mnt/*) die "the record names a directory on the CIFS share; this lane never walks /mnt" 3 ;;
esac
if [ "$CHILD_CONVENTION_MATCHES" != "yes" ]; then
  say "    NOTE: the record's directory is not the QF-<TAG> convention path.  The"
  say "          RECORD wins -- it is what was digested at registration -- and the"
  say "          disagreement is logged here rather than silently resolved."
fi
if [ "$CHILD_REVIEWED" = "yes" ] && [ "$HEAD_IS_OUR_BOUND" = "no" ]; then
  say "    NOTE: the head record is NOT this script's own bound (event=$CHILD_HEAD_EVENT,"
  say "          reviewer='$CHILD_REVIEWER', is_upper_bound=$CHILD_REVIEW_IS_BOUND)."
  say "          Nothing in this run will supersede it."
fi
#: R7: `OC/oc<NN>/human-review/<checkpoint id>/`; the wave comes from the record.
PKG_ROOT="$ROOT/$CHILD_WAVE/human-review"
PKG="$PKG_ROOT/$ID"

# ---------------------------------------------------------------------------
# the parent artifacts, named from the RECORD's parent_code -- never hardwired
# to Q35-4B, because G0-11 runs this same loop on the 0.8B.
# ---------------------------------------------------------------------------
if [ -z "$PARENT_LABEL" ]; then
  if have "$PARENT_OUT/oc-dossier-$CHILD_PARENT_CODE-NOTHINK-v2.json"; then
    PARENT_LABEL="$CHILD_PARENT_CODE-NOTHINK-v2"
  else
    PARENT_LABEL="$CHILD_PARENT_CODE-NOTHINK"
    say "    NOTE: no oc-dossier-$CHILD_PARENT_CODE-NOTHINK-v2.json in $PARENT_OUT; using the"
    say "          un-suffixed $PARENT_LABEL files.  Whether they carry the v2 tuple is"
    say "          decided by the tuple check below, not by the file name."
  fi
fi
if [ -z "$PARENT_B2_LABEL" ]; then
  if have "$PARENT_OUT/oc-dossier-b2-$CHILD_PARENT_CODE-v2.json"; then
    PARENT_B2_LABEL="b2-$CHILD_PARENT_CODE-v2"
  else
    PARENT_B2_LABEL="b2-$CHILD_PARENT_CODE"
  fi
fi
PARENT_DOSSIER="$PARENT_OUT/oc-dossier-$PARENT_LABEL.json"
PARENT_ASSESSMENT="$PARENT_OUT/assessment-$PARENT_LABEL.json"
PARENT_DOSSIER_B2="$PARENT_OUT/oc-dossier-$PARENT_B2_LABEL.json"
say "    parent v1  $PARENT_DOSSIER"
say "    parent v1  $PARENT_ASSESSMENT"
say "    parent b2  $PARENT_DOSSIER_B2"

# ---------------------------------------------------------------------------
# the card.  A review taken while another process owns the card measures the
# contention, not the child (run_r7.sh's own refusal, kept).
# ---------------------------------------------------------------------------
DEVICE=""
if [ "$TINY" -eq 1 ]; then
  RUN_DEVICE=cpu
  DTYPE=float32
  ARCH_ARGS=""
  say "--- GPU gate: SKIPPED (--tiny-random decodes on the CPU) ---"
else
  [ "$GPU" != "0" ] || die "GPU0 is the build queue and is never used by this script" 3
  GPUINFO=$(timeout 120 nvidia-smi --query-gpu=index,name,memory.used,uuid \
            --format=csv,noheader,nounits -i "$GPU" 2>>"$LOG") \
    || die "nvidia-smi could not read GPU $GPU" 3
  GPUNAME=$(printf '%s' "$GPUINFO" | cut -d, -f2 | sed 's/^ *//;s/ *$//')
  # memory.used comes out of the SAME query as the name, so the gate reads one
  # instant, and it is validated as a number before it is compared: an empty
  # or non-numeric value used to make `[ -gt ]` fail, which `if` read as
  # "not busy".
  GPUUSED=$(printf '%s' "$GPUINFO" | cut -d, -f3 | tr -dc '0-9')
  say "--- GPU gate: $GPUINFO ---"
  case "$GPUUSED" in
    ''|*[!0-9]*) die "nvidia-smi returned no numeric memory.used for GPU $GPU from: '$GPUINFO'" 3 ;;
  esac
  case "$GPUNAME" in
    *P40*) die "GPU $GPU is a $GPUNAME; sm61 is unsupported by this torch build" 3 ;;
    *3090*) : ;;
    *) die "GPU $GPU is a '$GPUNAME'; the parent lane was measured on 'sm86 NVIDIA GeForce RTX 3090' and the runner is told --require-compute-architecture sm86, so only an RTX 3090 is accepted here" 3 ;;
  esac
  if [ "$GPUUSED" -gt "$MIN_FREE_USED" ]; then
    say "    busy: ${GPUUSED} MiB used (limit ${MIN_FREE_USED})"
    timeout 120 nvidia-smi --query-compute-apps=pid,used_memory,process_name \
      --format=csv,noheader 2>&1 | stamped | tee -a "$LOG"
    die "GPU $GPU is not free; a review taken under contention is not evidence about this child" 3
  fi
  say "    free: ${GPUUSED} MiB used (limit ${MIN_FREE_USED}); $GPUNAME"
  DEVICE="$GPU"
  RUN_DEVICE=cuda:0     # cuda:0 IS GPU $GPU once CUDA_VISIBLE_DEVICES masks it
  DTYPE=bfloat16
  ARCH_ARGS="--require-compute-architecture sm86"
fi

MAXCASE_ARGS=""
[ -n "$MAX_CASES" ] && MAXCASE_ARGS="--max-cases $MAX_CASES"
TINY_ARGS=""
ASSESS_TINY=""
if [ "$TINY" -eq 1 ]; then
  TINY_ARGS="--tiny-random"
  ASSESS_TINY="--allow-random-init"
fi

#: One place decides how a bank run is launched, so the two runs cannot drift
#: apart in device, dtype, prompt files or masking.
run_bank () {
  local step="$1" out="$2" steplog="$3"; shift 3
  if [ "$TINY" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES="" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    timeout 5400 "$PY" "$LANE/oc_run_frozen_bank.py" \
      --model-dir "$CHILD_DIR" --label "$ID-$step$SFX" --stage fold \
      --parent-code "$CHILD_PARENT_CODE" --device "$RUN_DEVICE" --dtype "$DTYPE" \
      --probe-prompts "$PROBE" --smoke-prompts "$SMOKE" \
      $TINY_ARGS $MAXCASE_ARGS "$@" --out "$out" >>"$steplog" 2>&1
  else
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$DEVICE" \
    timeout 5400 "$PY" "$LANE/oc_run_frozen_bank.py" \
      --model-dir "$CHILD_DIR" --label "$ID-$step$SFX" --stage fold \
      --parent-code "$CHILD_PARENT_CODE" --device "$RUN_DEVICE" --dtype "$DTYPE" \
      $ARCH_ARGS --probe-prompts "$PROBE" --smoke-prompts "$SMOKE" \
      $MAXCASE_ARGS "$@" --out "$out" >>"$steplog" 2>&1
  fi
}

#: An existing dossier is only a finished step if it describes THIS surface.
#: The runner records what it decoded (model directory, cap, thinking, dtype,
#: random-init flag, case count); a dossier written under other settings is a
#: refusal, never a skip.
assert_surface () {
  local dossier="$1" kind="$2" expect_cases="$3"
  timeout 120 "$PY" - "$dossier" "$CHILD_DIR" "$kind" "$expect_cases" "$TINY" "$DTYPE" "$B2_LANES" <<'PYSURF' 2>&1 | stamped | tee -a "$LOG"
import json, sys
from pathlib import Path
path, child_dir, kind, expect, tiny, dtype, lanes = sys.argv[1:8]
d = json.loads(Path(path).read_text(encoding="utf-8"))
v = d.get("lane_digest_validity") or {}
g = d.get("generation") or {}
b = d.get("bank") or {}
problems = []
got_dir = str((d.get("provenance") or {}).get("model_directory"))
if Path(got_dir).resolve() != Path(child_dir).resolve():
    problems.append(f"provenance.model_directory {got_dir!r} != record directory {child_dir!r}")
cap = int(expect.split(":")[0])
if int(v.get("maximum_new_tokens", g.get("maximum_new_tokens", -1))) != cap:
    problems.append(f"maximum_new_tokens {v.get('maximum_new_tokens')} != {cap}")
if str(v.get("chat_template_thinking")) != "off":
    problems.append(f"chat_template_thinking {v.get('chat_template_thinking')!r} != 'off'")
if str(v.get("dtype")) != dtype:
    problems.append(f"dtype {v.get('dtype')!r} != {dtype!r}")
if bool(d.get("is_random_init")) != (tiny == "1"):
    problems.append(f"is_random_init {d.get('is_random_init')} but --tiny-random is {'set' if tiny == '1' else 'not set'}")
max_cases = expect.split(":")[1]
ids = [r.get("id") for r in (d.get("child_responses") or [])]
if kind == "v1":
    want = int(max_cases) if max_cases else int(b.get("cases_in_bank", -1))
    if int(b.get("cases_run", -2)) != want:
        problems.append(f"bank.cases_run {b.get('cases_run')} != {want} (cases_in_bank {b.get('cases_in_bank')}, --max-cases {max_cases or 'unset'})")
else:
    wanted = lanes.split(",")
    if max_cases:
        want = min(int(max_cases), len(wanted))
        if int(b.get("cases_run", -2)) != want:
            problems.append(f"bank.cases_run {b.get('cases_run')} != {want} under --max-cases {max_cases}")
    elif sorted(ids) != sorted(wanted):
        problems.append(f"child_responses ids {sorted(ids)} != the five B2 lanes {sorted(wanted)}")
if problems:
    print(f"SURFACE MISMATCH in {path}:")
    for p in problems:
        print(f"  - {p}")
    print("  this dossier was written under other settings; move it aside by hand or use --out; it is NOT reused")
    sys.exit(4)
print(f"surface verified: {Path(path).name} dossier_sha256 {str(d.get('dossier_sha256'))[:16]} cap {cap} thinking off dtype {dtype} random_init {bool(d.get('is_random_init'))} cases_run {b.get('cases_run')}")
PYSURF
  return "${PIPESTATUS[0]}"
}

#: The child's v1 dossier and the parent's must be ONE surface before a parent
#: count is sealed beside the child's.  Real run: any difference is a refusal
#: naming the element.  Mechanics run: the difference is expected (that is what
#: mechanics means) and is logged; the parent column is then omitted from the
#: sheet and the record's notes say so.
compare_tuples () {
  local child="$1" parent="$2" what="$3"
  timeout 120 "$PY" - "$child" "$parent" "$what" <<'PYTUPLE' 2>&1 | stamped | tee -a "$LOG"
import json, sys
from pathlib import Path
child, parent, what = sys.argv[1:4]
c = json.loads(Path(child).read_text(encoding="utf-8"))
p = json.loads(Path(parent).read_text(encoding="utf-8"))
cv, pv = c.get("lane_digest_validity") or {}, p.get("lane_digest_validity") or {}
keys = ["bank_file_sha256", "dtype", "transformers_version", "maximum_new_tokens",
        "chat_template_thinking", "linear_attention_kernel_report_schema",
        "decode_path", "runtime"]
diffs = []
for k in keys:
    if cv.get(k) != pv.get(k):
        diffs.append(f"{k}: child {cv.get(k)!r} vs parent {pv.get(k)!r}")
ca, pa = str(cv.get("compute_architecture")), str(pv.get("compute_architecture"))
if not (ca.startswith("sm86") and pa.startswith("sm86")):
    diffs.append(f"compute_architecture: child {ca!r} vs parent {pa!r} (both must be sm86)")
if cv.get("linear_attention_kernel_path") != pv.get("linear_attention_kernel_path"):
    print(f"NOTE {what}: linear_attention_kernel_path child {cv.get('linear_attention_kernel_path')!r} vs parent {pv.get('linear_attention_kernel_path')!r}")
print(f"{what}: parent {Path(parent).name} dossier_sha256 {str(p.get('dossier_sha256'))[:16]} label {p.get('label')} parent_code {p.get('parent_code')} "
      f"tuple: cap {pv.get('maximum_new_tokens')} thinking {pv.get('chat_template_thinking')} dtype {pv.get('dtype')} transformers {pv.get('transformers_version')} "
      f"kernel_schema {pv.get('linear_attention_kernel_report_schema')} arch {pa!r}")
if diffs:
    print(f"TUPLE DIFFERENCE {what}: the child's lane_digest_validity and the parent's are not one surface:")
    for d in diffs:
        print(f"  - {d}")
    sys.exit(4)
print(f"{what}: tuple matches on {len(keys) + 1} elements")
PYTUPLE
  return "${PIPESTATUS[0]}"
}

PARENT_V1_CHECKED=0
PARENT_V1_COMPARABLE=no
PARENT_DOSSIER_SHA=""
#: Called by every step that needs the parent v1 artifacts (3, 4, 5b) and never
#: by the ones that do not, so a finished child re-run to the dashboard step does
#: not die on a moved parent file.
need_parent_v1 () {
  [ "$PARENT_V1_CHECKED" -eq 1 ] && return 0
  PARENT_V1_CHECKED=1
  have "$PARENT_DOSSIER" || die "$PARENT_DOSSIER is absent.  The parent side-by-side and the tuple check need it: run the parent's --thinking off calibration (G0-2) first, or point --parent-out / --parent-label at it." 3
  have "$PARENT_ASSESSMENT" || die "$PARENT_ASSESSMENT is absent, so bank1.parent_useful_or_partial has no source.  It is never guessed: run the parent's NOTHINK assessment first." 3
  PARENT_DOSSIER_SHA=$(timeout 120 "$PY" - "$PARENT_DOSSIER" "$PARENT_ASSESSMENT" "$CHILD_PARENT_CODE" <<'PYPAR' 2>>"$LOG"
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
a = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
code = sys.argv[3]
if str(a.get("parent_code")) != code:
    print(f"parent assessment parent_code {a.get('parent_code')!r} != child's parent_code {code!r}", file=sys.stderr); sys.exit(4)
if str(d.get("parent_code")) != code:
    print(f"parent dossier parent_code {d.get('parent_code')!r} != child's parent_code {code!r}", file=sys.stderr); sys.exit(4)
if a.get("dossier_sha256") != d.get("dossier_sha256"):
    print(f"parent assessment is about dossier {str(a.get('dossier_sha256'))[:16]} but the parent dossier seals to {str(d.get('dossier_sha256'))[:16]}: not the same run", file=sys.stderr); sys.exit(4)
if d.get("is_random_init"):
    print("the parent dossier is RANDOM-INIT; it cannot be a side-by-side", file=sys.stderr); sys.exit(4)
print(d["dossier_sha256"])
PYPAR
  ) || die "the parent artifacts do not agree with each other or with this child ($PARENT_DOSSIER / $PARENT_ASSESSMENT); see $LOG" 3
  say "    parent v1 dossier_sha256 $PARENT_DOSSIER_SHA (assessment agrees; parent_code $CHILD_PARENT_CODE)"
  if compare_tuples "$D1" "$PARENT_DOSSIER" "parent-v1"; then
    PARENT_V1_COMPARABLE=yes
  elif [ "$MECHANICS" -eq 1 ]; then
    say "    NOTE: mechanics run -- the tuple difference above is expected; the parent"
    say "          column is OMITTED from the sheet and the record says so."
  else
    die "the parent v1 surface ($PARENT_DOSSIER) is not the child's surface; a parent count sealed beside it would compare two surfaces.  Re-run the parent on the current tuple (CAMPAIGN.md G0-2) and point --parent-label at it." 3
  fi
}

# ---------------------------------------------------------------------------
# step 1 -- bank v1, --thinking off.  ~11 minutes on a 3090 for a 4B-class child.
# ---------------------------------------------------------------------------
D1="$OUT/oc-dossier-$ID-NOTHINK$SFX.json"
say "--- step 1: bank v1 --thinking off -> $(basename "$D1") ---"
if have "$D1"; then
  assert_surface "$D1" v1 "$NEWTOK_V1:$MAX_CASES" || die "$D1 exists but is not this surface (see above); refusing to reuse it" 3
  say "    SKIP: $D1 already exists and describes this surface"
else
  L1="$OUT/wavea-v1-$ID$SFX.log"
  say "    running (log: $L1) ..."
  run_bank NOTHINK "$D1" "$L1" \
    --bank "$V1" --bank-sha256 "$V1_SHA" --thinking off --maximum-new-tokens "$NEWTOK_V1"
  rc=$?
  tail -3 "$L1" 2>/dev/null | stamped | tee -a "$LOG"
  [ $rc -eq 0 ] || die "bank v1 exited $rc; full log $L1" "$rc"
  assert_surface "$D1" v1 "$NEWTOK_V1:$MAX_CASES" || die "the dossier just written does not describe the requested surface" 3
  say "    OK"
fi
D1_SHA=$(timeout 120 "$PY" -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["dossier_sha256"])' "$D1" 2>>"$LOG") \
  || die "cannot read dossier_sha256 from $D1" 3
say "    child v1 dossier_sha256 $D1_SHA"

# ---------------------------------------------------------------------------
# step 2 -- the five bank v2 draft TEXT lanes.  The runner DOES support this:
# `--bank <v2 draft> --lanes <ids>` is exactly how run_r7.sh drove the
# long-context rungs, and `select_cases` refuses an id the bank does not carry.
# `--skip-smoke` because the coherence smoke belongs to the v1 surface and
# repeating it here would only spend tokens.
#
# The v2 bank is a DRAFT and may legitimately be re-cut, so its digest is pinned
# to the sidecar that travels with it rather than to a constant in this file: if
# the bank moves and the sidecar moves with it, the run proceeds and the dossier
# records the new digest; if the bank moves and the sidecar does not, the runner
# refuses, which is the case worth catching.
# ---------------------------------------------------------------------------
D2="$OUT/oc-dossier-$ID-B2$SFX.json"
if [ "$SKIP_B2" -eq 1 ]; then
  say "--- step 2: bank v2 text lanes -- SKIPPED (--skip-bank2) ---"
elif have "$D2"; then
  assert_surface "$D2" b2 "$NEWTOK_V2:$MAX_CASES" || die "$D2 exists but is not this surface (see above); refusing to reuse it" 3
  say "--- step 2: bank v2 text lanes -- SKIP: $D2 already exists and describes this surface ---"
else
  say "--- step 2: bank v2 text lanes ($B2_LANES) -> $(basename "$D2") ---"
  V2_SHA_ARGS=""
  if [ -s "$V2.sha256" ]; then
    V2_SHA=$(awk '{print $1; exit}' "$V2.sha256")
    V2_SHA_ARGS="--bank-sha256 $V2_SHA"
    say "    bank v2 pinned to its sidecar: $V2_SHA"
  else
    say "    NOTE: no $V2.sha256 sidecar; the v2 bank runs UNPINNED and the"
    say "          dossier will record digest_was_pinned false."
  fi
  L2="$OUT/wavea-b2-$ID$SFX.log"
  say "    running (log: $L2) ..."
  # shellcheck disable=SC2086
  run_bank B2 "$D2" "$L2" \
    --bank "$V2" $V2_SHA_ARGS --lanes "$B2_LANES" --text-only --skip-smoke \
    --thinking off --maximum-new-tokens "$NEWTOK_V2"
  rc=$?
  tail -3 "$L2" 2>/dev/null | stamped | tee -a "$LOG"
  [ $rc -eq 0 ] || die "bank v2 exited $rc; full log $L2" "$rc"
  assert_surface "$D2" b2 "$NEWTOK_V2:$MAX_CASES" || die "the v2 dossier just written does not describe the requested surface" 3
  say "    OK"
fi

# ---------------------------------------------------------------------------
# step 3 -- assess, WITHOUT --per-case.  `--per-case` is an INPUT: it is Cyril's
# sealed reading, and there is none yet, so both columns come out as UPPER
# BOUNDS and every canonical box reads READ-REQUIRED.  That is the point.
# ---------------------------------------------------------------------------
A1="$OUT/assessment-$ID-NOTHINK$SFX.json"
S1="$OUT/SCORING-SHEET-$ID-NOTHINK$SFX.md"
say "--- step 3: assess v1 (NO --per-case; upper-bound columns) ---"
if have "$A1" "$S1"; then
  say "    SKIP: $A1 and $S1 already exist"
else
  need_parent_v1
  PARENT_ARG=""
  if [ "$PARENT_V1_COMPARABLE" = "yes" ]; then
    PARENT_ARG="--parent-dossier $PARENT_DOSSIER"
    say "    side-by-side parent dossier: $(basename "$PARENT_DOSSIER") ($PARENT_DOSSIER_SHA)"
  else
    say "    NOTE: the parent v1 surface differs (mechanics run); the sheet gets no parent column."
  fi
  # shellcheck disable=SC2086
  timeout 3600 "$PY" "$LANE/oc_assess.py" --dossier "$D1" --bank "$V1" \
    $PARENT_ARG $ASSESS_TINY \
    --out-json "$A1" --out-markdown "$S1" 2>&1 | stamped | tee -a "$LOG"
  rc="${PIPESTATUS[0]}"
  [ "$rc" -eq 0 ] || die "oc_assess.py (v1) exited $rc" "$rc"
  say "    OK"
fi

A2="$OUT/assessment-$ID-B2$SFX.json"
S2="$OUT/SCORING-SHEET-$ID-B2$SFX.md"
B2_ARG=""
B2_PARENT_NOTE="bank v2 was not run"
B2_LANES_NO_PARENT=""
if [ -s "$D2" ]; then
  if have "$A2" "$S2"; then
    say "--- step 3b: assess v2 -- SKIP: $A2 already exists ---"
    B2_PARENT_NOTE="(step 3b skipped on re-run; see the earlier log for the parent B2 comparability note)"
  else
    say "--- step 3b: assess v2 text lanes ---"
    PARENT_ARG2=""
    if have "$PARENT_DOSSIER_B2"; then
      if compare_tuples "$D2" "$PARENT_DOSSIER_B2" "parent-b2"; then
        PARENT_ARG2="--parent-dossier $PARENT_DOSSIER_B2"
        B2_PARENT_NOTE="parent B2 side-by-side: $(basename "$PARENT_DOSSIER_B2") (tuple matches)"
      else
        B2_PARENT_NOTE="parent B2 side-by-side OMITTED: $(basename "$PARENT_DOSSIER_B2") is a different surface (tuple difference logged at step 3b)"
        say "    NOTE: $B2_PARENT_NOTE"
        say "          The B2 sheet has NO parent column; re-run the parent's B2 lanes on the"
        say "          current tuple to get one.  No sealed count depends on it."
      fi
      # The lanes with no parent text at all: the parent's lc-8k rungs live in a
      # separate 300-cap dossier, and oc_assess simply omits the block for an id
      # the parent dossier lacks.  Say which ones, so the sheet's gap is stated.
      B2_LANES_NO_PARENT=$(timeout 120 "$PY" - "$D2" "$PARENT_DOSSIER_B2" <<'PYLANES' 2>>"$LOG"
import json, sys
from pathlib import Path
c = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
p = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
have = {r.get("id") for r in (p.get("child_responses") or [])}
print(",".join(sorted(r.get("id") for r in (c.get("child_responses") or []) if r.get("id") not in have)))
PYLANES
      )
      if [ -n "$B2_LANES_NO_PARENT" ]; then
        say "    NOTE: these child B2 lanes have NO parent response in $(basename "$PARENT_DOSSIER_B2"):"
        say "          $B2_LANES_NO_PARENT"
        say "          (the parent's lc-8k text is in a separate 300-cap dossier and is not comparable)"
      fi
    else
      B2_PARENT_NOTE="parent B2 side-by-side OMITTED: $PARENT_DOSSIER_B2 is absent"
      say "    NOTE: $B2_PARENT_NOTE"
    fi
    # shellcheck disable=SC2086
    timeout 3600 "$PY" "$LANE/oc_assess.py" --dossier "$D2" --bank "$V2" \
      $PARENT_ARG2 $ASSESS_TINY \
      --out-json "$A2" --out-markdown "$S2" 2>&1 | stamped | tee -a "$LOG"
    rc="${PIPESTATUS[0]}"
    [ "$rc" -eq 0 ] || die "oc_assess.py (v2) exited $rc" "$rc"
    say "    OK"
  fi
  B2_ARG="--assessment-b2 $A2"
else
  say "--- step 3b: assess v2 -- no v2 dossier, bank2 stays null in the review ---"
fi

# ---------------------------------------------------------------------------
# step 4 -- the review record.  DRY RUN FIRST, ALWAYS: `record-review` prints
# the plan, the computed status and the reason it computed it, and those numbers
# are worth reading once before they are sealed.
# ---------------------------------------------------------------------------
REV="$OUT/review-$ID$SFX.json"
NOTICE="$OUT/UPPER-BOUND-NOTICE-$ID$SFX.md"
#: The record's notes name the parent artifact the count came from, so the
#: sealed row says which parent surface it was compared against.
review_notes () {
  printf '%s' "Wave A first pass, $ID-NOTHINK$SFX: bank v1 --thinking off (cap $NEWTOK_V1), assessed WITHOUT --per-case.  EVERY COUNT IN THIS RECORD IS AN UPPER BOUND and none of it is a reading -- see bank1.upper_bound_note.  parent_useful_or_partial comes from $(basename "$PARENT_ASSESSMENT") about parent dossier_sha256 $PARENT_DOSSIER_SHA (parent surface comparable: $PARENT_V1_COMPARABLE).  bank2: $B2_PARENT_NOTE.  The status this record implies routes the child to Cyril's queue; the reading that replaces it is his per-case file against the scoring sheet.  Sealed by wavea_review.sh; the package carries UPPER-BOUND-NOTICE-$ID$SFX.md."
}
#: The existing REV is only reusable if it is about D1 and, when it carries a
#: package, about the package this run will verify.
assert_review_file () {
  local rev="$1" want_pkg_sha="${2:-}"
  timeout 120 "$PY" - "$rev" "$D1_SHA" "$want_pkg_sha" "$REVIEWER" <<'PYREV' 2>&1 | stamped | tee -a "$LOG"
import json, sys
from pathlib import Path
rev, d1, pkg, reviewer = sys.argv[1:5]
r = json.loads(Path(rev).read_text(encoding="utf-8"))
bad = []
if r.get("dossier_sha256") != d1:
    bad.append(f"dossier_sha256 {str(r.get('dossier_sha256'))[:16]} != this run's D1 {d1[:16]}")
if not r.get("is_upper_bound"):
    bad.append("is_upper_bound is not true")
if str(r.get("reviewer")) != reviewer:
    bad.append(f"reviewer {r.get('reviewer')!r} != {reviewer!r}")
if pkg and r.get("package_sha256") != pkg:
    bad.append(f"package_sha256 {str(r.get('package_sha256'))[:16]} != the package on disk {pkg[:16]}")
if not pkg and r.get("package_sha256"):
    bad.append("carries a package_sha256 but this is the bare (pre-package) review")
if bad:
    print(f"REVIEW FILE MISMATCH {rev}: " + "; ".join(bad)); sys.exit(4)
print(f"review file verified: {Path(rev).name} dossier {d1[:16]} is_upper_bound true" + (f" package {pkg[:16]}" if pkg else ""))
PYREV
  return "${PIPESTATUS[0]}"
}

say "--- step 4: review JSON (upper bound) -> $(basename "$REV") ---"
if have "$REV"; then
  assert_review_file "$REV" || die "$REV exists but is not this run's bare bound; move it aside by hand" 3
  say "    SKIP: $REV already exists and is about $D1_SHA"
else
  need_parent_v1
  # shellcheck disable=SC2086
  timeout 300 "$PY" "$LANE/oc_review_upper_bound.py" \
    --assessment "$A1" --dossier "$D1" $B2_ARG \
    --parent-assessment "$PARENT_ASSESSMENT" \
    --reviewer "$REVIEWER" $ASSESS_TINY --notes "$(review_notes)" \
    --out "$REV" 2>&1 | stamped | tee -a "$LOG"
  rc="${PIPESTATUS[0]}"
  [ "$rc" -eq 0 ] || die "oc_review_upper_bound.py exited $rc" "$rc"
  say "    OK"
fi

#: The package's own statement that every number in it is a ceiling.  Until
#: oc_package/oc_dashboard render `is_upper_bound`, this file and the review
#: JSON travel in the package so the caveat is not only in a notes column.
if have "$NOTICE"; then
  say "    notice: $NOTICE already exists"
else
  need_parent_v1
  {
    echo "# UPPER-BOUND NOTICE -- $ID"
    echo
    echo "Written $(now) by wavea_review.sh (reviewer '$REVIEWER')."
    echo
    echo "EVERY 'sealed' NUMBER in REVIEW_<TAG>.md, in the ledger row for this checkpoint"
    echo "and in SCORECARD.md / REVIEW_QUEUE.md is the MACHINE CEILING produced by"
    echo "oc_review_upper_bound.py from an assessment run WITHOUT --per-case.  Nobody has"
    echo "read this child.  'sealed N/15 vs parent M/15' means: at most N scored lanes"
    echo "could turn out to hold their case (the other lanes provably emitted nothing,"
    echo "looped, or rode the cap), and the parent's M is the parent's own ceiling from"
    echo "the same translation.  canonical_content_miss is 0 BY CONSTRUCTION (a content"
    echo "miss is a reading and the only box that can make a checkpoint deletable), so"
    echo "nothing in this package can make the child \`bad\`.  The verdict word is the best"
    echo "verdict the ceiling could support, stamped sealed_verdict_is_upper_bound."
    echo
    echo "The READING that replaces these numbers is Cyril's per-case file against"
    echo "SCORING-SHEET-$ID-NOTHINK$SFX.md (and SCORING-SHEET-$ID-B2$SFX.md for the"
    echo "bank v2 text lanes).  The review JSON in this package (review-$ID$SFX.json)"
    echo "carries is_upper_bound / upper_bound_note on every block."
    echo
    echo "## Surfaces"
    echo
    echo "- child v1 dossier_sha256: $D1_SHA (cap $NEWTOK_V1, thinking off, dtype $DTYPE)"
    echo "- parent v1: $(basename "$PARENT_DOSSIER") dossier_sha256 $PARENT_DOSSIER_SHA; tuple comparable with the child: $PARENT_V1_COMPARABLE"
    echo "- bank v2: $B2_PARENT_NOTE"
    if [ -n "$B2_LANES_NO_PARENT" ]; then
      echo "- bank v2 lanes with NO parent response in the parent B2 dossier (the parent's lc-8k text lives in a separate 300-cap dossier and is not comparable): $B2_LANES_NO_PARENT"
    fi
    if [ "$MECHANICS" -eq 1 ]; then
      echo "- MECHANICS RUN (tiny_random=$TINY max_cases=${MAX_CASES:-all} cap override=${MAX_NEW_TOKENS:-none}) against throwaway root $ROOT: this is not a review of anything."
    fi
  } > "$NOTICE" || die "cannot write $NOTICE"
  say "    wrote $NOTICE"
fi

say "--- step 4a: record-review DRY RUN ---"
if [ "$CHILD_REVIEWED" = "yes" ]; then
  say "    SKIP: $ID already carries a review record ($CHILD_HEAD_RECORD)"
  if [ "$HEAD_IS_OUR_BOUND" = "yes" ] && [ "$CHILD_REVIEW_DOSSIER_SHA" != "$D1_SHA" ]; then
    die "the sealed bound on $ID is about dossier $CHILD_REVIEW_DOSSIER_SHA, not this run's $D1 ($D1_SHA); this script will not put a package digest on a record about a different run" 3
  fi
else
  timeout 300 "$PY" "$LEDGER_LANE/oc_checkpoint.py" --root "$ROOT" record-review \
    --checkpoint-id "$ID" --review-json "$REV" 2>&1 | stamped | tee -a "$LOG"
  rc="${PIPESTATUS[0]}"
  [ "$rc" -eq 0 ] || die "record-review dry run exited $rc" "$rc"

  say "--- step 4b: record-review --confirm (registry $REGISTRY) ---"
  timeout 300 "$PY" "$LEDGER_LANE/oc_checkpoint.py" --root "$ROOT" record-review \
    --checkpoint-id "$ID" --review-json "$REV" --confirm 2>&1 | stamped | tee -a "$LOG"
  rc="${PIPESTATUS[0]}"
  [ "$rc" -eq 0 ] || die "record-review --confirm exited $rc" "$rc"
  resolve_child
  say "    OK -- head is now $CHILD_HEAD_RECORD (status $CHILD_STATUS)"
  [ "$HEAD_IS_OUR_BOUND" = "yes" ] || die "after sealing, the head record is not this script's bound (event=$CHILD_HEAD_EVENT reviewer='$CHILD_REVIEWER' bound=$CHILD_REVIEW_IS_BOUND); stopping before the package" 3
fi

# ---------------------------------------------------------------------------
# step 5 -- the package, at R7's location.  NOT under qwen-fold/: that tree
# belongs to the build lane and this script writes nothing into it.
# ---------------------------------------------------------------------------
say "--- step 5: oc_package.py build -> $PKG ---"
if [ "$CHILD_PACKAGED" = "yes" ]; then
  # The record vouches for bytes at a named path with a named digest.  Verify
  # THOSE, never rebuild beside them: a second package under the same id with
  # a different digest is a ledger vouching for bytes that no longer exist.
  say "    the record already names package_path $CHILD_PACKAGE_PATH"
  say "    package_sha256 $CHILD_PACKAGE_SHA"
  [ "$CHILD_PACKAGE_PATH" = "$PKG" ] || say "    NOTE: the sealed package_path is not $PKG; the RECORD wins and is what is verified"
  have "$CHILD_PACKAGE_PATH/MANIFEST.json" || die "the record names package $CHILD_PACKAGE_PATH but it has no MANIFEST.json on disk; the ledger vouches for bytes that are not there -- find where the package went (it is never rebuilt from here)" 3
  timeout 900 "$PY" "$LEDGER_LANE/oc_package.py" verify --package "$CHILD_PACKAGE_PATH" 2>&1 | stamped | tee -a "$LOG"
  rc="${PIPESTATUS[0]}"
  [ "$rc" -eq 0 ] || die "oc_package.py verify exited $rc for the sealed package $CHILD_PACKAGE_PATH" "$rc"
  ON_DISK_SHA=$(timeout 120 "$PY" -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["package_sha256"])' "$CHILD_PACKAGE_PATH/MANIFEST.json" 2>>"$LOG") \
    || die "cannot read package_sha256 from $CHILD_PACKAGE_PATH/MANIFEST.json"
  [ "$ON_DISK_SHA" = "$CHILD_PACKAGE_SHA" ] || die "the package on disk digests $ON_DISK_SHA but the sealed record says $CHILD_PACKAGE_SHA" 3
  say "    SKIP build: the sealed package verifies and its digest matches the record"
  PKG="$CHILD_PACKAGE_PATH"
else
  [ "$HEAD_IS_OUR_BOUND" = "yes" ] || die "the head record ($CHILD_HEAD_RECORD, event=$CHILD_HEAD_EVENT, reviewer='$CHILD_REVIEWER', is_upper_bound=$CHILD_REVIEW_IS_BOUND) is a reading or a ruling, not this script's bound.  It carries no package digest, and this script will not supersede it with a ceiling: add package_path/package_sha256 to THAT payload by hand and record it." 3
  [ "$CHILD_REVIEW_DOSSIER_SHA" = "$D1_SHA" ] || die "the sealed bound is about dossier $CHILD_REVIEW_DOSSIER_SHA, not this run's $D1_SHA" 3
  if have "$PKG/MANIFEST.json"; then
    say "    $PKG/MANIFEST.json already exists (a previous run built it and stopped before 5b); verifying"
    timeout 900 "$PY" "$LEDGER_LANE/oc_package.py" verify --package "$PKG" 2>&1 | stamped | tee -a "$LOG"
    rc="${PIPESTATUS[0]}"
    [ "$rc" -eq 0 ] || die "oc_package.py verify exited $rc for $PKG; it is not moved or rebuilt from here" "$rc"
    say "    SKIP build: existing package verifies"
  else
    mkdir -p "$PKG_ROOT" || die "cannot create $PKG_ROOT"
    pkg_args=(--checkpoint-id "$ID" --out "$PKG" --dossier "$D1")
    [ -s "$D2" ] && pkg_args+=(--dossier-b2 "$D2")
    for extra in "$S1" "$A1" "$S2" "$A2" "$NOTICE" "$REV"; do
      [ -s "$extra" ] && pkg_args+=(--extra "$extra")
    done
    if [ -d "$PKG" ] && [ -n "$(ls -A "$PKG" 2>/dev/null)" ]; then
      say "    NOTE: $PKG is a partial package with no MANIFEST.json (a crashed build)"
      say "          and the record names no package digest.  Passing --overwrite, which"
      say "          MOVES it to $PKG.superseded-<utc> and deletes nothing."
      pkg_args+=(--overwrite)
    fi
    timeout 900 "$PY" "$LEDGER_LANE/oc_package.py" --root "$ROOT" build \
      "${pkg_args[@]}" 2>&1 | stamped | tee -a "$LOG"
    rc="${PIPESTATUS[0]}"
    [ "$rc" -eq 0 ] || die "oc_package.py build exited $rc" "$rc"
    say "    OK"
  fi
fi

# ---------------------------------------------------------------------------
# step 5b -- the SECOND review record, carrying the package digest.  See the
# ordering note at the top of the file: this is the record R4 would later need,
# and it can only be written after the package exists.  It supersedes ONLY this
# script's own bound about THIS dossier, and it is dry-run first like 4a.
# ---------------------------------------------------------------------------
say "--- step 5b: re-record the review with package_path + package_sha256 ---"
if [ "$CHILD_PACKAGED" = "yes" ]; then
  say "    SKIP: the current record already names package_sha256 $CHILD_PACKAGE_SHA"
else
  resolve_child
  [ "$CHILD_HEAD_EVENT" = "review" ] || die "head record is a $CHILD_HEAD_EVENT record; a status/ruling is never superseded by this script" 3
  [ "$HEAD_IS_OUR_BOUND" = "yes" ] || die "the existing review (reviewer '$CHILD_REVIEWER', is_upper_bound=$CHILD_REVIEW_IS_BOUND) is a reading, not this script's bound; not superseded" 3
  [ "$CHILD_REVIEW_DOSSIER_SHA" = "$D1_SHA" ] || die "sealed review is about dossier $CHILD_REVIEW_DOSSIER_SHA, not $D1 ($D1_SHA)" 3
  [ "$CHILD_PACKAGED" = "no" ] || die "the head record already carries a package digest; nothing to add" 3
  PKG_SHA=$(timeout 120 "$PY" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["package_sha256"])' \
    "$PKG/MANIFEST.json" 2>>"$LOG") || die "cannot read package_sha256 from $PKG/MANIFEST.json"
  say "    package_sha256 $PKG_SHA"
  REVP="$OUT/review-$ID$SFX-packaged.json"
  if have "$REVP"; then
    assert_review_file "$REVP" "$PKG_SHA" || die "$REVP exists but is about another dossier or package; move it aside by hand" 3
    say "    $REVP already exists and matches this dossier and package"
  else
    need_parent_v1
    # shellcheck disable=SC2086
    timeout 300 "$PY" "$LANE/oc_review_upper_bound.py" \
      --assessment "$A1" --dossier "$D1" $B2_ARG \
      --parent-assessment "$PARENT_ASSESSMENT" \
      --reviewer "$REVIEWER" $ASSESS_TINY --notes "$(review_notes)" \
      --package-path "$PKG" --package-sha256 "$PKG_SHA" \
      --out "$REVP" 2>&1 | stamped | tee -a "$LOG"
    rc="${PIPESTATUS[0]}"
    [ "$rc" -eq 0 ] || die "oc_review_upper_bound.py (packaged) exited $rc" "$rc"
  fi
  say "    record-review (packaged) DRY RUN"
  timeout 300 "$PY" "$LEDGER_LANE/oc_checkpoint.py" --root "$ROOT" record-review \
    --checkpoint-id "$ID" --review-json "$REVP" 2>&1 | stamped | tee -a "$LOG"
  rc="${PIPESTATUS[0]}"
  [ "$rc" -eq 0 ] || die "record-review (packaged) dry run exited $rc" "$rc"
  say "    record-review (packaged) --confirm (registry $REGISTRY)"
  timeout 300 "$PY" "$LEDGER_LANE/oc_checkpoint.py" --root "$ROOT" record-review \
    --checkpoint-id "$ID" --review-json "$REVP" --confirm 2>&1 | stamped | tee -a "$LOG"
  rc="${PIPESTATUS[0]}"
  [ "$rc" -eq 0 ] || die "record-review (packaged) --confirm exited $rc" "$rc"
  resolve_child
  [ "$CHILD_PACKAGED" = "yes" ] && [ "$CHILD_PACKAGE_SHA" = "$PKG_SHA" ] \
    || die "after sealing, the head record does not carry package_sha256 $PKG_SHA (has '$CHILD_PACKAGE_SHA')" 3
  say "    OK -- head is now $CHILD_HEAD_RECORD (status $CHILD_STATUS) with package_sha256 $CHILD_PACKAGE_SHA"
fi

# ---------------------------------------------------------------------------
# step 6 -- the dashboards.  SCORECARD.md and REVIEW_QUEUE.md are regenerated
# FROM the registry every time; neither is an input to anything.
# ---------------------------------------------------------------------------
say "--- step 6: oc_dashboard.py ---"
timeout 300 "$PY" "$LEDGER_LANE/oc_dashboard.py" --root "$ROOT" 2>&1 | stamped | tee -a "$LOG"
rc="${PIPESTATUS[0]}"
[ "$rc" -eq 0 ] || die "oc_dashboard.py exited $rc" "$rc"

# The dashboard writes ids inside backticks; match the delimited fixed string,
# so a shorter tag cannot match a longer one and a dot in a parent code is a dot.
if grep -qF "\`$ID\`" "$ROOT/REVIEW_QUEUE.md" 2>/dev/null; then
  say "    REVIEW_QUEUE.md names $ID:"
  grep -nF "\`$ID\`" "$ROOT/REVIEW_QUEUE.md" | head -3 | stamped | tee -a "$LOG"
else
  say "    NOTE: REVIEW_QUEUE.md does not name $ID.  Read the status line in"
  say "          step 4b -- an 'unmeasured' child is listed under a different"
  say "          heading and is NOT in Cyril's queue until it is re-run."
fi

if [ "$MECHANICS" -eq 1 ]; then
  say "=== DONE $ID (MECHANICS RUN, throwaway root $ROOT; nothing here is a review) -- log $LOG ==="
else
  say "=== DONE $ID -- log $LOG ==="
fi
exit 0
