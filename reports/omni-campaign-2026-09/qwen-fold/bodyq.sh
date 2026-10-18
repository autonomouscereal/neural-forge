#!/usr/bin/env bash
# One child, one transaction.  bodyq.sh <TAG> <qf_build args...>
#
#   ./bodyq.sh C2K2304 --mlp-keep 2304
#   ./bodyq.sh X03      --excise 3
#   ./bodyq.sh G5V16K8  --gdn-value-heads 16 --gdn-key-heads 8
#
# The receipt filename and the log prefix are asserted here by EXACT NAME.  The
# wave-34 lesson: the queue reads `qf-build-receipt.json` and nothing else, so a
# builder that renames it fails with FileNotFoundError three hours later instead
# of at the rename.
set -euo pipefail
source "$(dirname "$0")/env.sh"

TAG="${1:?usage: bodyq.sh <TAG> <qf_build args...>}"
shift
CHILD="$W/children/QF-$TAG"
LOG="$W/logs/build-$TAG.log"

gpu_preflight

timeout 86400 "$PY" "$W/qf_build.py" \
  --parent "$PARENT" \
  --out "$CHILD" \
  --evidence "$EVIDENCE" \
  --tag "$TAG" \
  --device cuda:0 \
  --dtype bfloat16 \
  --gguf "$LLAMA_CPP" \
  "$@" 2>&1 | tee "$LOG"

RECEIPT="$CHILD/qf-build-receipt.json"
test -f "$RECEIPT" || { echo "[qf] $RECEIPT was not written" >&2; exit 1; }

# Reproduction check: the digest the receipt claims must be the digest the
# directory has, recomputed here from the files rather than trusted.
"$PY" - "$CHILD" "$RECEIPT" <<'PYEOF'
import json, sys
sys.path.insert(0, "/home/cereal/omni-campaign/qwen-fold")
import qf_export

child, receipt_path = sys.argv[1], sys.argv[2]
receipt = json.loads(open(receipt_path, encoding="ascii").read())
digest, _ = qf_export.directory_digest(child)
claimed = receipt["export"]["child_tensors_sha256"]
if digest != claimed:
    raise SystemExit(f"[qf] child digest {digest} != receipt {claimed}")
print(f"[qf] reproduction ok: {digest[:16]}  "
      f"{receipt['storage']['child_parameter_count']:,} parameters  "
      f"net {receipt['storage']['net_stored_ratio']:.4f}x")
PYEOF
