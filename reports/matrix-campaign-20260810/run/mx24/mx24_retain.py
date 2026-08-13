#!/usr/bin/env python3
"""Wave 24 retention: twelve children built, TEN kept, two deleted as duplicates.

This wave departs from the standing best-five policy in one direction and one
direction only, and the reason is a fact about this wave rather than a preference:
**ten of the twelve carry a sealed ADDENDUM 6 verdict**, against five in wave 23,
because the lambda atlas grades four rungs of one ladder on each of two bases and
every rung is a separate measurement at an identical parameter count.  A verdict
without the checkpoint it was read from is a claim, and deleting five graded
checkpoints to satisfy a number would turn half this wave's findings into claims.
The children live on the 4.5 TB share (1.8 TB free), not on the root filesystem
the disk policy is about, and the whole wave is 3.6 GiB.

The two deletions are the two children this wave proved are **not children**:

* ``MX24-V8G`` is byte-identical to ``MX24-V8``
  (model.safetensors sha256 7815714b70b5a1f375d57222...).  The terminal-read
  solve on the vocabulary fold ALONE is an exact no-op: on the sealed corpus the
  child's segmentation equals the parent's canonical segmentation (position
  agreement 1.0000) and its body is the parent's bytes, so Z equals N, the
  identity gate fires, and the diagonal comes back at update scale 1.3e-15 with
  prior relative logit SSE 0.0.
* ``MX24-V16X13G29`` is byte-identical to ``MX24-V16X13G``
  (sha256 cbd629f9c2a5c48a39f41633...).  All 26 block diagonals came back at the
  prior to 1.9e-15 with prior relative SSE exactly 0.0, because block 13 is the
  LAST block and excising it cannot perturb any residual upstream of it.  The
  29-diagonal arm on an X13 base is a null by construction.

Both keep their build receipts, which carry the numbers above; what is deleted is
a second copy of weights whose digest already proves the identity.
"""

import json
import os
import shutil
import sys

W = "/mnt/ai_data/neural-forge-matrix/wave24"
KEEP = {
    "MX24-V32",
    "MX24-V8",
    "MX24-V8X13",
    "MX24-V8CL896",
    "MX24-V16X13G",
    "MX24-V16X13GL01",
    "MX24-V16X13GL1",
    "MX24-V8X13G",
    "MX24-V8X13GL01",
    "MX24-V8X13GL1",
}
REASON = {
    "MX24-V32": (
        "11/15 at 0.853916x, verdict alive-parent-band, zero charged lanes -- the "
        "highest review this campaign has recorded for a child that is not an "
        "excision, and the only wave-24 child that is not read dead"
    ),
    "MX24-V8": (
        "4/15 at 0.744354x -- the deep solo rung of the dose-response.  Nothing "
        "in this wave is attributable without it: V8X13's six is only a "
        "composition finding against this four"
    ),
    "MX24-V8X13": (
        "6/15 at 0.691832x -- the zero-solve baseline the bar-beating child is "
        "measured against, at an identical parameter count"
    ),
    "MX24-V8CL896": (
        "0/15 at 0.432784x -- the deepest standard-shape child in the campaign, "
        "kept for its SIGNATURE: five of sixteen lanes return the prompt itself "
        "verbatim with zero cap hits, which is a failure mode 662 rows had not "
        "seen and which a later wave will want the checkpoint for"
    ),
    "MX24-V16X13G": (
        "5/15 at 0.728353x -- the terminal-read solve at the split-selected "
        "lambda on the 16,384 base.  Same score as its source and three Useful "
        "lanes where its source had none"
    ),
    "MX24-V16X13GL01": (
        "6/15 at 0.728353x -- the lambda 0.1 rung, the peak of the atlas on this "
        "base, and the rung the internal fit split did NOT choose"
    ),
    "MX24-V16X13GL1": (
        "6/15 at 0.728353x -- the lambda 1.0 rung.  Three rungs of one ladder at "
        "one parameter count with three different held sets is the atlas, and it "
        "is not an atlas with a rung missing"
    ),
    "MX24-V8X13G": (
        "4/15 at 0.691832x -- the split-selected rung on the 8,192 base, and the "
        "low end of the ladder whose high end beats the bar.  The three-case "
        "swing between this and V8X13GL01 at five ten-thousandths of solver "
        "criterion is this wave's sharpest measurement"
    ),
    "MX24-V8X13GL01": (
        "7/15 at 0.691832x -- THE BAR.  The deepest child in 24 waves to hold "
        "seven review cases, 8.4 percent below the 0.7548x re-review line, "
        "standard shape, stock Lfm2ForCausalLM, 1,024 values re-solved and zero "
        "parameters added"
    ),
    "MX24-V8X13GL1": (
        "6/15 at 0.691832x -- the lambda 1.0 rung on the 8,192 base, six Partials "
        "and no Useful, which is the other end of the same swap"
    ),
}
REBUILD = (
    "byte-identical duplicate of a kept child, proved by sha256 in the build "
    "receipt beside it; re-derive with run/mx24/gainq.sh if a copy is ever wanted"
)


def main() -> int:
    apply = "--apply" in sys.argv
    rows = []
    freed = 0
    root = os.path.join(W, "children")
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        size = sum(
            os.path.getsize(os.path.join(base, f))
            for base, _, files in os.walk(path)
            for f in files
        )
        keep = name in KEEP
        rows.append(
            {
                "child": name,
                "kept": keep,
                "bytes": size,
                "reason": REASON.get(
                    name,
                    "a byte-identical duplicate of a kept child -- the solve that "
                    "produced it returned its own prior, so this directory is a "
                    "second copy and not a measurement",
                ),
                "rebuild": None if keep else REBUILD,
            }
        )
        if not keep:
            freed += size
            if apply:
                shutil.rmtree(path)
    report = {
        "schema": "neural-forge-matrix-wave24-children-retention-v1",
        "policy": "every child carrying a sealed ADDENDUM 6 verdict is kept, "
        "because a verdict without its checkpoint is a claim and TEN of this "
        "wave's twelve carry one (the lambda atlas grades four rungs per base "
        "and every rung is a separate measurement at an identical parameter "
        "count).  The two deletions are byte-identical duplicates of kept "
        "children, proved by sha256, whose solves returned their own priors",
        "built": len(rows),
        "kept": sum(1 for r in rows if r["kept"]),
        "deleted": sum(1 for r in rows if not r["kept"]),
        "freed_bytes": freed,
        "freed_gib": round(freed / (1024**3), 3),
        "applied": apply,
        "children": rows,
    }
    out = f"{W}/children-retention-manifest-wave24.json"
    with open(out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(
        f"kept {report['kept']}, deleted {report['deleted']}, "
        f"freed {report['freed_gib']} GiB, applied={apply} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
