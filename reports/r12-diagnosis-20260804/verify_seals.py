#!/usr/bin/env python3
"""Independently re-derive the sealed digests of the fetched diagnosis artifacts."""

import hashlib
import json
import sys
from pathlib import Path


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def check(path, seal_field, extra_pop=()):
    document = json.loads(Path(path).read_text())
    body = dict(document)
    for field in extra_pop:
        body.pop(field, None)
    claimed = body.pop(seal_field)
    derived = canonical_sha256(body)
    status = "OK  " if derived == claimed else "FAIL"
    print(f"{status} {seal_field}={claimed} file={Path(path).name}")
    return derived == claimed, document


def main(root: str) -> int:
    base = Path(root)
    ok = True

    fd_ok, fd_ledger = check(
        base / "first-divergence/first-divergence-ledger.json", "ledger_sha256"
    )
    ok &= fd_ok
    fd_req_ok, fd_request = check(
        base / "first-divergence/request.json", "request_sha256"
    )
    ok &= fd_req_ok
    fd_complete_ok, fd_complete = check(
        base / "first-divergence/complete.json", "completion_sha256"
    )
    ok &= fd_complete_ok

    if fd_ledger["request"] != fd_request:
        print("FAIL ledger-embedded request differs from request.json")
        ok = False
    else:
        print("OK   ledger-embedded request == request.json")
    if fd_complete["ledger_sha256"] != fd_ledger["ledger_sha256"]:
        print("FAIL completion ledger_sha256 differs from the ledger")
        ok = False
    else:
        print("OK   completion ledger_sha256 == ledger")

    for label in ("layer-00", "layer-12"):
        directory = base / "causal-diagnostic" / label
        causal_ok, causal_ledger = check(
            directory / "causal-ledger.json",
            "ledger_sha256",
            extra_pop=("ledger_hmac_sha256",),
        )
        ok &= causal_ok
        request_ok, causal_request = check(
            directory / "request.json", "request_sha256"
        )
        ok &= request_ok
        completion_ok, completion = check(
            directory / "complete.json", "completion_sha256"
        )
        ok &= completion_ok
        if causal_ledger["request"] != causal_request:
            print(f"FAIL {label} ledger request differs from request.json")
            ok = False
        else:
            print(f"OK   {label} ledger-embedded request == request.json")
        binding = causal_request["v3_binding"]
        if (
            binding["first_divergence_ledger_sha256"] != fd_ledger["ledger_sha256"]
            or binding["first_divergence_request_sha256"]
            != fd_request["request_sha256"]
            or binding["child_package_sha256"] != fd_request["child_package_sha256"]
            or binding["trace_cache_key"] != fd_request["trace_cache_key"]
        ):
            print(f"FAIL {label} v3 binding does not point at the fetched v3 ledger")
            ok = False
        else:
            print(
                f"OK   {label} v3_binding pins the fetched first-divergence "
                "ledger/request/child/trace"
            )
        audit = causal_ledger["operation_audit"]
        if audit["parent_forward_calls"] or audit["training_steps"]:
            print(f"FAIL {label} operation audit is not cache-only")
            ok = False
        else:
            print(
                f"OK   {label} parent_forward_calls=0 training_steps=0 "
                f"child_forward_calls={audit['child_forward_calls']}"
            )
        if completion["status"] != "complete-observation":
            print(f"FAIL {label} status {completion['status']}")
            ok = False

    audit = fd_ledger["operation_audit"]
    if audit["parent_forward_calls"] or audit["training_steps"] or audit["model_mutations"]:
        print("FAIL first-divergence operation audit is not cache-only")
        ok = False
    else:
        print(
            "OK   first-divergence parent_forward_calls=0 training_steps=0 "
            f"model_mutations=0 child_forward_calls={audit['child_forward_calls']}"
        )
    print("ALL SEALS VERIFIED" if ok else "SEAL VERIFICATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
