"""Closed directional compatibility for historical LFM2 topology phases.

This policy is intentionally outside the global-topology algorithm fingerprint.
The current global-frontier module imports the policy, so its reader path remains
source-bound, while the exact historical-to-current tuple can be pinned after
that reader source has reached its final digest without creating a self-hash.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping


GLOBAL_TOPOLOGY_PHASE_COMPATIBILITY_POLICY_SCHEMA = (
    "neural-forge-lfm2-global-topology-phase-compatibility-policy-v1"
)
GLOBAL_TOPOLOGY_PHASE_COMPATIBILITY_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-global-topology-phase-compatibility-receipt-v1"
)


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


# (historical global algorithm SHA-256, current global-frontier raw source
# SHA-256, current functional-width raw source SHA-256, current global
# algorithm SHA-256).  Populate only after the current reader source is final.
_EXPLICIT_GLOBAL_TOPOLOGY_PHASE_COMPATIBILITY = frozenset(
    {
        (
            # The authenticated A0007 capacity and functional completions
            # produced by commit da5b9da6 both carry this exact implementation
            # identity.  The previously used bb5541... value came from a
            # synthetic monkeypatched test publication, not either historical
            # completion, and is deliberately not allowlisted.  Commit
            # 2406ad0d changes only functional metadata enumeration; the
            # current global reader adds exact hit-only compatibility checks.
            "3e4531740792b3abe7ed54c02a49ca99caabc3ae94dd181c167913cbb5dbcab1",
            "4ff8fadb908bbc08bd03bc329f2cff435f7687e5e6ebda8e6c115f0f92543322",
            "3ca319c165d81c7fffb2f57a55edf3fcab49c647dd73cefb5a9b8d955a4741a7",
            "a91c75980d86da408ebef352d9ca29c30c5ffe30870e3ed24cfb3aa218359b39",
        ),
        (
            # Production reader after the callable fingerprint was made
            # source-portable.  The prior reader hashed ``code.co_code`` and
            # therefore produced different algorithm identities for the same
            # authenticated sources on Python 3.11 and 3.12.  This tuple binds
            # the exact v2 source reader and preserves the same directional,
            # hit-only A0007 compatibility contract.
            "3e4531740792b3abe7ed54c02a49ca99caabc3ae94dd181c167913cbb5dbcab1",
            "338d5b800bfac45dc8cee0017a98a850392610b34322db4cd55a0d8a0de73b12",
            "3ca319c165d81c7fffb2f57a55edf3fcab49c647dd73cefb5a9b8d955a4741a7",
            "d2ed9e076927dea1bfa44e47f3a2fff19b9617f486ca079e27948d212b6212dd",
        ),
    }
)


def explicit_global_topology_phase_compatibility_receipt(
    *,
    historical_algorithm_sha256: str,
    current_global_frontier_source_sha256: str,
    current_functional_width_source_sha256: str,
    current_algorithm_sha256: str,
) -> Mapping[str, Any] | None:
    """Seal one exact historical-to-current compatibility tuple, if listed."""

    identity = (
        historical_algorithm_sha256,
        current_global_frontier_source_sha256,
        current_functional_width_source_sha256,
        current_algorithm_sha256,
    )
    if identity not in _EXPLICIT_GLOBAL_TOPOLOGY_PHASE_COMPATIBILITY:
        return None
    policy_body = {
        "schema": GLOBAL_TOPOLOGY_PHASE_COMPATIBILITY_POLICY_SCHEMA,
        "directional_identities": [
            list(value)
            for value in sorted(_EXPLICIT_GLOBAL_TOPOLOGY_PHASE_COMPATIBILITY)
        ],
    }
    body = {
        "schema": GLOBAL_TOPOLOGY_PHASE_COMPATIBILITY_RECEIPT_SCHEMA,
        "policy_sha256": _canonical_sha256(policy_body),
        "historical_algorithm_sha256": historical_algorithm_sha256,
        "current_global_frontier_source_sha256": (
            current_global_frontier_source_sha256
        ),
        "current_functional_width_source_sha256": (
            current_functional_width_source_sha256
        ),
        "current_algorithm_sha256": current_algorithm_sha256,
        "direction": "historical-to-current",
        "historical_artifact_rewritten": False,
    }
    return MappingProxyType(
        {**body, "compatibility_receipt_sha256": _canonical_sha256(body)}
    )


__all__ = [
    "GLOBAL_TOPOLOGY_PHASE_COMPATIBILITY_POLICY_SCHEMA",
    "GLOBAL_TOPOLOGY_PHASE_COMPATIBILITY_RECEIPT_SCHEMA",
    "explicit_global_topology_phase_compatibility_receipt",
]
