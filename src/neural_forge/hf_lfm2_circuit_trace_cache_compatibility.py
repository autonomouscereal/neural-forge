"""Exact directional compatibility policy for sealed LFM2 trace caches.

This policy is deliberately separate from the collector whose source hash it
pins.  Keeping the receipt external avoids a self-referential source digest:
changing the collector changes its raw hash and invalidates every listed pair
until this policy is explicitly audited and updated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# (stored implementation SHA-256, current raw collector source SHA-256,
# current implementation SHA-256)
_EXPLICIT_IMPLEMENTATION_COMPATIBILITY = frozenset(
    {
        (
            "00700dc26c52f54e8abec5eb7f213d5123a9f55466b286cf3caaabe0a2ac98d5",
            "92ee067675f67e887b1e276fcf2bc2ccc46f853907b4e5f8ef38762fcc4ed0b0",
            "e3ae37aea119e3eec9caabb8a6ecc15c6e62e2f0a1fbeed51d28a08537e010a0",
        ),
        (
            "00700dc26c52f54e8abec5eb7f213d5123a9f55466b286cf3caaabe0a2ac98d5",
            "92ee067675f67e887b1e276fcf2bc2ccc46f853907b4e5f8ef38762fcc4ed0b0",
            "5d43adf027fe4a7926c37c7e9954766b74c240465491bc443a9a6bdb9ae82597",
        ),
    }
)


def permits_explicit_implementation_compatibility(
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    """Return whether this exact stored-to-current implementation pair is sealed."""

    return (
        stored.get("implementation_sha256"),
        current.get("collector_source_sha256"),
        current.get("implementation_sha256"),
    ) in _EXPLICIT_IMPLEMENTATION_COMPATIBILITY


__all__ = ["permits_explicit_implementation_compatibility"]
