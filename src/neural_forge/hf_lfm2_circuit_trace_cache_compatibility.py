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
            # Sealed compiler/interface cache used by the exact-4x composite
            # fold.  Its runtime, canonical contract, tensor schema, and
            # package identities are identical to the current reader.  The
            # collector delta adds authenticated/selective consumption and
            # hardens file-identity checks; it does not reinterpret any
            # committed manifest, record, or tensor field.
            "eb3bddeb3414c80b252b37dfa14b991e9066894f2534138dc9c118c9e711bff4",
            "92ee067675f67e887b1e276fcf2bc2ccc46f853907b4e5f8ef38762fcc4ed0b0",
            "e3ae37aea119e3eec9caabb8a6ecc15c6e62e2f0a1fbeed51d28a08537e010a0",
        ),
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
        (
            # This reader revision canonicalizes the redundant serialized
            # block_ff_dim alias against intermediate_size.  It changes no
            # trace tensor, manifest, record, or runtime algebra semantics.
            "eb3bddeb3414c80b252b37dfa14b991e9066894f2534138dc9c118c9e711bff4",
            "b27c0913a35952793beb3831ef3b90beeacc7034dec499b1a8a6abd1bb7b5dc9",
            "6310b33af154f9eba5aafa804dbc5052a4b658b48f43d9643a15a4626d1328b0",
        ),
        (
            "00700dc26c52f54e8abec5eb7f213d5123a9f55466b286cf3caaabe0a2ac98d5",
            "b27c0913a35952793beb3831ef3b90beeacc7034dec499b1a8a6abd1bb7b5dc9",
            "6310b33af154f9eba5aafa804dbc5052a4b658b48f43d9643a15a4626d1328b0",
        ),
        (
            "00700dc26c52f54e8abec5eb7f213d5123a9f55466b286cf3caaabe0a2ac98d5",
            "b27c0913a35952793beb3831ef3b90beeacc7034dec499b1a8a6abd1bb7b5dc9",
            "11b6b5c8646447ff11ce357863e6a6166d9675dcef9006fe32a49b585009816f",
        ),
        (
            # Descriptor-local RoPE capture is gated exclusively to the sealed
            # heterogeneous dynamic runtime.  Native LFM2 hook inputs, tensor
            # interpretation, and every committed cache field remain exact.
            "eb3bddeb3414c80b252b37dfa14b991e9066894f2534138dc9c118c9e711bff4",
            "e76989290bf53d018de3489b9257abe41c36d9c4d861b4ad1ddedc71a3da4ab7",
            "fbecf56b486905dfdcad53ca7ddec70a6d48aed4d6f46fde236297da751050cc",
        ),
        (
            "00700dc26c52f54e8abec5eb7f213d5123a9f55466b286cf3caaabe0a2ac98d5",
            "e76989290bf53d018de3489b9257abe41c36d9c4d861b4ad1ddedc71a3da4ab7",
            "fbecf56b486905dfdcad53ca7ddec70a6d48aed4d6f46fde236297da751050cc",
        ),
        (
            "00700dc26c52f54e8abec5eb7f213d5123a9f55466b286cf3caaabe0a2ac98d5",
            "e76989290bf53d018de3489b9257abe41c36d9c4d861b4ad1ddedc71a3da4ab7",
            "aaabc923b437992b8f0871a059cb6321bb83408f86399409f9359962ed8223a9",
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
