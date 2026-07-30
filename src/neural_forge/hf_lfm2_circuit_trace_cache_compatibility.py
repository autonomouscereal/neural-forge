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
        (
            # The canonical-contract delta adds fail-closed inspection for an
            # already materialized heterogeneous child.  The stored cache is
            # a native parent, so no native tensor, phase, topology, or trace
            # interpretation changes.  Package and LFM2 source identities are
            # otherwise byte-identical to the sealed 76ff compiler cache.
            "eb3bddeb3414c80b252b37dfa14b991e9066894f2534138dc9c118c9e711bff4",
            "e76989290bf53d018de3489b9257abe41c36d9c4d861b4ad1ddedc71a3da4ab7",
            "3127c0a937e1e731ca28ee15c094f395104f98aa113aca3b4e2463964faf2a62",
        ),
        (
            # This collector revision adds only child-side heterogeneous
            # single-pass replay, strict same-child equivalence certification,
            # and truthful proof-forward accounting below the committed parent
            # cache reader.  It does not change the stored parent manifest,
            # record, tensor schema, tensor interpretation, or native-parent
            # runtime algebra.  The canonical/runtime/package identities are
            # otherwise the exact audited identities for the sealed 76ff cache.
            "eb3bddeb3414c80b252b37dfa14b991e9066894f2534138dc9c118c9e711bff4",
            "d5e678580494096fd49b516b088e6acd93623074543ff5b57785d92423fa6708",
            "3bc19c78a2ea5a963ac09cbaf8f42204d85383410ff921598c0677eb7ebdb97c",
        ),
        (
            # The same child-only replay delta above, with the native-parent
            # canonical contract retained by the sealed compiler cache.
            "eb3bddeb3414c80b252b37dfa14b991e9066894f2534138dc9c118c9e711bff4",
            "d5e678580494096fd49b516b088e6acd93623074543ff5b57785d92423fa6708",
            "44c686dadfaef044be7e2cf5a5ade23e489c01311e40a53b0c87b4e3cf32b2f8",
        ),
        (
            "00700dc26c52f54e8abec5eb7f213d5123a9f55466b286cf3caaabe0a2ac98d5",
            "d5e678580494096fd49b516b088e6acd93623074543ff5b57785d92423fa6708",
            "44c686dadfaef044be7e2cf5a5ade23e489c01311e40a53b0c87b4e3cf32b2f8",
        ),
        (
            "00700dc26c52f54e8abec5eb7f213d5123a9f55466b286cf3caaabe0a2ac98d5",
            "d5e678580494096fd49b516b088e6acd93623074543ff5b57785d92423fa6708",
            "72e0fab9a5f330a6c7e415535f8a89708d7936dc7d2e201d3111ec82f71b5394",
        ),
        (
            # This reader revision adds only child-side LM-head operand
            # capture and a fail-closed IEEE forward-error certificate for
            # single-pass child replay.  It does not alter any committed
            # parent-cache tensor, manifest, record, topology, or execution
            # identity field.  Exact LM-head/runtime authentication is also
            # child-only and runs after the sealed parent cache is consumed.
            "eb3bddeb3414c80b252b37dfa14b991e9066894f2534138dc9c118c9e711bff4",
            "e2a1cca26e1beb133979066d46dbf23b1339d6c39da4257e9907a4e1a4439740",
            "a5788b59eee4db030928de8ea4f0ec4a20eadeca043e017ab90706acaa8d4507",
        ),
        (
            "00700dc26c52f54e8abec5eb7f213d5123a9f55466b286cf3caaabe0a2ac98d5",
            "e2a1cca26e1beb133979066d46dbf23b1339d6c39da4257e9907a4e1a4439740",
            "a5788b59eee4db030928de8ea4f0ec4a20eadeca043e017ab90706acaa8d4507",
        ),
        (
            "00700dc26c52f54e8abec5eb7f213d5123a9f55466b286cf3caaabe0a2ac98d5",
            "e2a1cca26e1beb133979066d46dbf23b1339d6c39da4257e9907a4e1a4439740",
            "f53e814312a38bc8aa4e1d47ad65916346d7e78d07698fd5df98a7da8aea1376",
        ),
        (
            "eb3bddeb3414c80b252b37dfa14b991e9066894f2534138dc9c118c9e711bff4",
            "e2a1cca26e1beb133979066d46dbf23b1339d6c39da4257e9907a4e1a4439740",
            "24ff9e995d1eeaba8d2441e1add67b9e315ae86bd025d6350ee304d19f1db3d7",
        ),
        (
            # Exact directional reuse of the sealed compiler/interface parent
            # cache produced by collector revision 42d97c35.  Every collector
            # delta through 72d1c3c4 was audited: authenticated/selective read
            # APIs preserve the committed manifest/record/tensor contracts;
            # block_ff_dim normalization removes only an alias equal to the
            # operative intermediate_size and rejects contradictory values;
            # heterogeneous topology and descriptor-local RoPE dispatch are
            # gated by the child-only dynamic-blank config marker; and replay,
            # LM-head, and numeric-envelope additions execute only for child
            # replay.  Canonical revision a98d2063 only adds heterogeneous
            # child inspectors and leaves native inspect_lfm2_layer semantics
            # unchanged.  Runtime and package identities remain exact.
            "eb3bddeb3414c80b252b37dfa14b991e9066894f2534138dc9c118c9e711bff4",
            "33b77b3fd7273556fc584531f4a7ad6c9720b1ed49ed16147de0f45c798f3a38",
            "ccbb880606e19654501e22a19af7aca76d798bb74f80e7f88d842b8a4a5c1fa5",
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
