# Inventory snapshot, 2026-10-21 (arch v2)

Written by the arch-v2 realignment (Claude session 0c189f21, workflow `wddsm1r3p`). Same rule as
`../2026-10-21`: ingesters read these files; they are never edited. A newer inventory goes in a new folder.

| File | Ingester | Contents |
|---|---|---|
| `directives.json` | `ingest/directives.py` | 13 verbatim quotes from Cyril's 2026-10-21 arch-v2 prompt (`agent_memory_events.id=1230100`). All 13 matched their source exactly on first ingest (batch 87). |
| `arch_requirements.json` | `ingest/arch_requirements.py` | R1-R16, the standing architecture requirements. R1-R12 are the review's `intended_arch.json` set (ARCHITECTURE_REVIEW_2026-10-21.md section 2), restated where 2026-10-21 changed them (R2 MoE "a maybe", R8 256k); R13-R16 are new on 2026-10-21. Every ref resolves to `directive` rows by fragment. |

Load order:

    omni_lab.py ingest directives --inventory inventory/2026-10-21
    omni_lab.py ingest directives --inventory inventory/2026-10-21-arch-v2
    omni_lab.py ingest arch_requirements --inventory inventory/2026-10-21-arch-v2
    omni_lab.py arch-check --report <chimera_omni.conformance report> [--model KEY]

## Public copy (2026-11-06)

This published copy differs from the sealed snapshot the ingesters read.
`arch_requirements.json` has Cyril's words withheld (`[Cyril quote withheld]`), and
`directives.json` is not published. The sealed files stay in the private checkout.
`SHA256SUMS` here covers the published files.
