# Inventory snapshot, 2026-10-21

Read-only inventories built on 2026-10-21 by the Omni Lab review (Claude session 0c189f21), frozen here so the
knowledge ingesters can be re-run after the session scratchpad is gone. The ingesters read these files; they are
never edited. A newer inventory goes in a new dated folder.

| File | Ingester | Contents |
|---|---|---|
| `directives.json` | `ingest/directives.py` | 136 verbatim, dated owner directives. Every quote was checked by script as an exact substring of its source (agent_memory_events prompt, Claude Code transcript line, or doc/memory line) when the inventory was built. |
| `docs.json` | `ingest/documents.py` | 1,927 unique documents (sha256-deduplicated) with type, arm, status, date, first-sentence summary and top-30 rank. |
| `foldhist.json` | `ingest/foldhist.py` | 43 folding experiments (schema `foldhist-v1`) with an evidence level per entry. |

The files were converted from CRLF to LF on copy (the repo pins LF). Parsed JSON is identical to the originals.

| File | sha256 here (LF) | sha256 of the scratchpad original (CRLF) |
|---|---|---|
| `directives.json` | `7ab7d3409ed21e0a8cb230d0a569ee90796928e5c28b79c9b5ed90b646ef8c19` | `41ad195045387faa450776c865dd932cccbe84300f72ac123fc4e17db504eaea` |
| `docs.json` | `e8123c26309f723169f794ffd19ace9395fff118b5dd899748641f72a21545a9` | `85d8dff7d566f67c6e57e82c559b53dd1d21713739c598dff90ba328de321f48` |
| `foldhist.json` | `094f398bed4ca08245f5fd205963874d9a3711b0c3b32b90c87a47f0f21ea9e7` | `d3b069b1c55544683a87cc7eef6b9a0ef97a0f5331124cc5d4d8693b7e2c42b3` |

Caveats carried from the inventories:
- `docs.json` summaries are the first sentence of each document, and `type`/`arm` come from keyword matching. 597 dates are file mtimes.
- Dot (ChatGPT sync) directives are dated by sync time; their original times are UNVERIFIED.
- Four architecture rulings exist only in Claude Code transcripts (queued mid-turn prompts the memory hook never filed).
