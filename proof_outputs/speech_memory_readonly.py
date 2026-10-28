"""Bounded PostgreSQL speech-source audit. SELECT only; never run trace commands."""
import argparse
import asyncio
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/cereal/.claude/memory_backend")
import agent_memory
from agent_memory_hook import redact_full_text
spec = importlib.util.spec_from_file_location("speech_archive_verifier", ROOT / "experiments/chimera_recovery/memory_harvest.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


async def query(args):
    import asyncpg
    cfg = await asyncio.wait_for(asyncio.to_thread(agent_memory.db_config), timeout=12)
    cfg.update(timeout=10, command_timeout=15, server_settings={"default_transaction_read_only": "on",
               "statement_timeout": "12000", "lock_timeout": "2000", "application_name": "speech_memory_readonly_audit"})
    conn = None
    try:
        conn = await asyncpg.connect(**cfg)
        cfg["password"] = ""
        await conn.set_type_codec("jsonb", schema="pg_catalog", encoder=json.dumps, decoder=json.loads)
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            rows = await conn.fetch("""
              WITH hits AS MATERIALIZED (
                SELECT id, ts_rank_cd(search_vector, websearch_to_tsquery('english',$1)) AS rank
                FROM agent_memory_events
                WHERE search_vector @@ websearch_to_tsquery('english',$1)
                  AND created_at >= $2 AND created_at < $3
                  AND ($5::text[] IS NULL OR space=ANY($5::text[]))
                ORDER BY rank DESC, id DESC LIMIT $4
              )
              SELECT e.id,e.event_uid,e.created_at,e.space,e.agent,e.event_type,e.role,e.source,
                e.metadata - 'raw_payload' AS metadata,
                ts_headline('english',substring(e.content for 100000),websearch_to_tsquery('english',$1),
                  'MaxWords=100,MinWords=20,MaxFragments=3') AS snippet,
                a.id AS archive_id,a.event_id AS archive_event_id,a.event_uid AS archive_event_uid,
                a.payload_kind,a.payload_sha256,a.payload_format,a.compression,a.raw_chars,a.compressed_bytes,a.redacted,
                CASE WHEN a.redacted AND a.raw_chars<=2000000 AND a.compressed_bytes<=1000000
                  AND sum(COALESCE(a.compressed_bytes,0)) OVER(ORDER BY hits.rank DESC,e.id DESC)<=16777216
                  THEN a.payload END AS payload
              FROM hits JOIN agent_memory_events e USING(id)
              LEFT JOIN LATERAL (
                SELECT * FROM agent_memory_payload_archives a
                WHERE a.event_id=e.id AND a.payload_kind='hook_payload'
                  AND (e.metadata->'trace_archive'->>'payload_sha256' IS NULL OR
                    a.payload_sha256=e.metadata->'trace_archive'->>'payload_sha256')
                ORDER BY a.id DESC LIMIT 1
              ) a ON true ORDER BY hits.rank DESC,e.id DESC
            """, args.query, args.since, args.until, args.limit, args.space or None)
        result = []
        for raw in rows:
            row = dict(raw)
            blob = row.pop("payload")
            metadata = row.pop("metadata") or {}
            # No local harness transcript is opened; omit its pointer from the audit.
            row["metadata_keys"] = sorted(k for k in metadata if k not in {"transcript_path", "session_id"})
            row["archive_integrity_verified"] = False
            if blob is not None:
                try:
                    if row["archive_event_id"] != row["id"] or row["archive_event_uid"] != row["event_uid"]:
                        raise ValueError("archive_relation")
                    payload = H.archive_payload({**row, "payload": blob})
                    row["archive_integrity_verified"] = True
                    row["captured_evidence"] = H.redact({key: payload[key] for key in
                        ("prompt", "last_assistant_message", "response", "tool_name", "tool_input", "tool_response", "messages")
                        if key in payload})
                except (ValueError, TypeError):
                    row["archive_status"] = "refused_integrity_or_codec"
            result.append(row)
        return {"query": args.query, "since": args.since.isoformat(), "until": args.until.isoformat(),
                "read_only": True, "limit": args.limit, "row_count": len(result), "rows": result}
    finally:
        cfg["password"] = ""
        if conn is not None:
            conn.terminate()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query", required=True)
    p.add_argument("--since", default="2025-01-01")
    p.add_argument("--until", default="2026-10-14")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--space", action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.since = datetime.datetime.fromisoformat(args.since).replace(tzinfo=datetime.timezone.utc)
    args.until = datetime.datetime.fromisoformat(args.until).replace(tzinfo=datetime.timezone.utc)
    if not 1<=args.limit<=100 or not datetime.datetime(2025,1,1,tzinfo=datetime.timezone.utc)<=args.since<args.until<=datetime.datetime(2027,1,1,tzinfo=datetime.timezone.utc):
        raise SystemExit("invalid2025-2026bounds")
    try:
        result = asyncio.run(asyncio.wait_for(query(args), timeout=45))
        text = redact_full_text(json.dumps(result,default=str,ensure_ascii=False,indent=2)) + "\n"
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as stream:
            stream.write(text)
        print(json.dumps({"out":str(args.out.resolve()),"sha256":hashlib.sha256(text.encode()).hexdigest(),
                          "read_only":True,"rows":result["row_count"],
                          "verified_archives":sum(r["archive_integrity_verified"] for r in result["rows"])}))
    except Exception as error:
        print(json.dumps({"status":"failed","error_type":type(error).__name__,"read_only":True}))
        raise SystemExit(2)


if __name__=="__main__":
    main()
