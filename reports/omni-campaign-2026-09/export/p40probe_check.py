#!/usr/bin/env python3
"""Read-only check of the running P40 envelope container + a short-context sanity completion.

This file never stops, kills or removes a container; the test suite greps it to prove that.
It inspects a container it did not start, reads its logs, and (only if the server is
healthy) sends small completion requests through its port.

The container in this campaign is named `p40_256k` (the plan text calls it "p40probe";
both names are tried, and `--container` overrides).

  python3 p40probe_check.py
  python3 p40probe_check.py --container gpu0_256k --port 18081 --tag GPU0-3090
  python3 p40probe_check.py --queue-anyway --request-timeout 3600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_probe import (  # noqa: E402
    LlamaServer,
    analyse_server_log,
    docker_logs,
    flash_attn_evidence,
    load_r3_control,
    parse_nvidia_smi_csv,
    parse_server_flags,
    props_n_ctx,
    run,
    safe_tag,
    utc_now,
    write_receipt,
)

CANDIDATES = ("p40_256k", "p40probe")

SANITY_PROMPTS = [
    {"id": "SC-1", "n_predict": 24, "prompt": "Compute 17 + 26. Give only the number.\nAnswer:"},
    {"id": "SC-2", "n_predict": 32, "prompt": "Facts: the spare key is in drawer 7; the manual is in drawer 3.\nQuestion: which drawer holds the spare key?\nAnswer:"},
    {"id": "SC-3", "n_predict": 32, "prompt": "Name the largest planet in the solar system in one word.\nAnswer:"},
]


def resolve_container(explicit):
    """Resolve the container to inspect, and say whether it RUNS, EXISTS, or is GONE.

    Three states, not two. A container that exited still has logs, an image, a command
    line and a `docker inspect` record -- all of it evidence. A container that has been
    removed from the daemon has none of that, and the honest receipt for it is "there is
    nothing to read", not "not running". The lane hit exactly this: the `p40_256k` and
    `gpu0_256k` envelope servers this check was written against were removed between
    sessions, and `docker logs p40probe` answers
    `Error response from daemon: No such container`.

    Returns (name, running, running_names, exists, status_line).
    """
    rc, out, err = run(["docker", "ps", "--format", "{{.Names}}"], timeout=30)
    running = out.split() if rc == 0 else []
    rc_all, out_all, _err_all = run(
        ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}"], timeout=30
    )
    status_by_name = {}
    if rc_all == 0:
        for line in out_all.splitlines():
            if "\t" in line:
                container_name, status = line.split("\t", 1)
                status_by_name[container_name.strip()] = status.strip()

    def _resolved(container_name):
        return (container_name, container_name in running, running,
                container_name in status_by_name, status_by_name.get(container_name))

    if explicit:
        return _resolved(explicit)
    for name in CANDIDATES:
        if name in running:
            return _resolved(name)
    # Nothing running: prefer a candidate that at least still EXISTS (stopped but readable)
    for name in CANDIDATES:
        if name in status_by_name:
            return _resolved(name)
    return _resolved(CANDIDATES[0])


def inspect_container(name):
    rc, out, err = run(["docker", "inspect", name], timeout=60)
    if rc != 0:
        return {"error": (err or out).strip()[:300]}
    try:
        data = json.loads(out)[0]
    except (ValueError, IndexError) as exc:
        return {"error": repr(exc)}
    state = data.get("State", {})
    config = data.get("Config", {})
    host_config = data.get("HostConfig", {})
    return {
        "image": config.get("Image"),
        "cmd": config.get("Cmd"),
        "entrypoint": config.get("Entrypoint"),
        "status": state.get("Status"),
        "running": state.get("Running"),
        "health": (state.get("Health") or {}).get("Status"),
        "started_at": state.get("StartedAt"),
        "device_requests": host_config.get("DeviceRequests"),
        "port_bindings": host_config.get("PortBindings"),
    }


def slot_state(server):
    """/slots if the server exposes it; otherwise a log-derived guess."""
    try:
        payload = server._get("/slots", timeout=15)  # noqa: SLF001 - deliberate probe
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}
    if isinstance(payload, list):
        return {
            "available": True,
            "slots": [
                {"id": slot.get("id"), "is_processing": slot.get("is_processing"),
                 "prompt_tokens": slot.get("n_prompt_tokens") or slot.get("prompt_n")}
                for slot in payload
            ],
        }
    return {"available": True, "raw_type": type(payload).__name__}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--container", default=None)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--host-url", default="http://127.0.0.1")
    parser.add_argument("--tag", default="P40-256K")
    parser.add_argument("--out-dir", default="/home/cereal/omni-campaign/export/probe")
    parser.add_argument("--log-tail", type=int, default=400)
    parser.add_argument("--log-head", type=int, default=600,
                        help="lines of the LOAD banner to read (flash_attn/KV/offload live there)")
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--queue-anyway", action="store_true",
                        help="send the sanity prompts even if a slot is mid-request "
                             "(they queue behind it; can take a long time)")
    parser.add_argument("--skip-completion", action="store_true")
    parser.add_argument("--r3-control", default=None,
                        help="fa_negative_control.py receipt that licenses (or voids) the "
                             "R3 quantized-KV inference on this build")
    args = parser.parse_args(argv)
    # --tag becomes the receipt filename; validated here too (see llama_probe.safe_tag).
    args.tag = safe_tag(args.tag)

    name, running, all_running, exists, status_line = resolve_container(args.container)
    os.makedirs(args.out_dir, exist_ok=True)

    receipt = {
        "kind": "p40probe_check",
        "created_at": utc_now(),
        "tag": args.tag,
        "container_requested": args.container,
        "container_resolved": name,
        "container_running": running,
        "container_exists": exists,
        "container_status_line": status_line,
        "containers_running": all_running,
        "read_only": True,
        "read_only_scope": "container lifecycle only",
        "note": (
            "LIFECYCLE read-only: this check never stops, kills, restarts or removes a "
            "container. It is NOT resource read-only -- unless --skip-completion is given "
            "it sends short generation requests into a server another lane owns, which "
            "occupies a slot and GPU time while they run."
        ),
    }

    # Inspect whenever the object exists: a stopped container still carries its image,
    # command line, device request and exit state, and those are the envelope evidence.
    receipt["inspect"] = (
        inspect_container(name) if exists
        else {"error": "container does not exist", "absent": True}
    )
    if not exists:
        rc_missing, out_missing, err_missing = run(["docker", "inspect", name], timeout=30)
        rc_logs, out_logs, err_logs = run(["docker", "logs", name], timeout=30)
        receipt["container_absent_evidence"] = {
            "docker_inspect_rc": rc_missing,
            "docker_inspect_stderr": (err_missing or out_missing).strip()[:300],
            "docker_logs_rc": rc_logs,
            "docker_logs_stderr": (err_logs or out_logs).strip()[:300],
            "reading": (
                "the container named %r is not present on this host in any state (running "
                "or exited), so there is no log, no launch command and no server to send a "
                "sanity completion to. Nothing was started to replace it: that decision "
                "belongs to whoever owns the card." % name
            ),
        }

    # The flash-attention / KV / offload banner is printed ONCE at load and then scrolls
    # away behind per-request progress lines, so the head of the log is what carries the
    # envelope evidence; the tail carries the current state.
    head_text = docker_logs(name, head=args.log_head) if exists else ""
    tail_text = docker_logs(name, tail=args.log_tail) if exists else ""
    load_analysis = analyse_server_log(head_text)
    tail_analysis = analyse_server_log(tail_text)
    receipt["log_analysis"] = load_analysis
    receipt["log_analysis_tail"] = {
        "slots_idle_seen": tail_analysis["slots_idle_seen"],
        "fallback_evidence": tail_analysis["fallback_evidence"],
        "line_count": tail_analysis["line_count"],
    }
    receipt["log_head_lines"] = head_text.splitlines()[:60]
    receipt["log_tail_lines"] = tail_text.splitlines()[-30:]

    # The requested envelope comes from the container's own command line; recent llama.cpp
    # builds no longer print the flash_attn / KV banner into the server log at all.
    inspect = receipt["inspect"] if isinstance(receipt.get("inspect"), dict) else {}
    launch_command = (inspect.get("entrypoint") or []) + (inspect.get("cmd") or [])
    receipt["launch_command"] = launch_command
    receipt["requested_flags"] = parse_server_flags(launch_command)
    receipt["requested_flags_source"] = (
        "docker_inspect" if receipt["requested_flags"] else "cli_defaults_unverified"
    )
    r3_control = load_r3_control(args.r3_control)
    receipt["r3_control"] = r3_control
    receipt["flash_attn_evidence"] = flash_attn_evidence(
        load_analysis, receipt["requested_flags"],
        flags_source=receipt["requested_flags_source"], r3_control=r3_control,
    )
    # The R3 negative control was run against the binary THIS LANE built. A container
    # running someone else's image is a different binary, so say so rather than letting the
    # licence look like it was earned on the image being inspected.
    if r3_control:
        receipt["r3_control_scope_note"] = (
            "the R3 premise was verified on the lane's own build (%s); this container runs "
            "image %r, so the premise is INHERITED across builds here, not measured on it. "
            "Re-run fa_negative_control.py with that image to make it local."
            % (r3_control.get("llama_cpp_commit"), inspect.get("image"))
        )

    rc, smi, _ = run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        timeout=30,
    )
    receipt["nvidia_smi"] = parse_nvidia_smi_csv(smi) if rc == 0 else {"error": "nvidia-smi rc=%d" % rc}

    base_url = "%s:%d" % (args.host_url, args.port)
    server = LlamaServer(base_url, timeout=args.request_timeout)
    receipt["health"] = server.health()
    receipt["server_url"] = base_url

    props = server.props()
    receipt["props_model_path"] = props.get("model_path") if isinstance(props, dict) else None
    receipt["props_n_ctx"] = props_n_ctx(props)

    slots = slot_state(server)
    receipt["slots"] = slots
    busy = False
    if slots.get("available") and slots.get("slots"):
        busy = any(bool(slot.get("is_processing")) for slot in slots["slots"])
    receipt["slot_busy"] = busy

    loaded = (
        receipt["health"].get("status") == "ok"
        and (receipt["log_analysis"]["server_listening"] or receipt["log_analysis"]["model_loaded_seen"]
             or receipt["props_n_ctx"] is not None)
    )
    receipt["finished_loading"] = bool(loaded)

    print("container=%s running=%s health=%s loaded=%s slot_busy=%s" % (
        name, running, receipt["health"].get("status"), loaded, busy))
    evidence = receipt["flash_attn_evidence"]
    print("requested: -fa %s  -ctk %s -ctv %s  -c %s  -np %s" % (
        evidence["requested_flash_attn"], evidence["requested_type_k"],
        evidence["requested_type_v"], receipt["requested_flags"].get("n_ctx"),
        receipt["requested_flags"].get("n_parallel")))
    print("n_ctx(log/props)=%s/%s kv_unified=%s kv_buffer_MiB=%s offloaded=%s/%s" % (
        receipt["log_analysis"]["n_ctx"], receipt["props_n_ctx"],
        receipt["log_analysis"]["kv_unified"], receipt["log_analysis"]["kv_buffer_mib_total"],
        receipt["log_analysis"]["layers_offloaded"], receipt["log_analysis"]["layers_total"]))
    print("FA reading: %s" % evidence["reading"])
    for item in evidence["basis"]:
        print("   basis: %s" % item)
    for item in receipt["log_analysis"]["fallback_evidence"][:8]:
        print("  fallback evidence: %s :: %s" % (item["signal"], item["line"][:150]))

    rows = []
    if args.skip_completion:
        receipt["completions_skipped"] = "--skip-completion"
    elif not exists:
        # Not "did not finish loading" -- there is no container and nothing on the port.
        receipt["completions_skipped"] = (
            "container %r does not exist on this host, so there was no server to send the "
            "short-context sanity prompts to" % name
        )
        print("container %s DOES NOT EXIST (not merely stopped); no completion sent" % name)
        print("  docker inspect said: %s"
              % receipt["container_absent_evidence"]["docker_inspect_stderr"])
    elif not loaded:
        receipt["completions_skipped"] = "server not finished loading"
        print("server has not finished loading; no completion sent")
    elif busy and not args.queue_anyway:
        receipt["completions_skipped"] = (
            "a slot is mid-request (an in-flight long-context run); pass --queue-anyway to queue behind it"
        )
        print(receipt["completions_skipped"])
    else:
        for item in SANITY_PROMPTS:
            started = time.time()
            try:
                result = server.completion(
                    item["prompt"], n_predict=item["n_predict"], cache_prompt=False,
                    timeout=args.request_timeout,
                )
            except Exception as exc:  # noqa: BLE001
                result = {"error": repr(exc), "wall_s": round(time.time() - started, 2)}
            rows.append({"id": item["id"], "prompt": item["prompt"], "result": result})
            print("%-5s %-6s pp=%s tg=%s :: %r" % (
                item["id"], result.get("stop_type"), result.get("prompt_per_second"),
                result.get("predicted_per_second"),
                (result.get("content") or result.get("error", ""))[:120]))
    receipt["sanity_completions"] = rows

    path = os.path.join(args.out_dir, "probe-%s.json" % args.tag)
    digest = write_receipt(path, receipt)
    print("\nprobe receipt: %s sha256=%s" % (path, digest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
