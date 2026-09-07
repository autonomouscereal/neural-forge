"""Duty-cycle one exact operator-owned allocation; never allocate, delete or kill.

This only pauses its own Docker job at the declared temperature, then resumes it
after cooling. Hardware thresholds must be read by the operator before launch.
"""
import argparse
import json
import re
import subprocess
import time
import uuid


def command(*args):
    return subprocess.check_output(args, text=True, timeout=20).strip()


def checked_state(container, allocation, gpu):
    item = json.loads(command("docker", "inspect", container))[0]
    labels = item["Config"].get("Labels", {})
    devices = [device for request in item["HostConfig"].get("DeviceRequests", [])
               for device in request.get("DeviceIDs", [])]
    if (item["Id"] != container or labels.get("ai.local.allocation") != allocation
            or labels.get("ai.local.owner") != "codex"
            or labels.get("ai.local.gpu-uuid") != gpu or devices != [gpu]):
        raise RuntimeError("allocation/container/device identity mismatch; refusing mutation")
    return item["State"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--pause-c", type=int, default=80)
    parser.add_argument("--resume-c", type=int, default=70)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{64}", args.container):
        parser.error("requires an immutable full Docker ID")
    if str(uuid.UUID(args.allocation)) != args.allocation or not args.gpu.startswith("GPU-"):
        parser.error("requires canonical allocation and exact GPU identity")
    uuid.UUID(args.gpu[4:])
    if not 30 <= args.resume_c < args.pause_c <= 85:
        parser.error("invalid cooling thresholds")
    state = lambda: checked_state(args.container, args.allocation, args.gpu)
    if state()["Paused"]:
        raise RuntimeError("refuse to adopt an already paused container")
    paused = False
    print(json.dumps({"event": "guard_started", **vars(args)}), flush=True)
    while True:
        current = state()
        if not current["Running"]:
            print(json.dumps({"event": "container_terminal", "state": current}), flush=True)
            return
        if current["Paused"] != paused:
            raise RuntimeError("external pause state change; refusing mutation")
        temperature = int(command("nvidia-smi", "-i", args.gpu,
                                  "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"))
        if not 0 <= temperature <= 110:
            raise RuntimeError("unusable GPU temperature reading")
        action = ("pause" if temperature >= args.pause_c and not paused else
                  "unpause" if temperature <= args.resume_c and paused else None)
        if action:
            before = state()
            if not before["Running"]:
                continue
            if before["Paused"] != paused:
                raise RuntimeError("external pause state change immediately before action")
            command("docker", action, args.container)
            paused = action == "pause"
            print(json.dumps({"event": action, "temperature_c": temperature,
                              "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
