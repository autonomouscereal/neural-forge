#!/usr/bin/env python3
"""proc_scan.py 1.0.0 - every path a live process holds or names (run as root: sudo -n python3 proc_scan.py).

Prints one JSON object:
  open:    paths from /proc/<pid>/{fd/*, cwd, exe, root} and the file column of /proc/<pid>/maps
  strings: [pid, kind, text] for each cmdline argument and environ value that contains '/'
  errors:  count of processes that vanished or could not be read
Stdlib only, read-only. As a plain user, root and container processes hide their fd/environ (which is why
the cleanup gate runs this under sudo).
"""
import json
import os

out_open: set[str] = set()
strings: list[list] = []
errors = 0
me = os.getpid()
for name in os.listdir("/proc"):
    if not name.isdigit() or int(name) == me:
        continue
    base = f"/proc/{name}"
    try:
        for link in ("cwd", "exe", "root"):
            try:
                out_open.add(os.readlink(f"{base}/{link}"))
            except OSError:
                pass
        try:
            for fd in os.listdir(f"{base}/fd"):
                try:
                    out_open.add(os.readlink(f"{base}/fd/{fd}"))
                except OSError:
                    pass
        except OSError:
            pass
        try:
            with open(f"{base}/maps", "r", errors="replace") as fh:
                for line in fh:
                    parts = line.split(None, 5)
                    if len(parts) == 6 and parts[5].startswith("/"):
                        out_open.add(parts[5].rstrip("\n").replace(" (deleted)", ""))
        except OSError:
            pass
        for kind in ("cmdline", "environ"):
            try:
                with open(f"{base}/{kind}", "rb") as fh:
                    for item in fh.read().split(b"\0"):
                        text = item.decode("utf-8", "replace")
                        if "/" in text:
                            strings.append([int(name), kind, text])
            except OSError:
                pass
    except Exception:  # a process can vanish between listdir and read
        errors += 1
print(json.dumps({"open": sorted(p for p in out_open if p.startswith("/")), "strings": strings, "errors": errors}))
