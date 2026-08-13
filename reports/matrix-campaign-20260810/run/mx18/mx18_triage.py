#!/usr/bin/env python3
"""One line per lane: termination, raw length, collapsed length.  Triage only --
it decides nothing and is only a reading order for the frozen text itself."""
import json, sys, os
sys.path.insert(0, os.environ["MX18"])
from mx18_read_dossier import collapsed

for path in sorted(sys.argv[1:]):
    body = json.loads(open(path).read())
    rows = body["child_responses"]
    caps = sum(1 for r in rows if r["termination"] == "censored-at-cap")
    thin = sum(1 for r in rows if len(collapsed(r["response_verbatim"])) < 350)
    lens = " ".join("%5d" % len(collapsed(r["response_verbatim"])) for r in rows)
    print("%-16s caps %2d/16 thin %2d/16  %s" % (body["cell_id"], caps, thin, lens))
