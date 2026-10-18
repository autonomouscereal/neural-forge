import json
src = "/mnt/lexar/omni-campaign/distill/media_m1/media_data_v4.json"
dst = "/mnt/lexar/omni-campaign/distill/media_m1/media_data_v5.json"
d = json.load(open(src))
d["notes"].append("v5 (2026-10-18): vision_in group_tokens 10000. vision_in is full-depth (all 64 layers of boundaries, the "
                  "C11 bridge trains) like music_codes; at the default 30000 its first group OOMed the 24 GB 3090 (step 35, 15:19:50Z).")
hit = [r for r in d["routes"] if r["route"] == "vision_in"]
assert len(hit) == 1
hit[0]["group_tokens"] = 10000
open(dst, "w").write(json.dumps(d, indent=1) + "\n")
print("wrote", dst, {r["route"]: r.get("group_tokens") for r in d["routes"]})
