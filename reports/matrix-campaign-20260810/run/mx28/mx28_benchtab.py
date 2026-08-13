import json, glob, os
OUT = "/mnt/ai_data/neural-forge-matrix/wave28/bench"
rows = {}
for path in glob.glob(OUT + "/*/**/results_*.json", recursive=True) + glob.glob(OUT + "/*/results_*.json"):
    body = json.load(open(path))
    label = path[len(OUT) + 1:].split("/")[0]
    r = body["results"]
    rows[label] = (
        r["arc_easy"]["acc,none"], r["hellaswag"]["acc_norm,none"], r["piqa"]["acc_norm,none"],
        r["winogrande"]["acc,none"], r["lambada_openai"]["acc,none"],
        r["lambada_openai"]["perplexity,none"], r["wikitext"]["word_perplexity,none"],
    )
print("%-14s %7s %7s %7s %7s %7s %9s %9s" % ("model", "arc_e", "hella_n", "piqa_n", "wino", "lambada", "lam_ppl", "wiki_wppl"))
for label in sorted(rows):
    v = rows[label]
    print("%-14s %7.4f %7.4f %7.4f %7.4f %7.4f %9.2f %9.2f" % (label, *v))
