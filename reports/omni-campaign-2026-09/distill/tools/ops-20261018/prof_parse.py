import collections, sys
own = collections.Counter(); leafc = collections.Counter(); total = 0
for line in open(sys.argv[1]):
    stack, _, n = line.rstrip().rpartition(" ")
    n = int(n); total += n
    frames = stack.split(";")
    ours = [f for f in frames if "streamed_" in f]
    key = ours[-1].split(" (")[0] + " @" + ours[-1].split(":")[-1].rstrip(")") if ours else "(other)"
    own[key] += n
    leafc[frames[-1].split(" (")[0] + " [" + frames[-1].split("(")[-1].split(":")[0].split("/")[-1] + "]"] += n
print("samples", total)
print("-- deepest trainer frame")
for k, v in own.most_common(12): print(f"{100*v/total:5.1f}%  {k}")
print("-- leaf frame")
for k, v in leafc.most_common(12): print(f"{100*v/total:5.1f}%  {k}")
