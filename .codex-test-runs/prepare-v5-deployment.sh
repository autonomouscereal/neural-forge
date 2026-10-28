set -euo pipefail
root=/home/cereal/model-surgery-lab/deployments/neural-forge-f7bbf16d
archive=/home/cereal/model-surgery-lab/upload_staging/neural-forge-f7bbf16d.zip
if [ -e "$root" ]; then
    echo "deployment-root-already-exists $root" >&2
    exit 42
fi
test -f "$archive"
printf '%s  %s\n' '1440b188d792d68a339c4891fb17d3627f2330be95275bc621ce12247de1cab0' "$archive" | sha256sum --check --strict
install -d -m 700 "$root"
mv "$archive" "$root/source.zip"
printf '%s\n' "$root"
