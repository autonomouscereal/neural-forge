set -euo pipefail

root=/home/cereal/model-surgery-lab/deployments/neural-forge-c8cfa511
archive=/home/cereal/model-surgery-lab/upload_staging/neural-forge-c8cfa511.zip
if [ -e "$root" ]; then
    printf 'deployment-root-already-exists=%s\n' "$root" >&2
    exit 42
fi
test -f "$archive"
printf '%s  %s\n' \
    'c92d6d92da5f9e205871b50f7a017eaac1c045eb10f364a2409750c8fb4dea0f' \
    "$archive" | sha256sum --check --strict
install -d -m 700 "$root"
mv "$archive" "$root/source.zip"
printf 'deployment_root=%s\n' "$root"
