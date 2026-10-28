$ErrorActionPreference = "Stop"

$python = "C:\\Python314\\python.exe"
$ssh = "C:\\Users\\cereal\\.agents\\skills\\server-manager\\ssh_client.py"
$localRoot = "C:\\Users\\cereal\\Documents\\ai training\\neural-forge\\.codex-test-runs"
$remoteRoot = "/home/cereal/model-surgery-lab/upload_staging"
$files = @(
    "materialize-v5-c8cfa511.sh",
    "watch-materialize-v5-c8cfa511.sh",
    "monitor-v5-c8cfa511-compact.sh"
)

foreach ($name in $files) {
    & $python $ssh --server ai --upload "$localRoot\\$name" "$remoteRoot/$name"
    if ($LASTEXITCODE -ne 0) {
        throw "Upload failed: $name"
    }
}

$remoteScript = @'
set -euo pipefail

umask 077
staging=/home/cereal/model-surgery-lab/upload_staging
artifact_root=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_moving_gauge_v5_20260804
watcher_pidfile="$artifact_root/watch-materialize-v5-c8cfa511.pid"
watcher_log="$artifact_root/watch-materialize-v5-c8cfa511-supervisor.log"

chmod 700 \
    "$staging/materialize-v5-c8cfa511.sh" \
    "$staging/watch-materialize-v5-c8cfa511.sh" \
    "$staging/monitor-v5-c8cfa511-compact.sh"
sha256sum \
    "$staging/materialize-v5-c8cfa511.sh" \
    "$staging/watch-materialize-v5-c8cfa511.sh" \
    "$staging/monitor-v5-c8cfa511-compact.sh"

if [ -s "$watcher_pidfile" ]; then
    watcher_pid=$(tr -d '[:space:]' < "$watcher_pidfile")
else
    watcher_pid=''
fi
if [ -n "$watcher_pid" ] && kill -0 "$watcher_pid" 2>/dev/null; then
    printf 'watcher_already_live=%s\n' "$watcher_pid"
else
    setsid nohup bash "$staging/watch-materialize-v5-c8cfa511.sh" \
        >"$watcher_log" 2>&1 </dev/null &
    watcher_pid=$!
    temporary="$watcher_pidfile.$$.tmp"
    printf '%s\n' "$watcher_pid" > "$temporary"
    mv -f "$temporary" "$watcher_pidfile"
    printf 'watcher_launched=%s\n' "$watcher_pid"
fi

bash "$staging/monitor-v5-c8cfa511-compact.sh"
'@

& $python $ssh --server ai --script $remoteScript
if ($LASTEXITCODE -ne 0) {
    throw "Remote watcher installation or compact monitor failed"
}
