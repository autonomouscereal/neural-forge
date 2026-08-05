$ErrorActionPreference = "Stop"

$python = "C:\Python314\python.exe"
$ssh = "C:\Users\cereal\.agents\skills\server-manager\ssh_client.py"
$localRoot = "C:\Users\cereal\Documents\ai training\neural-forge"
$archive = "$localRoot\upload_staging\neural-forge-a884b2d1.zip"
$review = "$localRoot\.codex-test-runs\review-v5-c8cfa511.sh"
$prompts = "$localRoot\research\lfm2_semantic_whole_response_prompts_v1.json"
$remoteDeploy = "$localRoot\.codex-test-runs\deploy-v5-review-a884b2d1.sh"
$remoteRoot = "/home/cereal/model-surgery-lab/upload_staging"

if (-not (Test-Path -LiteralPath $archive)) {
    Push-Location $localRoot
    try {
        & git archive --format=zip --output=$archive a884b2d1 -- `
            src experiments research pyproject.toml `
            reports/r12-certified-20260804/V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md
        if ($LASTEXITCODE -ne 0) {
            throw "Review deployment archive creation failed"
        }
    }
    finally {
        Pop-Location
    }
}

$expectedFiles = @{
    $archive = "6b985ff160054bafb135430dcfe80253b9f1fcdf6b394a627c42a18999a3b311"
    $review = "30018c3128122edabefe1431369bc1018c1953412d9b7f9730b7cdbff5c559c4"
    $prompts = "0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b"
}
foreach ($entry in $expectedFiles.GetEnumerator()) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $entry.Key).Hash
    if ($actual.ToLowerInvariant() -ne $entry.Value) {
        throw "Local deployment input digest mismatch: $($entry.Key)"
    }
}

& $python $ssh --server ai --timeout 600 --upload `
    $archive "$remoteRoot/neural-forge-a884b2d1.zip"
if ($LASTEXITCODE -ne 0) {
    throw "Review deployment archive upload failed"
}

& $python $ssh --server ai --timeout 600 --upload `
    $review "$remoteRoot/review-v5-c8cfa511.sh"
if ($LASTEXITCODE -ne 0) {
    throw "Review launcher upload failed"
}

& $python $ssh --server ai --timeout 600 --upload `
    $prompts "$remoteRoot/lfm2_semantic_whole_response_prompts_v1.a884b2d1.raw.json"
if ($LASTEXITCODE -ne 0) {
    throw "Frozen review prompt upload failed"
}

& $python $ssh --server ai --timeout 600 --command-timeout 300 `
    --command-file $remoteDeploy
if ($LASTEXITCODE -ne 0) {
    throw "Review deployment extraction or authentication failed"
}
