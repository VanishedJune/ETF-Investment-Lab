param(
    [ValidateSet("Status", "Baseline", "Due")]
    [string]$Mode = "Status",
    [string]$ExecutedBy = "Codex",
    [string]$AsOf = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python runtime not found: $python"
}

$arguments = @(
    (Join-Path $PSScriptRoot "agent_iteration.py"),
    "--mode",
    $Mode.ToLowerInvariant(),
    "--executed-by",
    $ExecutedBy
)
if ($AsOf) {
    $arguments += @("--as-of", $AsOf)
}
& $python $arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
