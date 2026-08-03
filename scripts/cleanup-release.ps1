[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$ManifestPath = Join-Path $ProjectRoot "audit\current-20260803\cleanup-manifest.json"
$Targets = [System.Collections.Generic.List[object]]::new()

function Add-CleanupTarget {
    param([string]$Path, [string]$Reason)

    if (-not (Test-Path -LiteralPath $Path)) { return }
    $Resolved = [System.IO.Path]::GetFullPath($Path)
    $Prefix = $ProjectRoot.TrimEnd('\') + '\'
    if (-not $Resolved.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup target escaped project root: $Resolved"
    }
    if ($Resolved -eq $ProjectRoot) {
        throw "Cleanup target cannot be the project root"
    }
    $Item = Get-Item -LiteralPath $Resolved -Force
    $Size = $null
    if (-not $Item.PSIsContainer) {
        $Size = [int64]$Item.Length
    }
    else {
        try {
            $Size = [int64](
                Get-ChildItem -LiteralPath $Resolved -File -Recurse -Force -ErrorAction Stop |
                    Measure-Object -Property Length -Sum
            ).Sum
        }
        catch {
            $Size = $null
        }
    }
    $Targets.Add([pscustomobject]@{
        path = $Resolved.Substring($Prefix.Length)
        absolute_path = $Resolved
        kind = if ($Item.PSIsContainer) { "directory" } else { "file" }
        size_bytes = $Size
        last_write_time = $Item.LastWriteTime.ToString("o")
        reason = $Reason
    })
}

Get-ChildItem -LiteralPath $ProjectRoot -Directory -Force | ForEach-Object {
    if (
        $_.Name -like ".pytest-*" -or
        $_.Name -like ".test_tmp_*" -or
        $_.Name -like ".v33-temp-*" -or
        $_.Name -eq "tmp3as4ixr0"
    ) {
        Add-CleanupTarget $_.FullName "temporary test or replay workspace"
    }
}

Add-CleanupTarget (Join-Path $ProjectRoot "build") "rebuildable PyInstaller work cache"
Add-CleanupTarget (Join-Path $ProjectRoot "frontend\test-results") "rebuildable Playwright failure artifacts"
Add-CleanupTarget (Join-Path $ProjectRoot "frontend\playwright-report") "rebuildable Playwright HTML report"
Add-CleanupTarget (Join-Path $ProjectRoot "audit\current-20260803\work") "validated test database workspace; production backup retained"
Add-CleanupTarget (Join-Path $ProjectRoot "audit\v344\work") "rebuildable packaging snapshot workspace"
Add-CleanupTarget (Join-Path $ProjectRoot "dist\InvestmentLab-root.exe") "obsolete slow-start one-file trial artifact"

$Manifest = [pscustomobject]@{
    generated_at = (Get-Date).ToString("o")
    project_root = $ProjectRoot
    preserved = @(
        "data\investment_lab.db",
        "data\model_iterations\399006",
        "data\model_iterations\NDX",
        "data\weekly_analysis_v2",
        "audit\current-20260803\backups",
        "reports",
        "InvestmentLab.exe",
        "_internal"
    )
    target_count = $Targets.Count
    targets = $Targets
}
New-Item -ItemType Directory -Path (Split-Path $ManifestPath) -Force | Out-Null
$Manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8

$DeletionResults = [System.Collections.Generic.List[object]]::new()
foreach ($Target in $Targets) {
    $Resolved = [string]$Target.absolute_path
    try {
        if ($Target.kind -eq "directory") {
            Remove-Item -LiteralPath $Resolved -Recurse -Force
        }
        else {
            Remove-Item -LiteralPath $Resolved -Force
        }
        $DeletionResults.Add([pscustomobject]@{
            path = $Target.path
            status = "removed"
            error = $null
        })
    }
    catch {
        $DeletionResults.Add([pscustomobject]@{
            path = $Target.path
            status = "blocked"
            error = $_.Exception.Message
        })
    }
}

$Manifest | Add-Member -NotePropertyName deletion_results -NotePropertyValue $DeletionResults
$Manifest | Add-Member -NotePropertyName removed_count -NotePropertyValue @($DeletionResults | Where-Object status -eq "removed").Count
$Manifest | Add-Member -NotePropertyName blocked_count -NotePropertyValue @($DeletionResults | Where-Object status -eq "blocked").Count
$Manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8

Write-Host "Cleanup manifest: $ManifestPath"
Write-Host "Removed targets: $($Manifest.removed_count)"
Write-Host "Blocked targets: $($Manifest.blocked_count)"
if ($Manifest.blocked_count -gt 0) { exit 2 }
