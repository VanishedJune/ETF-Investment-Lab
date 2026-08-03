[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ReleaseRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$Executable = Join-Path $ReleaseRoot "InvestmentLab.exe"
$BackupsRoot = Join-Path $ReleaseRoot "data\backups"
$BackupPattern = '^(imported-)?[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}(-[0-9]{6})?$'

if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "InvestmentLab.exe was not found beside this restore script."
}
if (-not (Test-Path -LiteralPath $BackupsRoot -PathType Container)) {
    throw "No backup directory exists yet: $BackupsRoot"
}

$Available = @(
    Get-ChildItem -LiteralPath $BackupsRoot -Directory |
        Where-Object { $_.Name -match $BackupPattern } |
        Sort-Object Name -Descending
)
if ($Available.Count -eq 0) {
    throw "No restorable backup was found under data\backups."
}

Write-Host "Available backups:"
$Available | ForEach-Object { Write-Host "  $($_.Name)" }
$BackupName = Read-Host "Enter the exact backup folder name to restore"
if ($BackupName -notmatch $BackupPattern) {
    throw "The backup name has an invalid format."
}
if (-not (Test-Path -LiteralPath (Join-Path $BackupsRoot $BackupName) -PathType Container)) {
    throw "The selected backup does not exist."
}

& $Executable --restore $BackupName
if ($LASTEXITCODE -ne 0) {
    throw "Restore failed with exit code $LASTEXITCODE."
}
