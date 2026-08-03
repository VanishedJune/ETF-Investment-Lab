[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^(imported-)?[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}(-[0-9]{6})?$')]
    [string]$BackupName
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) { throw "Virtual environment not found. Run setup.bat first." }
Push-Location $ProjectRoot
try {
    & $VenvPython -c "from pathlib import Path; from backend.app.services.backup_service import BackupService; print(BackupService(Path.cwd()).restore('$BackupName'))"
    if ($LASTEXITCODE -ne 0) { throw "Restore could not be completed." }
}
finally { Pop-Location }
