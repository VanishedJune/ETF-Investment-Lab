[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) { throw "Virtual environment not found. Run setup.bat first." }
Push-Location $ProjectRoot
try {
    & $VenvPython -c "from pathlib import Path; from backend.app.services.backup_service import BackupService; print(BackupService(Path.cwd()).create()['path'])"
    if ($LASTEXITCODE -ne 0) { throw "Backup could not be created." }
}
finally { Pop-Location }
