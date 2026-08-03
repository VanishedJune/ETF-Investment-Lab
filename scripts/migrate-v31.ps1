param(
    [string]$DatabasePath = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python runtime not found: $python"
}

Push-Location $projectRoot
try {
    if ($DatabasePath) {
        $resolvedDatabase = [System.IO.Path]::GetFullPath($DatabasePath)
        & $python -c "from backend.app.database.initialize import initialize_database; engine=initialize_database(r'$resolvedDatabase'); print('V3.1 schema migration complete:', r'$resolvedDatabase'); engine.dispose()"
    }
    else {
        & $python -c "from backend.app.database.initialize import initialize_database; engine=initialize_database(); print('V3.1 schema migration complete: project database'); engine.dispose()"
    }
}
finally {
    Pop-Location
}
