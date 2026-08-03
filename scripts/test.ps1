[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Virtual environment not found. Run setup.bat first."
}

Push-Location $ProjectRoot
try {
    $PytestTemp = Join-Path $ProjectRoot ("data\pytest-test-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
    & $VenvPython -m pytest backend/tests -p no:cacheprovider --basetemp $PytestTemp
    if ($LASTEXITCODE -ne 0) {
        throw "Backend test suite failed."
    }

    & npm.cmd run build --prefix (Join-Path $ProjectRoot "frontend")
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend build failed."
    }

    & npm.cmd run test:e2e --prefix (Join-Path $ProjectRoot "frontend")
    if ($LASTEXITCODE -ne 0) {
        throw "Playwright test suite failed."
    }
}
finally {
    Pop-Location
}
