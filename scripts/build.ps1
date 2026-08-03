[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$FrontendRoot = Join-Path $ProjectRoot "frontend"
$NodeModules = Join-Path $FrontendRoot "node_modules"

if (-not (Test-Path -LiteralPath $NodeModules)) {
    throw "Frontend dependencies are not installed. Run setup.bat first."
}

Push-Location $ProjectRoot
try {
    & npm.cmd run build --prefix $FrontendRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend build failed."
    }
}
finally {
    Pop-Location
}
