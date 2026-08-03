[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$RequirementsLock = Join-Path $ProjectRoot "backend\requirements.lock.txt"
$FrontendRoot = Join-Path $ProjectRoot "frontend"

function Assert-Python311OrNewer {
    param(
        [string]$PythonExecutable,
        [string[]]$PythonArguments = @()
    )

    $version = & $PythonExecutable @PythonArguments -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not determine the Python version for $PythonExecutable."
    }

    $match = [regex]::Match($version.Trim(), '^(?<major>\d+)\.(?<minor>\d+)$')
    if (-not $match.Success) {
        throw "Could not parse the Python version reported by ${PythonExecutable}: $version"
    }

    $major = [int]$match.Groups['major'].Value
    $minor = [int]$match.Groups['minor'].Value
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
        throw "Python 3.11+ is required; found Python $version at $PythonExecutable."
    }
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $SystemPython = Get-Command python -ErrorAction SilentlyContinue
    $PythonLauncher = Get-Command py -ErrorAction SilentlyContinue

    if ($null -ne $SystemPython) {
        $PythonCommand = $SystemPython.Source
        $PythonArguments = @()
    }
    elseif ($null -ne $PythonLauncher) {
        $PythonCommand = $PythonLauncher.Source
        $PythonArguments = @("-3")
    }
    else {
        throw "Python 3.11+ is required to create .venv. Install Python 3.11 or newer and run setup again."
    }

    Assert-Python311OrNewer -PythonExecutable $PythonCommand -PythonArguments $PythonArguments
    & $PythonCommand @PythonArguments -m venv $VenvPath

    if ($LASTEXITCODE -ne 0) {
        throw "Python could not create the virtual environment."
    }
}

if (-not (Test-Path -LiteralPath $RequirementsLock)) {
    throw "Resolved dependency lock file not found: $RequirementsLock"
}

Assert-Python311OrNewer -PythonExecutable $VenvPython
& $VenvPython -m pip install --disable-pip-version-check --no-input --requirement $RequirementsLock
if ($LASTEXITCODE -ne 0) {
    throw "pip could not install the resolved backend dependency lock."
}

Push-Location $ProjectRoot
try {
    & $VenvPython -c "from backend.runtime import ensure_runtime_tree; from backend.app.database.initialize import initialize_database; ensure_runtime_tree('.'); initialize_database()"
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime directories could not be initialized."
    }

    & npm.cmd ci --prefix $FrontendRoot
    if ($LASTEXITCODE -ne 0) {
        throw "npm could not install frontend dependencies."
    }

    & npm.cmd run build --prefix $FrontendRoot
    if ($LASTEXITCODE -ne 0) {
        throw "The frontend could not be built."
    }
}
finally {
    Pop-Location
}

Write-Host "Setup complete. Existing data, configuration, and reports were left unchanged."
