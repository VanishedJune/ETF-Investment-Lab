[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$BackendApp = Join-Path $ProjectRoot "backend\app\main.py"

function Get-ListeningProcessId {
    param([int]$Port)

    foreach ($entry in (& netstat -ano -p tcp)) {
        if ($entry -match "^\s*TCP\s+127\.0\.0\.1:$Port\s+\S+\s+LISTENING\s+(\d+)\s*$") {
            return [int]$Matches[1]
        }
    }
    return $null
}

function Open-InvestmentLab {
    $url = "http://127.0.0.1:8765"
    & "$env:SystemRoot\System32\rundll32.exe" url.dll,FileProtocolHandler $url
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Virtual environment not found. Run setup.bat first."
}

if (-not (Test-Path -LiteralPath $BackendApp)) {
    throw "FastAPI entrypoint not found: $BackendApp. Re-run setup.bat or restore the backend folder."
}

$existingListener = Get-ListeningProcessId -Port 8765
if ($null -ne $existingListener) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8765" -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "ETF Investment Lab is already running. Opening the browser..."
            Open-InvestmentLab
            exit 0
        }
    }
    catch {
        throw "Port 8765 is occupied by another process and ETF Investment Lab did not respond."
    }
}

$launcherScript = Join-Path $PSScriptRoot "launch-server.bat"
if (-not (Test-Path -LiteralPath $launcherScript)) {
    throw "Server launcher not found: $launcherScript"
}
& $launcherScript
if ($LASTEXITCODE -ne 0) {
    throw "Unable to start the project Python launcher."
}
$ServerProcessId = $null

try {
    Write-Host "Starting ETF Investment Lab. The first launch may take a few minutes..."
    $started = $false
    for ($attempt = 0; $attempt -lt 600; $attempt++) {
        $ServerProcessId = Get-ListeningProcessId -Port 8765
        if ($null -ne $ServerProcessId) {
            $started = $true
            break
        }
        if ($attempt -gt 0 -and $attempt % 20 -eq 0) {
            Write-Host "." -NoNewline
        }
        Start-Sleep -Milliseconds 500
    }
    Write-Host ""

    if (-not $started) {
        throw "FastAPI did not bind http://127.0.0.1:8765 within five minutes."
    }

    Write-Host "ETF Investment Lab is running. Opening the browser..."
    Open-InvestmentLab
    Wait-Process -Id $ServerProcessId
}
finally {
    if ($null -ne $ServerProcessId -and (Get-Process -Id $ServerProcessId -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $ServerProcessId
    }
}
