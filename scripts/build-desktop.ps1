[CmdletBinding()]
param([switch]$StageOnly)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$Stage = Join-Path $ProjectRoot "dist\desktop-$Stamp"
$Work = Join-Path $ProjectRoot "build\desktop-$Stamp"
$Backup = Join-Path $ProjectRoot "desktop-backups\$Stamp"
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run setup first: local Python environment is missing.' }
Push-Location $ProjectRoot
try {
    npm.cmd run build --prefix frontend
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    & $Python -m PyInstaller --noconfirm --distpath $Stage --workpath $Work investment_lab.spec
    if ($LASTEXITCODE -ne 0) { throw 'Desktop build failed.' }
    $Release = Join-Path $Stage 'InvestmentLab'
    foreach ($Required in @('InvestmentLab.exe', '_internal\frontend\dist\index.html')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Release $Required))) { throw "Missing release resource: $Required" }
    }
    if ($StageOnly) { Write-Host "Staged: $Release"; return }
    if (Get-Process InvestmentLab -ErrorAction SilentlyContinue) { throw "Application is running. Staged release retained: $Release" }
    New-Item -ItemType Directory -Path $Backup -Force | Out-Null
    # Program files only. Never replace data, config, or the user's ledger.
    foreach ($Name in @('InvestmentLab.exe', '_internal')) {
        $Target = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $Name))
        if (-not $Target.StartsWith($ProjectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe release target.' }
        if (Test-Path -LiteralPath $Target) { Move-Item -LiteralPath $Target -Destination (Join-Path $Backup $Name) }
        try { Copy-Item -LiteralPath (Join-Path $Release $Name) -Destination $Target -Recurse }
        catch { throw "Publication interrupted; previous program retained at $Backup. $($_.Exception.Message)" }
    }
    Write-Host "Updated: $(Join-Path $ProjectRoot 'InvestmentLab.exe')"
    Write-Host "Recoverable previous program: $Backup"
    Write-Host 'No database, configuration, model, or ledger was replaced.'
} finally { Pop-Location }
