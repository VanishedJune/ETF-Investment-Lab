[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ReleaseRoot = Join-Path $ProjectRoot "dist\InvestmentLab"
$ReleaseMarker = Join-Path $ReleaseRoot "V3.3-20W.release"
$V34ReleaseMarker = Join-Path $ReleaseRoot "V3.4-13W.release"
$ReleaseDatabase = Join-Path $ReleaseRoot "data\investment_lab.db"
$SourceDatabase = Join-Path $ProjectRoot "data\investment_lab.db"
$PackagingWorkRoot = Join-Path $ProjectRoot "audit\v344\work"
$StagedReleaseDatabase = Join-Path $PackagingWorkRoot "packaging-production-snapshot.db"
$RootExecutableBuild = Join-Path $ReleaseRoot "InvestmentLab.exe"
$RootExecutable = Join-Path $ProjectRoot "InvestmentLab.exe"
$RootInternalBuild = Join-Path $ReleaseRoot "_internal"
$RootInternal = Join-Path $ProjectRoot "_internal"
$RootReleaseMarker = Join-Path $ProjectRoot "V3.4-13W.release"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found. Run setup.bat first."
}

& $Python -c "import PyInstaller, webview"
if ($LASTEXITCODE -ne 0) {
    throw "Desktop build dependencies are missing. Install backend\desktop-requirements.txt first."
}

Push-Location $ProjectRoot
try {
    # The installed portable database is the production database once a release
    # exists.  Preserve it before PyInstaller recreates dist\InvestmentLab.
    # Falling back to data\investment_lab.db is only valid for a first build.
    $ProductionDatabaseSource = $SourceDatabase
    if (Test-Path -LiteralPath $ReleaseDatabase) {
        $ProductionDatabaseSource = $ReleaseDatabase
    }
    New-Item -ItemType Directory -Path $PackagingWorkRoot -Force | Out-Null
    & $Python -c "from pathlib import Path; from backend.app.services.backup_service import snapshot_sqlite_database; snapshot_sqlite_database(Path(r'$ProductionDatabaseSource'), Path(r'$StagedReleaseDatabase'))"
    if ($LASTEXITCODE -ne 0) { throw "The production database could not be staged with SQLite online backup." }
    & $Python -c "import sqlite3; connection=sqlite3.connect(r'$StagedReleaseDatabase'); integrity=connection.execute('PRAGMA integrity_check').fetchone()[0]; quick=connection.execute('PRAGMA quick_check').fetchone()[0]; foreign_keys=connection.execute('PRAGMA foreign_key_check').fetchall(); connection.close(); assert integrity == 'ok' and quick == 'ok' and not foreign_keys, (integrity, quick, foreign_keys[:5])"
    if ($LASTEXITCODE -ne 0) { throw "The staged production database failed integrity validation." }

    Remove-Item -LiteralPath $ReleaseMarker -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $V34ReleaseMarker -Force -ErrorAction SilentlyContinue

    & $Python "scripts\audit-v33-readonly-baseline.py" `
        --project-root $ProjectRoot `
        --database $StagedReleaseDatabase
    if ($LASTEXITCODE -ne 0) { throw "V3.2 read-only baseline audit failed." }

    & $Python "scripts\audit-v33-model-state.py" `
        --db $StagedReleaseDatabase `
        --report "reports\V33_MODEL_STATE_AUDIT.md"
    if ($LASTEXITCODE -ne 0) { throw "V3.3 persisted model-state audit failed." }

    & $Python "scripts\audit-v34-state.py" --db $StagedReleaseDatabase --report "reports\V34_MODEL_STATE_AUDIT.json"
    if ($LASTEXITCODE -ne 0) { throw "V3.4 persisted model-state audit failed." }

    & $Python -c "from backend.web import app; import backend.app.services.v33_runtime_service; import backend.app.services.v33_public_sources; required={'/api/v33/model/status','/api/v33/model/train','/api/v33/model/analysis','/api/v33/iterations/{market}','/api/v33/forecast/{market}'}; paths={route.path for route in app.routes}; missing=sorted(required-paths); assert not missing, f'Missing V3.3 routes: {missing}'"
    if ($LASTEXITCODE -ne 0) { throw "V3.3 production module/route check failed." }

    & $Python -c "from backend.web import app; import backend.app.services.v34_runtime_service; required={'/api/v34/model/status','/api/v34/model/train','/api/v34/model/analysis','/api/v34/iterations/{market}','/api/v34/forecast/{market}','/api/v34/training/{market}/bootstrap','/api/v34/training/{market}/incremental','/api/v34/training/runs/{run_id}','/api/v34/models/{market}/champion'}; paths={route.path for route in app.routes}; missing=sorted(required-paths); assert not missing, f'Missing V3.4 routes: {missing}'"
    if ($LASTEXITCODE -ne 0) { throw "V3.4 production module/route check failed." }

    & npm.cmd run build --prefix frontend
    if ($LASTEXITCODE -ne 0) { throw "Frontend production build failed." }

    & $Python -m PyInstaller --clean --noconfirm investment_lab.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller desktop build failed." }
    if (-not (Test-Path -LiteralPath $RootExecutableBuild) -or -not (Test-Path -LiteralPath $RootInternalBuild)) {
        throw "PyInstaller did not create the complete one-folder desktop runtime."
    }
    Copy-Item -LiteralPath $RootExecutableBuild -Destination $RootExecutable -Force
    if (Test-Path -LiteralPath $RootInternal) {
        Remove-Item -LiteralPath $RootInternal -Recurse -Force
    }
    Copy-Item -LiteralPath $RootInternalBuild -Destination $RootInternal -Recurse -Force
    Set-Content -LiteralPath $RootReleaseMarker -Value "V3.4-13W" -Encoding UTF8

    New-Item -ItemType Directory -Path (Join-Path $ReleaseRoot "data") -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $ReleaseRoot "config") -Force | Out-Null
    & $Python -c "from pathlib import Path; from backend.app.services.backup_service import snapshot_sqlite_database; snapshot_sqlite_database(Path(r'$StagedReleaseDatabase'), Path(r'$ReleaseDatabase'))"
    if ($LASTEXITCODE -ne 0) { throw "A consistent release database snapshot could not be created." }
    foreach ($auditFolder in @("model_iterations", "weekly_analysis_v2")) {
        $source = Join-Path $ProjectRoot "data\$auditFolder"
        if (Test-Path -LiteralPath $source) {
            Copy-Item -LiteralPath $source -Destination (Join-Path $ReleaseRoot "data") -Recurse -Force
        }
    }
    Copy-Item -Path "config\*" -Destination (Join-Path $ReleaseRoot "config") -Recurse -Force
    # Keep release artifact names ASCII-only. Windows PowerShell 5.1 reads
    # UTF-8 scripts without a BOM through the active ANSI code page, which can
    # corrupt non-ASCII argv/path values before Python receives them.
    Copy-Item -LiteralPath "scripts\release-backup.bat" -Destination (Join-Path $ReleaseRoot "Create-Data-Backup.bat") -Force
    Copy-Item -LiteralPath "scripts\release-restore.bat" -Destination (Join-Path $ReleaseRoot "Restore-Data-Backup.bat") -Force
    Copy-Item -LiteralPath "scripts\release-restore.ps1" -Destination (Join-Path $ReleaseRoot "restore-data.ps1") -Force
    Copy-Item -LiteralPath "docs\V33_DESKTOP_README.txt" -Destination (Join-Path $ReleaseRoot "README.txt") -Force
    Copy-Item -LiteralPath "docs\V33_MODEL_PROTOCOL.md" -Destination (Join-Path $ReleaseRoot "V3.3-MODEL-PROTOCOL.md") -Force

    # Audit the exact portable snapshot, not only the source database that
    # existed before the potentially long PyInstaller build.
    & $Python "scripts\audit-v33-readonly-baseline.py" `
        --project-root $ReleaseRoot `
        --database $ReleaseDatabase `
        --freeze (Join-Path $ProjectRoot "reports\V33_BASELINE_FREEZE.md") `
        --report (Join-Path $ReleaseRoot "V3.2-readonly-baseline-audit.md")
    if ($LASTEXITCODE -ne 0) { throw "Packaged V3.2 read-only baseline audit failed." }

    & $Python "scripts\audit-v33-model-state.py" `
        --db $ReleaseDatabase `
        --report (Join-Path $ReleaseRoot "V3.3-model-state-audit.md")
    if ($LASTEXITCODE -ne 0) { throw "Packaged V3.3 model-state audit failed." }

    & $Python "scripts\audit-v34-state.py" `
        --db $ReleaseDatabase `
        --report (Join-Path $ReleaseRoot "V3.4-model-state-audit.json")
    if ($LASTEXITCODE -ne 0) { throw "Packaged V3.4 model-state audit failed." }

    Set-Content -LiteralPath $ReleaseMarker -Value "V3.3-20W" -Encoding UTF8
    Set-Content -LiteralPath $V34ReleaseMarker -Value "V3.4-13W" -Encoding UTF8
}
finally {
    Pop-Location
}

Write-Host "Desktop release created: $ReleaseRoot\InvestmentLab.exe"
Write-Host "Root desktop executable created: $RootExecutable"
