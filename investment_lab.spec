# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_submodules


webview_datas, webview_binaries, webview_hidden = collect_all("webview")
hiddenimports = webview_hidden
for package in ("uvicorn", "akshare", "exchange_calendars", "tushare"):
    hiddenimports += collect_submodules(package)

# The V3.3 runtime is imported by the FastAPI application factory.  Keep its
# production modules explicit so a future lazy import cannot silently produce
# an EXE that starts but has no training/analysis endpoint.
hiddenimports += [
    "backend.web",
    "backend.app.services.v33_analysis_service",
    "backend.app.services.v33_data_service",
    "backend.app.services.v33_feature_service",
    "backend.app.services.v33_public_sources",
    "backend.app.services.v33_runtime_service",
    "backend.app.services.v33_training_service",
]

a = Analysis(
    ["desktop_launcher.py"],
    pathex=["."],
    binaries=webview_binaries,
    datas=[("frontend/dist", "frontend/dist"), *webview_datas],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "playwright"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="InvestmentLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="InvestmentLab",
)
