# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules


webview_datas, webview_binaries, webview_hidden = collect_all("webview")
hiddenimports = webview_hidden
for package in ("uvicorn", "akshare", "exchange_calendars", "tushare"):
    hiddenimports += collect_submodules(package)
akshare_datas = collect_data_files("akshare")

# The data-only FastAPI application and ETF replacement service are explicit so
# the packaged desktop build contains the same chart/calendar workflow as the
# source build.  Legacy model tables remain in SQLite for compatibility, but
# their endpoints are blocked by the V3.7 runtime guard.
hiddenimports += [
    "backend.web",
    "backend.app.services.instrument_universe",
    "backend.app.services.v351_slot_service",
    "backend.app.services.indicator_service",
    "backend.app.services.market_data",
]

a = Analysis(
    ["desktop_launcher.py"],
    pathex=["."],
    binaries=webview_binaries,
    datas=[("frontend/dist", "frontend/dist"), *webview_datas, *akshare_datas],
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
