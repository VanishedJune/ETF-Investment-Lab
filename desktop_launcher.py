"""Windows desktop entrypoint for the packaged V3.7 data-only Investment Lab."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import socket
import sys
from threading import Thread
import time
from typing import Sequence
from urllib.request import urlopen


# Keep the V3.2 mutex name for cross-version exclusion: an older installed
# build and V3.4 must not write the same local database concurrently.
APP_TITLE = "ETF Investment Lab V3.7 · Data Only"
MUTEX_NAME = "Local\\ETFInvestmentLabV32SingleInstance"
ERROR_ALREADY_EXISTS = 183
WEBVIEW2_CLIENT_GUID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"


def _application_home() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _resource_home() -> Path:
    frozen = getattr(sys, "_MEIPASS", None)
    return Path(frozen).resolve() if frozen else Path(__file__).resolve().parent


def _message(message: str, *, error: bool = False) -> None:
    flags = 0x10 if error else 0x40
    ctypes.windll.user32.MessageBoxW(None, message, APP_TITLE, flags)


def _acquire_single_instance() -> int | None:
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    last_error = kernel32.GetLastError()
    if not handle:
        raise ctypes.WinError(last_error)
    if last_error == ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return None
    return int(handle)


def _webview2_version() -> str | None:
    try:
        import winreg
    except ImportError:
        return None
    views = (winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY)
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in views:
            try:
                with winreg.OpenKey(
                    hive,
                    rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_CLIENT_GUID}",
                    0,
                    winreg.KEY_READ | view,
                ) as runtime:
                    version, _ = winreg.QueryValueEx(runtime, "pv")
                    if version and str(version) != "0.0.0.0":
                        return str(version)
            except OSError:
                pass
            try:
                with winreg.OpenKey(
                    hive,
                    r"SOFTWARE\Microsoft\EdgeUpdate\Clients",
                    0,
                    winreg.KEY_READ | view,
                ) as clients:
                    for index in range(winreg.QueryInfoKey(clients)[0]):
                        child_name = winreg.EnumKey(clients, index)
                        with winreg.OpenKey(clients, child_name) as child:
                            product, _ = winreg.QueryValueEx(child, "name")
                            if "webview2" not in str(product).lower():
                                continue
                            version, _ = winreg.QueryValueEx(child, "pv")
                            if version and str(version) != "0.0.0.0":
                                return str(version)
            except OSError:
                continue
    return None


def _reserve_loopback_socket() -> socket.socket:
    """Bind a dynamic port and retain ownership until Uvicorn accepts it."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind(("127.0.0.1", 0))
        return listener
    except Exception:
        listener.close()
        raise


def _wait_until_ready(
    url: str,
    timeout: float = 90.0,
    server_thread: Thread | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if server_thread is not None and not server_thread.is_alive():
            raise RuntimeError("本地分析服务在完成启动前意外退出。")
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as error:  # server is still starting
            last_error = error
        time.sleep(0.25)
    raise RuntimeError(f"本地分析服务未能在{timeout:.0f}秒内启动：{last_error}")


def _parse_command(arguments: Sequence[str]) -> tuple[str, str | None]:
    values = list(arguments)
    if not values:
        return ("run", None)
    if values == ["--backup"]:
        return ("backup", None)
    if len(values) == 2 and values[0] == "--restore" and values[1].strip():
        return ("restore", values[1])
    raise ValueError("参数无效。可用参数：--backup 或 --restore <备份名称>。")


def _ensure_runtime_tree(home: Path) -> None:
    from backend.runtime import ensure_runtime_tree

    ensure_runtime_tree(home)


def _run_maintenance(command: tuple[str, str | None], home: Path) -> int:
    """Run backup/restore without migrating or seeding the source database."""

    from backend.app.services.backup_service import BackupService

    action, value = command
    backups = BackupService(home)
    if action == "backup":
        created = backups.create()
        _message(f"备份已创建：\n{created['path']}")
        return 0
    if action == "restore" and value is not None:
        restored = backups.restore(value)
        _message(
            f"已恢复备份：{restored['restored']}\n"
            f"恢复前状态已自动保存为：{restored['safety_backup']}"
        )
        return 0
    raise ValueError("不支持的维护操作。")


def _initialize_runtime(home: Path) -> None:
    from backend.app.database.initialize import initialize_database

    initialize_database(home / "data" / "investment_lab.db", home / "config")


def _check_schema_compatibility(home: Path, *, notify: bool = True) -> bool:
    """Refuse to open a database that was upgraded by a newer application."""

    import sqlite3

    from backend.app.database.migrations import SCHEMA_VERSION as APP_SCHEMA_VERSION

    database = home / "data" / "investment_lab.db"
    status_file = home / "data" / "startup-status.json"
    if not database.is_file():
        return True
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            db_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        finally:
            connection.close()
    except (sqlite3.Error, OSError) as error:
        status_file.write_text(
            json.dumps(
                {
                    "status": "SCHEMA_CHECK_ERROR",
                    "detail": str(error),
                    "app_schema_version": APP_SCHEMA_VERSION,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return False
    if db_version > APP_SCHEMA_VERSION:
        status_file.write_text(
            json.dumps(
                {
                    "status": "SCHEMA_MISMATCH",
                    "database_schema_version": db_version,
                    "app_schema_version": APP_SCHEMA_VERSION,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if notify:
            _message(
                "数据库已由更新版本的程序升级（schema "
                f"{db_version} > 当前程序 {APP_SCHEMA_VERSION}）。\n"
                "请使用配套的新版程序打开，不要用旧版覆盖。",
                error=True,
            )
        return False
    return True


def _run_desktop(home: Path) -> int:
    from backend.app.services.webview_profile import (
        cleanup_profiles,
        write_owner_marker,
    )

    version = _webview2_version()
    if version is None:
        _message(
            "未检测到Microsoft Edge WebView2 Runtime。\n\n"
            "请安装Microsoft Edge WebView2 Evergreen Runtime后重新启动。",
            error=True,
        )
        return 2

    # The GUI is the only mode that initializes/migrates the application DB.
    # A missing WebView2 runtime is detected before that first write.
    if not _check_schema_compatibility(home):
        return 66
    _initialize_runtime(home)
    import sqlite3

    try:
        connection = sqlite3.connect(f"file:{home / 'data' / 'investment_lab.db'}?mode=ro", uri=True)
        try:
            db_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        finally:
            connection.close()
    except (sqlite3.Error, OSError):
        db_version = None
    (home / "data" / "startup-status.json").write_text(
        json.dumps(
            {"status": "OK", "database_schema_version": db_version},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    import uvicorn
    import webview
    from backend.web import app

    data_dir = home / "data"
    # The cross-version mutex is already held here, so no other InvestmentLab
    # desktop instance can be using one of these profiles.  Keep the newest
    # confirmed-stale profile as a recovery fallback until this run succeeds.
    cleanup_profiles(data_dir, retain_newest_stale=True)
    listener = _reserve_loopback_socket()
    port = int(listener.getsockname()[1])
    url = f"http://127.0.0.1:{port}"
    port_file = home / "data" / "desktop-port.json"
    server = None
    server_thread: Thread | None = None
    try:
        port_file.write_text(
            json.dumps(
                {"port": port, "url": url, "pid": os.getpid()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
            timeout_graceful_shutdown=10,
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None
        server_thread = Thread(
            target=server.run,
            kwargs={"sockets": [listener]},
            name="investment-lab-server",
            daemon=True,
        )
        server_thread.start()
        _wait_until_ready(url, server_thread=server_thread)

        window = webview.create_window(
            APP_TITLE,
            url,
            width=1500,
            height=960,
            min_size=(1050, 700),
            background_color="#f2eee5",
            text_select=True,
        )

        def stop_server() -> None:
            if server is not None:
                server.should_exit = True

        window.events.closed += stop_server
        # Keep this desktop instance out of the shared WebView2 profile.  The
        # default profile can be locked by another Edge/WebView2 application
        # (or by a previously crashed review process), which makes
        # CoreWebView2Initialization fail with 0x800700AA.  A per-process
        # directory keeps startup deterministic while the application data
        # itself remains in the SQLite database under ``home / data``.
        webview_storage = data_dir / f"webview2-{os.getpid()}"
        write_owner_marker(
            webview_storage,
            launcher_path=Path(sys.executable if getattr(sys, "frozen", False) else __file__),
        )
        webview.start(
            gui="edgechromium",
            debug=False,
            private_mode=False,
            storage_path=str(webview_storage),
        )
        return 0
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=15)
        listener.close()
        port_file.unlink(missing_ok=True)
        # WebView2 child processes can release files slightly after the window
        # closes.  Removal is retry-based and failures never mask app shutdown.
        cleanup_profiles(
            data_dir,
            keep_profile=locals().get("webview_storage"),
            retain_newest_stale=False,
        )


def main(arguments: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if arguments is None else arguments)
    if len(args) == 2 and args[0] == '--check-data-ui':
        # Read-only packaged smoke check: no lifespan, migration, model, or GUI.
        import sqlite3
        home, resources = _application_home(), _resource_home()
        os.environ['INVESTMENT_LAB_HOME'] = str(home)
        os.environ['INVESTMENT_LAB_RESOURCE_ROOT'] = str(resources)
        try:
            from backend.app.schemas.investment_calendar import PositionEventCreate
            from backend.app.services.instrument_universe import AI_ASSISTANT_ETFS
            from backend.web import app
            index = resources / 'frontend/dist/index.html'
            markup = index.read_text(encoding='utf-8')
            assert '/assets/index.js' in markup
            assert (resources / 'frontend/dist/assets/index.js').is_file()
            with sqlite3.connect((home / 'data/investment_lab.db').as_uri() + '?mode=ro', uri=True) as conn:
                rows = conn.execute('SELECT instrument_code,slot_order FROM v351_instrument_slots WHERE active=1 ORDER BY slot_order').fetchall()
            assert [row[0] for row in rows] == list(AI_ASSISTANT_ETFS)
            PositionEventCreate(instrument_code='515220', direction='increase', operation_date='2026-09-26', change_percent=5)
            report = {'status': 'passed', 'slots': rows, 'bundled_frontend': True,
                      'backend_routes': len(app.routes), 'database_mode': 'read_only', 'gui_tested': False}
            result = 0
        except Exception as error:
            report = {'status': 'failed', 'error': str(error)}
            result = 1
        Path(args[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return result
    if os.name != "nt":
        raise RuntimeError("桌面版仅支持Windows。")
    try:
        command = _parse_command(sys.argv[1:] if arguments is None else arguments)
    except ValueError as error:
        _message(str(error), error=True)
        return 64

    try:
        mutex = _acquire_single_instance()
    except OSError as error:
        _message(f"无法建立单实例保护：\n\n{error}", error=True)
        return 1
    if mutex is None:
        if command[0] == "run":
            _message("投资研究台已经在运行，请切换到已有窗口。")
            return 0
        _message("执行备份或恢复前，请先关闭正在运行的投资研究台。", error=True)
        return 3

    try:
        home = _application_home()
        resources = _resource_home()
        os.environ["INVESTMENT_LAB_HOME"] = str(home)
        os.environ["INVESTMENT_LAB_RESOURCE_ROOT"] = str(resources)
        os.chdir(home)

        _ensure_runtime_tree(home)
        if command[0] != "run":
            return _run_maintenance(command, home)
        return _run_desktop(home)
    except Exception as error:
        _message(f"投资研究台启动失败：\n\n{error}", error=True)
        return 1
    finally:
        ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(mutex))


if __name__ == "__main__":
    raise SystemExit(main())
