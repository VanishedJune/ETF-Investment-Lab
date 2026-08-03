from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
import socket
import sqlite3
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

import desktop_launcher


def test_desktop_release_identifies_v34_13w() -> None:
    assert desktop_launcher.APP_TITLE.endswith("V3.4-13W")


def test_desktop_command_parser_is_strict() -> None:
    assert desktop_launcher._parse_command([]) == ("run", None)
    assert desktop_launcher._parse_command(["--backup"]) == ("backup", None)
    assert desktop_launcher._parse_command(["--restore", "2026-08-01_12-00-00"]) == (
        "restore",
        "2026-08-01_12-00-00",
    )
    with pytest.raises(ValueError):
        desktop_launcher._parse_command(["--restore"])
    with pytest.raises(ValueError):
        desktop_launcher._parse_command(["--backup", "unexpected"])
    with pytest.raises(ValueError):
        desktop_launcher._parse_command(["--unknown"])


def test_dynamic_loopback_port_remains_reserved_until_server_owns_socket() -> None:
    listener = desktop_launcher._reserve_loopback_socket()
    try:
        host, port = listener.getsockname()
        assert host == "127.0.0.1"
        assert int(port) > 0
        contender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError):
                contender.bind((host, port))
        finally:
            contender.close()
    finally:
        listener.close()


def test_missing_webview2_is_reported_before_database_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[tuple[str, bool]] = []
    monkeypatch.setattr(desktop_launcher, "_webview2_version", lambda: None)
    monkeypatch.setattr(
        desktop_launcher,
        "_initialize_runtime",
        lambda _home: pytest.fail("database initialization ran before WebView2 gate"),
    )
    monkeypatch.setattr(
        desktop_launcher,
        "_message",
        lambda message, error=False: messages.append((message, error)),
    )

    assert desktop_launcher._run_desktop(tmp_path) == 2
    assert messages and messages[-1][1] is True
    assert "WebView2" in messages[-1][0]


def test_maintenance_backup_does_not_initialize_or_migrate_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    database = data / "investment_lab.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE marker (value INTEGER NOT NULL)")
        connection.execute("INSERT INTO marker VALUES (1)")

    monkeypatch.setattr(desktop_launcher, "_message", lambda *_args, **_kwargs: None)
    assert desktop_launcher._run_maintenance(("backup", None), tmp_path) == 0

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert tables == {"marker"}
    assert not (tmp_path / "config").exists()


def test_maintenance_mode_reports_busy_instance_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[tuple[str, bool]] = []
    monkeypatch.setattr(desktop_launcher, "_acquire_single_instance", lambda: None)
    monkeypatch.setattr(
        desktop_launcher,
        "_ensure_runtime_tree",
        lambda _home: pytest.fail("busy maintenance command touched the runtime tree"),
    )
    monkeypatch.setattr(
        desktop_launcher,
        "_message",
        lambda message, error=False: messages.append((message, error)),
    )

    assert desktop_launcher.main(["--backup"]) == 3
    assert messages[-1][1] is True
    assert "先关闭" in messages[-1][0]


@pytest.mark.skipif(not hasattr(ctypes, "windll"), reason="Windows mutex test")
def test_cross_version_mutex_name_excludes_a_second_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        desktop_launcher,
        "MUTEX_NAME",
        rf"Local\ETFInvestmentLabV32SingleInstance-Test-{uuid4()}",
    )
    first = desktop_launcher._acquire_single_instance()
    assert first is not None
    try:
        assert desktop_launcher._acquire_single_instance() is None
    finally:
        ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(first))


def test_embedded_server_uses_dynamic_port_and_preserves_restart_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "portable"
    resources = tmp_path / "resources"
    (home / "data").mkdir(parents=True)
    (resources / "frontend" / "dist").mkdir(parents=True)
    (resources / "frontend" / "dist" / "index.html").write_text(
        "<!doctype html><title>desktop smoke</title>", encoding="utf-8"
    )

    class _ClosedEvent:
        def __iadd__(self, _handler):
            return self

    fake_webview = SimpleNamespace(
        create_window=lambda *_args, **_kwargs: SimpleNamespace(
            events=SimpleNamespace(closed=_ClosedEvent())
        ),
        start=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "webview", fake_webview)
    monkeypatch.setattr(desktop_launcher, "_webview2_version", lambda: "test")
    monkeypatch.setenv("INVESTMENT_LAB_HOME", str(home))
    monkeypatch.setenv("INVESTMENT_LAB_RESOURCE_ROOT", str(resources))

    assert desktop_launcher._run_desktop(home) == 0
    database = home / "data" / "investment_lab.db"
    assert database.is_file()
    assert not (home / "data" / "desktop-port.json").exists()
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE desktop_persistence (value TEXT NOT NULL)")
        connection.execute("INSERT INTO desktop_persistence VALUES ('preserved')")

    assert desktop_launcher._run_desktop(home) == 0
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT value FROM desktop_persistence"
        ).fetchone() == ("preserved",)
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)
    assert not (home / "data" / "desktop-port.json").exists()


def test_packaged_release_build_uses_consistent_snapshot_and_post_copy_audits() -> None:
    project_root = Path(__file__).resolve().parents[2]
    build_script = (project_root / "scripts" / "build-desktop.ps1").read_text(
        encoding="utf-8"
    )
    assert "snapshot_sqlite_database" in build_script
    assert "Copy-Item -LiteralPath \"data\\investment_lab.db\"" not in build_script
    assert "if (Test-Path -LiteralPath $ReleaseDatabase)" in build_script
    assert "$ProductionDatabaseSource = $ReleaseDatabase" in build_script
    assert "packaging-production-snapshot.db" in build_script
    assert "PRAGMA integrity_check" in build_script
    assert "Path(r'$StagedReleaseDatabase'), Path(r'$ReleaseDatabase')" in build_script
    assert "Packaged V3.2 read-only baseline audit failed" in build_script
    assert "Packaged V3.3 model-state audit failed" in build_script
    assert build_script.index("snapshot_sqlite_database") < build_script.index(
        'Set-Content -LiteralPath $ReleaseMarker'
    )

    audit_script = (project_root / "scripts" / "audit-v34-state.py").read_text(
        encoding="utf-8"
    )
    assert "SUPPORTED_SCHEMA_VERSIONS = frozenset({20, 21, 22})" in audit_script
    assert '"schema_version_supported"' in audit_script
    assert '"schema_version_20"' not in audit_script
