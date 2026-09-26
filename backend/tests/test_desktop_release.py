from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

import desktop_launcher
from backend.app.services.webview_profile import (
    OWNER_FILE,
    ProcessProbe,
    cleanup_profiles,
    profile_state,
)


def test_desktop_release_identifies_v37_data_only() -> None:
    assert "V3.7" in desktop_launcher.APP_TITLE
    assert desktop_launcher.APP_TITLE.endswith("Data Only")


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


def test_webview_cleanup_preserves_active_unknown_and_newest_stale(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    stale_old = data / "webview2-101"
    stale_new = data / "webview2-102"
    active = data / "webview2-103"
    unknown = data / "webview2-104"
    unrelated = data / "webview-cache"
    for path in (stale_old, stale_new, active, unknown, unrelated):
        path.mkdir()
    os.utime(stale_old, (1, 1))
    os.utime(stale_new, (2, 2))
    (active / OWNER_FILE).write_text(
        json.dumps({"pid": 103, "process_start_filetime": 300}), encoding="utf-8"
    )

    def probe(pid: int) -> ProcessProbe:
        return {
            101: ProcessProbe("absent"),
            102: ProcessProbe("absent"),
            103: ProcessProbe("active", 300),
            104: ProcessProbe("unknown"),
        }[pid]

    result = cleanup_profiles(data, process_probe=probe, retry_delays=())
    assert stale_old.exists() is False
    assert stale_new.exists()
    assert active.exists()
    assert unknown.exists()
    assert unrelated.exists()
    assert result["retained_stale"] == stale_new.name


def test_webview_pid_reuse_is_stale_but_bad_marker_fails_closed(tmp_path: Path) -> None:
    reused = tmp_path / "webview2-201"
    ambiguous = tmp_path / "webview2-202"
    reused.mkdir()
    ambiguous.mkdir()
    (reused / OWNER_FILE).write_text(
        json.dumps({"pid": 201, "process_start_filetime": 1}), encoding="utf-8"
    )
    (ambiguous / OWNER_FILE).write_text("not-json", encoding="utf-8")

    assert profile_state(
        reused, process_probe=lambda _pid: ProcessProbe("active", 2)
    ) == "stale"
    assert profile_state(
        ambiguous, process_probe=lambda _pid: ProcessProbe("active", 2)
    ) == "unknown"


def test_packaged_release_build_uses_consistent_snapshot_and_post_copy_audits() -> None:
    project_root = Path(__file__).resolve().parents[2]
    build_script = (project_root / "scripts" / "build-desktop.ps1").read_text(
        encoding="utf-8"
    )
    assert "snapshot_sqlite_database" in build_script
    assert "Copy-Item -LiteralPath \"data\\investment_lab.db\"" not in build_script
    assert "if (-not (Test-Path -LiteralPath $DistExecutable)" in build_script
    assert "PRAGMA integrity_check" in build_script
    assert "Path(r'$SourceDatabase'), Path(r'$ReleaseDatabase')" in build_script
    assert "Copy-Item -LiteralPath $DistExecutable -Destination $RootExecutable" in build_script
    assert "$rootHash -ne $releaseHash" in build_script
    assert 'Remove-Item -LiteralPath $BuildStaging -Recurse -Force' in build_script
    assert 'Remove-Item -LiteralPath $DistStaging -Recurse -Force' in build_script
    assert build_script.index("$rootHash -ne $releaseHash") < build_script.index(
        'Remove-Item -LiteralPath $DistStaging -Recurse -Force'
    )
    assert "V3.7 data-only desktop build created." in build_script
    assert build_script.index("snapshot_sqlite_database") < build_script.index(
        'Set-Content -LiteralPath $RootMarker'
    )
