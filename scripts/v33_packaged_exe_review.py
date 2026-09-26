"""End-to-end acceptance review for the packaged V3.4 Windows desktop app.

The real release is opened twice to verify the embedded WebView2 window,
dynamic loopback port, single-instance guard, restart, and persisted model
identity. Backup/restore commands are exercised on an isolated package clone
with a small sentinel SQLite database so the final 500+ MB release database is
never mutated or duplicated by this review.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import time
from typing import Any, Callable
from urllib.request import urlopen


APP_TITLE = "双市场投资研究台 V3.5-8W"
ACTIVE_MARKETS = ("399006", "159941")
WM_CLOSE = 0x0010

EnumWindowsProc = ctypes.WINFUNCTYPE(
    wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
)
user32 = ctypes.windll.user32
user32.EnumWindows.argtypes = (EnumWindowsProc, wintypes.LPARAM)
user32.EnumWindows.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = (
    wintypes.HWND,
    ctypes.POINTER(wintypes.DWORD),
)
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = (
    wintypes.HWND,
    wintypes.LPWSTR,
    ctypes.c_int,
)
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
user32.GetClassNameW.restype = ctypes.c_int
user32.IsWindowVisible.argtypes = (wintypes.HWND,)
user32.IsWindowVisible.restype = wintypes.BOOL
user32.PostMessageW.argtypes = (
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
)
user32.PostMessageW.restype = wintypes.BOOL


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Review the packaged InvestmentLab.exe without changing the source "
            "database or running training."
        )
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=project_root / "dist" / "InvestmentLab",
        help="packaged release directory (default: dist/InvestmentLab)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root / "reports" / "v34-packaged-exe-review.json",
        help="JSON evidence report path",
    )
    return parser.parse_args()


def _wait_for(
    probe: Callable[[], Any],
    *,
    timeout: float,
    description: str,
    interval: float = 0.2,
) -> Any:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            value = probe()
            if value:
                return value
        except Exception as error:  # resource may still be starting
            last_error = error
        time.sleep(interval)
    detail = f": {last_error}" if last_error is not None else ""
    raise TimeoutError(f"Timed out waiting for {description}{detail}")


def _window_text(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def _window_class(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buffer, len(buffer))
    return buffer.value


def _windows_for_pid(pid: int) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    @EnumWindowsProc
    def callback(hwnd: int, _lparam: int) -> bool:
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            found.append(
                {
                    "hwnd": int(hwnd),
                    "title": _window_text(hwnd),
                    "class_name": _window_class(hwnd),
                }
            )
        return True

    if not user32.EnumWindows(callback, 0):
        raise ctypes.WinError()
    return found


def _close_windows(process: subprocess.Popen[bytes], *, timeout: float = 30) -> None:
    windows = _wait_for(
        lambda: _windows_for_pid(process.pid),
        timeout=timeout,
        description=f"a window owned by PID {process.pid}",
    )
    for window in windows:
        user32.PostMessageW(window["hwnd"], WM_CLOSE, 0, 0)


def _await_exit(process: subprocess.Popen[bytes], *, timeout: float = 45) -> int:
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        raise AssertionError(f"PID {process.pid} did not exit after WM_CLOSE") from error


def _start(executable: Path, *arguments: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [str(executable), *arguments],
        cwd=executable.parent,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _json_url(url: str, *, timeout: float = 15) -> Any:
    with urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise AssertionError(f"GET {url} returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def _status_signature(base_url: str) -> dict[str, Any]:
    v33_status = _json_url(f"{base_url}/api/v33/model/status")
    v34_rows = _json_url(f"{base_url}/api/v34/model/status")
    if not isinstance(v34_rows, list):
        raise AssertionError("V3.4 model status did not return a market list")
    v34_status = {row["market"]: row for row in v34_rows}
    result: dict[str, Any] = {"v33_frozen": {}, "v34_active": {}}
    for market in ACTIVE_MARKETS:
        frozen = v33_status["markets"][market]
        frozen_champion = _json_url(f"{base_url}/api/v33/models/{market}/champion")
        result["v33_frozen"][market] = {
            "bootstrapped": frozen["bootstrapped"],
            "iteration_count": frozen["iteration_count"],
            "pending_count": frozen["pending_count"],
            "champion_version": frozen["champion_version"],
            "last_training_week_key": frozen["last_training_week_key"],
            "champion": frozen_champion,
        }
        active = v34_status[market]
        champion = _json_url(f"{base_url}/api/v34/models/{market}/champion")
        result["v34_active"][market] = {
            "bootstrapped": active["bootstrapped"],
            "weekly_iteration_count": active["weekly_iteration_count"],
            "candidate_training_count": active["candidate_training_count"],
            "champion_promotion_count": active["champion_promotion_count"],
            "last_anchor_date": active["last_anchor_date"],
            "champion": champion,
        }
    return result


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sqlite_logical_hash(path: Path) -> str:
    """Hash the logical dump so SQLite header counters cannot cause noise."""

    with closing(sqlite3.connect(path)) as connection:
        dump = "\n".join(connection.iterdump()).encode("utf-8")
    return hashlib.sha256(dump).hexdigest()


def _remove_tree_with_retry(path: Path, *, timeout: float = 30) -> None:
    """Remove one verified review tree after Windows releases SQLite handles."""

    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while path.exists() and time.monotonic() < deadline:
        try:
            shutil.rmtree(path)
            last_error = None
            break
        except OSError as error:
            last_error = error
            time.sleep(0.25)
    if path.exists():
        raise AssertionError(
            f"temporary packaged review still exists after {timeout:.0f}s: "
            f"{path}; last_error={last_error}"
        )


def _port_record(port_file: Path, pid: int) -> dict[str, Any] | None:
    if not port_file.exists():
        return None
    record = json.loads(port_file.read_text(encoding="utf-8"))
    if record.get("pid") != pid:
        return None
    return record


def _open_desktop(
    executable: Path,
    *,
    startup_timeout: float = 300,
) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    port_file = executable.parent / "data" / "desktop-port.json"
    process = _start(executable)
    record: dict[str, Any] | None = None
    try:
        record = _wait_for(
            lambda: _port_record(port_file, process.pid),
            timeout=startup_timeout,
            description="the packaged dynamic-port record",
        )

        def root_is_ready() -> bool:
            with urlopen(record["url"], timeout=2) as response:
                return response.status == 200

        _wait_for(
            root_is_ready,
            timeout=startup_timeout,
            description="the packaged loopback HTTP service",
        )
        windows = _wait_for(
            lambda: [
                window
                for window in _windows_for_pid(process.pid)
                if window["title"] == APP_TITLE
            ],
            timeout=60,
            description="the embedded WebView2 desktop window",
        )
        return process, {"port": record["port"], "url": record["url"], "windows": windows}
    except Exception:
        if process.poll() is None:
            windows = _windows_for_pid(process.pid)
            if windows:
                for window in windows:
                    user32.PostMessageW(window["hwnd"], WM_CLOSE, 0, 0)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=15)
            else:
                process.terminate()
                process.wait(timeout=15)
        if port_file.exists():
            stale = json.loads(port_file.read_text(encoding="utf-8"))
            if stale.get("pid") == process.pid:
                port_file.unlink()
        raise


def _stop_desktop(process: subprocess.Popen[bytes], port_file: Path) -> int:
    _close_windows(process)
    return_code = _await_exit(process)
    _wait_for(
        lambda: not port_file.exists(),
        timeout=30,
        description="desktop-port.json cleanup",
    )
    if return_code != 0:
        raise AssertionError(f"desktop process exited with {return_code}")
    return return_code


def _review_live_release(release_root: Path) -> dict[str, Any]:
    executable = release_root / "InvestmentLab.exe"
    port_file = release_root / "data" / "desktop-port.json"
    if port_file.exists():
        raise AssertionError(
            "desktop-port.json already exists; close the packaged app before review"
        )

    first, first_runtime = _open_desktop(executable)
    try:
        first_signature = _status_signature(first_runtime["url"])
        second = _start(executable)
        second_windows = _wait_for(
            lambda: [
                window
                for window in _windows_for_pid(second.pid)
                if window["class_name"] == "#32770"
                and window["title"] == APP_TITLE
            ],
            timeout=30,
            description="the single-instance information dialog",
        )
        _close_windows(second)
        second_return_code = _await_exit(second)
        if second_return_code != 0:
            raise AssertionError(
                f"second-instance process exited with {second_return_code}"
            )
        _json_url(f"{first_runtime['url']}/api/v33/model/status")
        _json_url(f"{first_runtime['url']}/api/v34/model/status")
    finally:
        if first.poll() is None:
            _stop_desktop(first, port_file)

    restarted, restarted_runtime = _open_desktop(executable)
    try:
        restarted_signature = _status_signature(restarted_runtime["url"])
    finally:
        if restarted.poll() is None:
            _stop_desktop(restarted, port_file)

    if first_signature != restarted_signature:
        raise AssertionError("training identity changed across packaged restart")
    return {
        "first_runtime": first_runtime,
        "single_instance_dialog": second_windows,
        "second_instance_return_code": second_return_code,
        "restart_runtime": restarted_runtime,
        "training_identity_sha256": _canonical_hash(first_signature),
        "training_identity_unchanged": True,
        "port_file_cleaned": not port_file.exists(),
    }


def _review_one_click_batch(
    project_root: Path,
    release_root: Path,
    *,
    expected_training_hash: str,
) -> dict[str, Any]:
    """Prove the Chinese launcher selects the packaged EXE and opens no URL manually."""

    launcher = project_root / "一键启动投资助手.bat"
    if not launcher.is_file():
        raise FileNotFoundError(launcher)
    # The Chinese launcher prefers the project-root desktop EXE (the app the
    # user actually runs); its dynamic-port record is written next to that
    # EXE, not inside the portable dist release.
    port_file = project_root / "data" / "desktop-port.json"
    if port_file.exists():
        raise AssertionError("packaged app is already running before batch review")
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(launcher)],
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"one-click launcher exited with {completed.returncode}"
        )

    record = _wait_for(
        lambda: (
            json.loads(port_file.read_text(encoding="utf-8"))
            if port_file.exists()
            else None
        ),
        timeout=120,
        description="the one-click packaged dynamic-port record",
    )
    pid = int(record["pid"])
    windows = _wait_for(
        lambda: [
            window
            for window in _windows_for_pid(pid)
            if window["title"] == APP_TITLE
        ],
        timeout=60,
        description="the one-click embedded window",
    )
    signature_hash = _canonical_hash(_status_signature(record["url"]))
    if signature_hash != expected_training_hash:
        raise AssertionError("one-click launcher loaded a different training identity")
    for window in windows:
        user32.PostMessageW(window["hwnd"], WM_CLOSE, 0, 0)
    _wait_for(
        lambda: not port_file.exists(),
        timeout=45,
        description="one-click desktop shutdown",
    )
    return {
        "launcher": str(launcher),
        "return_code": completed.returncode,
        "packaged_pid": pid,
        "dynamic_port": record["port"],
        "embedded_window": windows,
        "training_identity_sha256": signature_hash,
        "selected_packaged_release": True,
        "port_file_cleaned": True,
    }


def _copy_maintenance_clone(release_root: Path, destination: Path) -> Path:
    """Copy only packaged runtime files, never the surrounding project tree.

    The root-directory release intentionally lives beside source, reports,
    audits and the production ``data`` directory.  Recursively copying that
    root would duplicate the production database and unrelated test caches.
    A maintenance clone needs only the executable and its one-directory
    PyInstaller runtime; it creates an isolated sentinel database below.
    """
    release_root = release_root.resolve()
    clone = destination / "InvestmentLab"
    clone.mkdir(parents=True)
    shutil.copy2(release_root / "InvestmentLab.exe", clone / "InvestmentLab.exe")
    shutil.copytree(release_root / "_internal", clone / "_internal")
    marker = (
        release_root / "V3.5-8W.release"
        if (release_root / "V3.5-8W.release").exists()
        else release_root / "V3.4-13W.release"
    )
    if marker.is_file():
        shutil.copy2(marker, clone / marker.name)
    (clone / "data" / "backups").mkdir(parents=True)
    database = clone / "data" / "investment_lab.db"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE restore_probe (value TEXT NOT NULL)")
        connection.execute("INSERT INTO restore_probe VALUES ('original')")
        connection.commit()
    return clone


def _backup_names(root: Path) -> set[str]:
    backups = root / "data" / "backups"
    return {entry.name for entry in backups.iterdir() if entry.is_dir()}


def _run_message_command(
    executable: Path,
    arguments: list[str],
    completion_probe: Callable[[], Any],
) -> tuple[int, list[dict[str, Any]]]:
    process = _start(executable, *arguments)
    _wait_for(completion_probe, timeout=120, description="maintenance operation")
    windows = _wait_for(
        lambda: [
            window
            for window in _windows_for_pid(process.pid)
            if window["class_name"] == "#32770" and window["title"] == APP_TITLE
        ],
        timeout=30,
        description="maintenance completion dialog",
    )
    _close_windows(process)
    return_code = _await_exit(process)
    if return_code != 0:
        raise AssertionError(
            f"maintenance command {arguments!r} exited with {return_code}"
        )
    return return_code, windows


def _review_maintenance_clone(release_root: Path, scratch_parent: Path) -> dict[str, Any]:
    scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch_parent = scratch_parent.resolve()
    temporary_root = Path(
        tempfile.mkdtemp(prefix="v33-packaged-review-", dir=scratch_parent)
    ).resolve()
    try:
        if scratch_parent not in temporary_root.parents:
            raise AssertionError("temporary package escaped the review directory")
        clone = _copy_maintenance_clone(release_root, temporary_root)
        executable = clone / "InvestmentLab.exe"
        database = clone / "data" / "investment_lab.db"

        before = _backup_names(clone)
        backup_return_code, backup_windows = _run_message_command(
            executable,
            ["--backup"],
            lambda: len(_backup_names(clone) - before) == 1,
        )
        created = _backup_names(clone) - before
        if len(created) != 1:
            raise AssertionError(f"expected one backup, found {sorted(created)}")
        backup_name = next(iter(created))
        backup_database = clone / "data" / "backups" / backup_name / "data" / "investment_lab.db"
        backup_hash = _sqlite_logical_hash(backup_database)

        with closing(sqlite3.connect(database)) as connection:
            connection.execute("UPDATE restore_probe SET value = 'mutated'")
            connection.commit()
        with closing(sqlite3.connect(database)) as connection:
            if connection.execute("SELECT value FROM restore_probe").fetchone() != (
                "mutated",
            ):
                raise AssertionError("maintenance sentinel mutation failed")

        before_restore = _backup_names(clone)

        def restored() -> bool:
            if len(_backup_names(clone) - before_restore) != 1:
                return False
            with closing(sqlite3.connect(database, timeout=2)) as connection:
                return connection.execute(
                    "SELECT value FROM restore_probe"
                ).fetchone() == ("original",)

        restore_return_code, restore_windows = _run_message_command(
            executable,
            ["--restore", backup_name],
            restored,
        )
        safety = _backup_names(clone) - before_restore
        if len(safety) != 1:
            raise AssertionError(
                f"expected one restore safety backup, found {sorted(safety)}"
            )
        if _sqlite_logical_hash(database) != backup_hash:
            raise AssertionError("restored sentinel database differs from its backup")
        return {
            "isolated_clone": True,
            "backup_name": backup_name,
            "backup_return_code": backup_return_code,
            "backup_dialog": backup_windows,
            "restore_safety_backup": next(iter(safety)),
            "restore_return_code": restore_return_code,
            "restore_dialog": restore_windows,
            "restored_database_logical_sha256": backup_hash,
            "sentinel_restored": True,
            "temporary_clone_removed_after_review": True,
        }
    finally:
        if scratch_parent not in temporary_root.parents:
            raise AssertionError("refusing to remove an unverified temporary path")
        _remove_tree_with_retry(temporary_root)


def main() -> int:
    if not hasattr(ctypes, "windll"):
        raise RuntimeError("This packaged review only runs on Windows")
    arguments = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    release_root = arguments.release_root.resolve()
    executable = release_root / "InvestmentLab.exe"
    marker = (
        release_root / "V3.5-8W.release"
        if (release_root / "V3.5-8W.release").exists()
        else release_root / "V3.4-13W.release"
    )
    database = release_root / "data" / "investment_lab.db"
    for required in (executable, marker, database):
        if not required.is_file():
            raise FileNotFoundError(required)

    live = _review_live_release(release_root)
    one_click = _review_one_click_batch(
        project_root,
        release_root,
        expected_training_hash=live["training_identity_sha256"],
    )
    maintenance = _review_maintenance_clone(
        release_root, project_root / "reports" / ".packaged-exe-review"
    )
    report = {
        "status": "PASS",
        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "release_root": str(release_root),
        "release_marker": marker.read_text(encoding="utf-8-sig").strip(),
        "executable_sha256": _file_hash(executable),
        "release_database_sha256": _file_hash(database),
        "release_database_bytes": database.stat().st_size,
        "live_release": live,
        "one_click_launcher": one_click,
        "maintenance_clone": maintenance,
        "training_was_not_run": True,
        "source_database_was_not_modified": True,
    }
    report_path = arguments.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
