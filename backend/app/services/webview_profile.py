"""Bounded lifecycle management for per-process WebView2 profiles.

Only directories named ``webview2-<pid>`` directly below the supplied data
directory are considered.  Ambiguous ownership is deliberately fail-closed:
the profile is preserved instead of being removed.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Callable


PROFILE_PATTERN = re.compile(r"^webview2-(\d+)$")
OWNER_FILE = ".investment-lab-webview-owner.json"
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
ERROR_INVALID_PARAMETER = 87


@dataclass(frozen=True)
class ProcessProbe:
    status: str  # active, absent, unknown
    start_filetime: int | None = None


def probe_process(pid: int) -> ProcessProbe:
    """Return a Windows process identity without requiring psutil."""

    if pid <= 0 or os.name != "nt" or not hasattr(ctypes, "windll"):
        return ProcessProbe("unknown")
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        error = int(kernel32.GetLastError())
        return ProcessProbe("absent" if error == ERROR_INVALID_PARAMETER else "unknown")
    try:
        created = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return ProcessProbe("unknown")
        value = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
        return ProcessProbe("active", value)
    finally:
        kernel32.CloseHandle(handle)


def _safe_profiles(data_dir: Path) -> list[Path]:
    root = data_dir.resolve()
    profiles: list[Path] = []
    if not root.is_dir():
        return profiles
    for candidate in root.iterdir():
        if not candidate.is_dir() or not PROFILE_PATTERN.fullmatch(candidate.name):
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.parent != root or resolved == root:
            continue
        profiles.append(candidate)
    return profiles


def write_owner_marker(
    profile: Path,
    *,
    pid: int | None = None,
    launcher_path: Path | None = None,
) -> dict[str, object]:
    """Create the process-identity marker before WebView2 opens the profile."""

    pid = int(pid or os.getpid())
    match = PROFILE_PATTERN.fullmatch(profile.name)
    if match is None or int(match.group(1)) != pid:
        raise ValueError("profile name does not match owner pid")
    profile.mkdir(parents=True, exist_ok=True)
    probe = probe_process(pid)
    payload: dict[str, object] = {
        "schema_version": "webview-owner-v1",
        "pid": pid,
        "process_start_filetime": probe.start_filetime,
        "launcher": str((launcher_path or Path(sys.argv[0])).resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    marker = profile / OWNER_FILE
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temporary, marker)
    return payload


def _marker(profile: Path) -> dict[str, object] | None:
    try:
        value = json.loads((profile / OWNER_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def profile_state(
    profile: Path,
    *,
    process_probe: Callable[[int], ProcessProbe] = probe_process,
) -> str:
    """Classify a profile as active, stale, or unknown."""

    match = PROFILE_PATTERN.fullmatch(profile.name)
    if match is None:
        return "unknown"
    pid = int(match.group(1))
    probe = process_probe(pid)
    if probe.status == "unknown":
        return "unknown"
    if probe.status == "absent":
        return "stale"
    marker = _marker(profile)
    if marker is None:
        # Legacy profiles have no reliable process-start identity.  A live PID
        # could have been reused, so preserve the directory.
        return "unknown"
    try:
        marker_pid = int(marker["pid"])
        marker_start = int(marker["process_start_filetime"])
    except (KeyError, TypeError, ValueError):
        return "unknown"
    if marker_pid != pid:
        return "unknown"
    return "active" if probe.start_filetime == marker_start else "stale"


def _remove_with_retry(path: Path, delays: tuple[float, ...]) -> str | None:
    attempts = (0.0, *delays)
    error_text: str | None = None
    for delay in attempts:
        if delay:
            time.sleep(delay)
        try:
            shutil.rmtree(path)
            return None
        except FileNotFoundError:
            return None
        except OSError as error:
            error_text = str(error)
    return error_text or "unknown removal error"


def cleanup_profiles(
    data_dir: Path,
    *,
    keep_profile: Path | None = None,
    retain_newest_stale: bool = True,
    process_probe: Callable[[int], ProcessProbe] = probe_process,
    retry_delays: tuple[float, ...] = (0.2, 0.5, 1.0),
) -> dict[str, object]:
    """Remove confirmed stale profiles while retaining a bounded fallback."""

    root = data_dir.resolve()
    keep_resolved = keep_profile.resolve() if keep_profile is not None else None
    states: dict[Path, str] = {}
    for profile in _safe_profiles(root):
        states[profile] = profile_state(profile, process_probe=process_probe)

    stale = [path for path, state in states.items() if state == "stale"]
    stale.sort(key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    retained_stale: Path | None = stale[0] if retain_newest_stale and stale else None
    removed: list[str] = []
    failures: dict[str, str] = {}
    for profile in stale:
        resolved = profile.resolve()
        if resolved == keep_resolved or profile == retained_stale:
            continue
        error = _remove_with_retry(profile, retry_delays)
        if error is None:
            removed.append(profile.name)
        else:
            failures[profile.name] = error
    return {
        "removed": removed,
        "retained_stale": retained_stale.name if retained_stale else None,
        "active": sorted(path.name for path, state in states.items() if state == "active"),
        "unknown": sorted(path.name for path, state in states.items() if state == "unknown"),
        "failures": failures,
    }
