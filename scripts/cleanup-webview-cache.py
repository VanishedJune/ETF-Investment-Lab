"""Explicit maintenance command for confirmed stale InvestmentLab WebView2 profiles."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "backend" / "app" / "services" / "webview_profile.py"
SPEC = importlib.util.spec_from_file_location("investment_lab_webview_profile", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load WebView profile manager: {MODULE_PATH}")
profile_manager = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile_manager
SPEC.loader.exec_module(profile_manager)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data_dir = ROOT / "data"
    port_file = data_dir / "desktop-port.json"
    if port_file.is_file():
        try:
            pid = int(json.loads(port_file.read_text(encoding="utf-8"))["pid"])
            state = profile_manager.probe_process(pid)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            state = profile_manager.ProcessProbe("unknown")
        if state.status != "absent":
            print(json.dumps({"status": "blocked", "reason": "InvestmentLab may be running"}, ensure_ascii=False))
            return 2
    profiles = profile_manager._safe_profiles(data_dir)
    states = {
        path.name: profile_manager.profile_state(path)
        for path in profiles
    }
    if args.check:
        print(json.dumps({"status": "check", "profiles": states}, ensure_ascii=False, indent=2))
        return 0
    report = profile_manager.cleanup_profiles(data_dir, retain_newest_stale=True)
    report["status"] = "success" if not report["failures"] else "completed_with_warnings"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
