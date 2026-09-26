"""Runtime self-check for AkShare bundled data files."""

from __future__ import annotations

import json
from pathlib import Path


AKSHARE_RESOURCE_OK = "AKSHARE_RESOURCE_OK"
AKSHARE_RESOURCE_MISSING = "AKSHARE_RESOURCE_MISSING"
AKSHARE_RESOURCE_INVALID = "AKSHARE_RESOURCE_INVALID"


def akshare_resource_status() -> str:
    """Return OK/MISSING/INVALID for the AkShare trading-calendar file."""

    try:
        import akshare

        root = Path(akshare.__file__).resolve().parent
        calendar = root / "file_fold" / "calendar.json"
        if not calendar.is_file():
            return AKSHARE_RESOURCE_MISSING
        data = json.loads(calendar.read_text(encoding="utf-8"))
        if not isinstance(data, list) or len(data) == 0:
            return AKSHARE_RESOURCE_INVALID
        return AKSHARE_RESOURCE_OK
    except Exception:  # noqa: BLE001 - status must never raise
        return AKSHARE_RESOURCE_INVALID
