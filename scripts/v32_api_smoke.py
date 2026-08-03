"""Black-box V3.2 API smoke test used by local and desktop verification."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE = os.environ.get("INVESTMENT_LAB_BASE_URL", "http://127.0.0.1:8765").rstrip("/")


def request(path: str, payload: dict | None = None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    method = "GET" if body is None else "POST"
    try:
        with urlopen(Request(BASE + path, data=body, headers=headers, method=method), timeout=60) as response:
            return json.loads(response.read())
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise AssertionError(f"{method} {path} returned {error.code}: {detail}") from error


before = request("/api/v3.2/training/status")
before_counts = {market: before[market]["iteration_count"] for market in ("399006", "NDX")}
assert before_counts["399006"] >= 500
assert before_counts["NDX"] >= 500

for market in ("399006", "NDX"):
    result = request(f"/api/v3.2/analysis/{market}/run", {"refresh": False})
    assert result["training_mutated"] is False
    assert result["model"]["weekly"]["iteration_number"] == before_counts[market]
    assert len(result["path"]["points"]) == 13
    assert result["advice"]["final_target_position"] % 5 == 0
    assert result["advice"]["fund_etf_ratio"] in {"7:3", "6:4", "5:5", "4:6", "3:7"}
    assert all(batch["position_points"] % 5 == 0 for batch in result["advice"]["batches"])

after = request("/api/v3.2/training/status")
after_counts = {market: after[market]["iteration_count"] for market in ("399006", "NDX")}
assert after_counts == before_counts
print(json.dumps({"before": before_counts, "after": after_counts, "status": "ok"}, ensure_ascii=False))
