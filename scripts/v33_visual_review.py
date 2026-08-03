"""Native-Python Playwright review for the V3.3 single research workbench.

The page is served by the local Vite development server while every API call
is fulfilled in-browser with deterministic release-contract data.  The review
therefore exercises the real Vue/ECharts bundle without reading or writing the
production SQLite database.
"""

from __future__ import annotations

from datetime import date, timedelta
import json
import math
import os
from pathlib import Path
from typing import Any

from playwright.sync_api import Route, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
BASE_URL = os.environ.get("V33_BASE_URL", "http://127.0.0.1:4173")


def _json(route: Route, payload: Any, *, status: int = 200) -> None:
    route.fulfill(
        status=status,
        content_type="application/json; charset=utf-8",
        body=json.dumps(payload, ensure_ascii=False),
    )


def _market_rows(symbol: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Keep the visual quote strip aligned with the analysis fixture's latest
    # completed trading week (126 Friday observations ending 2026-07-31).
    start = date(2024, 3, 8)
    base = 1_850.0 if symbol == "399006" else 0.82
    rows: list[dict[str, Any]] = []
    indicators: list[dict[str, Any]] = []
    for index in range(126):
        day = start + timedelta(days=index * 7)
        trend = 1.0 + index * 0.0032
        wave = 1.0 + 0.055 * math.sin(index / 6.5)
        close = base * trend * wave
        open_price = close * (1.0 - 0.009 * math.sin(index / 2.8))
        high = max(open_price, close) * 1.018
        low = min(open_price, close) * 0.982
        volume = 8.0e8 * (1.0 + 0.30 * math.sin(index / 8.0))
        stamp = day.isoformat()
        rows.append(
            {
                "date": stamp,
                "open": f"{open_price:.6f}",
                "high": f"{high:.6f}",
                "low": f"{low:.6f}",
                "close": f"{close:.6f}",
                "volume": f"{volume:.2f}",
                "source": "NATIVE_PLAYWRIGHT_AGGREGATED_DAILY_VOLUME",
            }
        )
        dif = 0.018 * math.sin(index / 5.5)
        dea = 0.015 * math.sin((index - 2) / 5.5)
        indicators.append(
            {
                "date": stamp,
                "dif": f"{dif:.8f}",
                "dea": f"{dea:.8f}",
                "macd_histogram": f"{2 * (dif - dea):.8f}",
                "rsi_6": f"{50 + 22 * math.sin(index / 7.0):.4f}",
            }
        )
    return {"rows": rows}, indicators


def _progressive(market: str) -> dict[str, Any]:
    total = 511 if market == "399006" else 512
    curve = []
    for index in range(72):
        pending = index >= 52
        curve.append(
            {
                "iteration": index + 1,
                "cutoff_date": (date(2025, 3, 7) + timedelta(days=7 * index)).isoformat(),
                "status": "pending" if pending else "full",
                "loss": None if pending else 1.35 - index * 0.011,
                "price_turn_error_days": None if pending else 10.5 - index * 0.09,
                "dif_turn_error_days": None if pending else 8.2 - index * 0.06,
                "promoted": False,
                "champion_version": f"{'CYB' if market == '399006' else 'GFNDXETF'}_HYBRID_20W_V3.3.{index + 1}",
            }
        )
    return {
        "implementation_revision": "V3.3-20W",
        "market": market,
        "state": "completed",
        "iterations": [],
        "pending_count": 20,
        "full_count": 52,
        "curve": curve,
        "expected_iteration_count": total,
        "audited_iteration_count": total,
        "weekly_sampling_seed": 330020,
        "sampled_dates": [],
    }


STATUS = {
    "implementation_revision": "V3.3-20W",
    "markets": {
        "399006": {
            "instrument_code": "399006",
            "is_training": False,
            "bootstrapped": True,
            "iteration_count": 511,
            "champion_version": "CYB_HYBRID_20W_V3.3.511",
            "last_training_week_key": "2026-W31",
            "last_successful_training_at": "2026-08-01T04:00:00Z",
            "pending_count": 20,
            "message": "训练空闲",
        },
        "159941": {
            "instrument_code": "159941",
            "is_training": False,
            "bootstrapped": True,
            "iteration_count": 512,
            "champion_version": "GFNDXETF_HYBRID_20W_V3.3.512",
            "last_training_week_key": "2026-W31",
            "last_successful_training_at": "2026-08-01T04:05:00Z",
            "pending_count": 20,
            "message": "训练空闲",
        },
    },
}


ANALYSIS = {
    "implementation_revision": "V3.3-20W",
    "run_id": "V33-NATIVE-VISUAL-399006",
    "market": "399006",
    "benchmark": "399006",
    "training_mutated": False,
    "model": {
        "version": "CYB_HYBRID_20W_V3.3.511",
        "iteration_number": 511,
        "state_hash": "native-visual-state-hash",
        "trained_through": "2026-07-31",
        "training_sample_count": 491,
        "horizon_weeks": 20,
        "daily_window_sessions": 100,
    },
    "path": {
        "weeks": list(range(1, 21)),
        "p10": [-0.04 + index * 0.001 for index in range(20)],
        "p50": [index * 0.004 for index in range(20)],
        "p90": [0.04 + index * 0.007 for index in range(20)],
        "expected": [index * 0.0045 for index in range(20)],
        "weekly_base": [index * 0.004 for index in range(20)],
        "daily_correction": [0.0, 0.000375, 0.0005, 0.000375] + [0.0] * 16,
        "direction": "up",
        "weekly_confidence": 74,
        "daily_confidence_adjustment": 4,
        "confidence": 78,
        "up_probability": 72,
        "sideways_probability": 18,
        "down_probability": 10,
        "expected_max_drawdown": -8,
        "predicted_high_week": 18,
        "predicted_low_week": 3,
        "analogue_role": "residual_interval_calibration_only",
        "analogue_calibration_count": 80,
    },
    "turning_points": {
        "stable": True,
        "kind": "local_minimum",
        "dif_derivative_zero": {
            "start": "2026-08-10",
            "center": "2026-08-12",
            "end": "2026-08-14",
        },
        "price_turn": {
            "start": "2026-08-13",
            "center": "2026-08-17",
            "end": "2026-08-19",
        },
        "dif_axis_zero_is_distinct": True,
        "date_basis": "actual exchange sessions",
    },
    "position": {
        "current": 20,
        "weekly_base_target": 75,
        "daily_adjustment": 5,
        "target": 80,
        "change": 60,
        "source": "investment_calendar",
    },
    "advice": {
        "action": "buy",
        "summary": "分4批增仓60个百分点",
        "batches": [
            {
                "batch": index + 1,
                "action": "buy",
                "percentage_points": percentage,
                "window_start": "2026-08-10",
                "expected_date": f"2026-08-{12 + index * 5:02d}",
                "window_end": "2026-09-02",
                "condition": "DIF斜率向上且成交量确认",
            }
            for index, percentage in enumerate((20, 15, 15, 10))
        ],
        "fund_etf_ratio": "7:3",
        "policy": ["20-week probability path", "drawdown penalty"],
        "conditional": True,
        "position_grid": 5,
    },
    "features": {"weekly": {}, "daily": {}, "daily_sequence_count": 100},
    "data_quality": {
        "source_data_max_date": "2026-07-31",
        "cutoff_date": "2026-07-31",
        "price_data_as_of": "2026-07-31",
        "analysis_as_of": "2026-08-01T04:12:30+00:00",
        "missing_masks": {},
        "missing_series": [],
        "degraded": False,
        "provenance": {
            "price": "direct-index",
            "valuation": "point-in-time",
            "price_as_of": "2026-07-31",
            "analysis_as_of": "2026-08-01T04:12:30+00:00",
        },
    },
    "analysis_hash": "0123456789abcdef0123456789abcdef",
}


def install_routes(page, counters: dict[str, int]) -> None:
    price_payloads = {
        market: _market_rows(market) for market in ("399006", "159941")
    }

    def market_prices(route: Route) -> None:
        market = "159941" if "/159941/" in route.request.url else "399006"
        _json(route, price_payloads[market][0])

    def market_indicators(route: Route) -> None:
        market = "159941" if "/159941" in route.request.url else "399006"
        _json(route, price_payloads[market][1])

    def iterations(route: Route) -> None:
        market = "159941" if route.request.url.endswith("/159941") else "399006"
        _json(route, _progressive(market))

    def analysis(route: Route) -> None:
        counters["analysis"] += 1
        _json(route, ANALYSIS)

    def training(route: Route) -> None:
        counters["training"] += 1
        _json(route, _progressive("399006"))

    page.route("**/api/market/*/prices?*", market_prices)
    page.route("**/api/indicators/*?*", market_indicators)
    page.route("**/api/v33/model/status", lambda route: _json(route, STATUS))
    page.route("**/api/v33/iterations/*", iterations)
    page.route(
        "**/api/v33/forecast/*",
        lambda route: _json(route, {"detail": "no live forecast"}, status=404),
    )
    page.route("**/api/v33/model/analysis", analysis)
    page.route("**/api/v33/model/train", training)
    page.route(
        "**/api/investment-calendar/current-positions",
        lambda route: _json(route, {"399006": 20, "159941": 0, "NDX": 95}),
    )
    page.route("**/api/v2/position-events?*", lambda route: _json(route, []))
    for pattern in (
        "**/api/v3.2/**",
        "**/api/v3.1/**",
        "**/api/v2/models/**",
        "**/api/v2/advice/**",
    ):
        page.route(pattern, lambda route: _json(route, {"detail": "legacy audit unavailable"}, status=404))


def review_page(page, *, mobile: bool) -> dict[str, Any]:
    counters = {"analysis": 0, "training": 0}
    console_errors: list[str] = []
    http_failures: list[tuple[int, str]] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.on(
        "response",
        lambda response: http_failures.append((response.status, response.url))
        if response.status >= 400
        else None,
    )
    install_routes(page, counters)
    page.goto(BASE_URL, wait_until="networkidle")
    heading = page.get_by_role("heading", name="双市场周线研究台")
    try:
        heading.wait_for(state="visible", timeout=30_000)
    except Exception as exc:
        diagnostic = REPORTS / (
            "v33-native-visual-mobile-error.png"
            if mobile
            else "v33-native-visual-desktop-error.png"
        )
        page.screenshot(path=str(diagnostic), full_page=True)
        body_text = page.locator("body").inner_text(timeout=5_000)
        raise AssertionError(
            "V3.3 workbench heading did not render; "
            f"url={page.url!r}, title={page.title()!r}, "
            f"body={body_text[:2000]!r}, console_errors={console_errors!r}, "
            f"screenshot={diagnostic}"
        ) from exc
    assert page.get_by_test_id("v33-train").inner_text() == "训练模型"
    assert page.get_by_test_id("v33-analyze").inner_text() == "数据分析"
    page.wait_for_selector('[data-testid="v33-curve-399006"] canvas', timeout=30_000)
    page.wait_for_selector('[data-testid="v33-curve-159941"] canvas', timeout=30_000)
    page.wait_for_selector('[data-testid="unified-market-chart"] canvas', timeout=30_000)

    page.get_by_test_id("v33-analyze").click()
    page.get_by_test_id("v33-analysis-result").wait_for(state="visible", timeout=30_000)
    result_text = page.get_by_test_id("v33-analysis-result").inner_text()
    for expected in (
        "分4批增仓60个百分点",
        "72.0%",
        "78.0%",
        "-8.00%",
        "7:3",
    ):
        assert expected in result_text, f"missing analysis text: {expected}"
    assert "7200.0%" not in result_text
    assert "-800.00%" not in result_text
    assert counters == {"analysis": 1, "training": 0}
    assert page.get_by_test_id("v33-batches").locator("li").count() == 4
    page.wait_for_selector('[data-testid="v33-path-chart"] canvas', timeout=30_000)
    decomposition = page.get_by_test_id("v33-model-decomposition").inner_text()
    assert "20周完整" in decomposition and "5–20周为0" in decomposition
    assert "周K基础目标仓位" in decomposition and "75%" in decomposition
    assert "日K仓位修正" in decomposition and "+5个百分点" in decomposition
    assert "2026-07-31" in page.get_by_test_id("v33-price-data-as-of").inner_text()
    assert "2026-08-01T04:12:30+00:00" in page.get_by_test_id("v33-analysis-as-of").inner_text()

    measurements = page.evaluate(
        """
        () => ({
          bodyOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
          bodyFont: parseFloat(getComputedStyle(document.body).fontSize),
          mainHeadingFont: parseFloat(getComputedStyle(document.querySelector('h1')).fontSize),
        })
        """
    )
    assert not measurements["bodyOverflow"], "page has horizontal overflow"
    assert measurements["bodyFont"] >= 14, "body font is too small"
    assert measurements["mainHeadingFont"] >= (30 if mobile else 40), "main heading is too small"
    target = REPORTS / ("v33-native-visual-mobile.png" if mobile else "v33-native-visual-desktop.png")
    page.screenshot(path=str(target), full_page=True)
    allowed_optional_prefixes = (
        f"{BASE_URL}/api/v33/forecast/",
        f"{BASE_URL}/api/v3.2/",
        f"{BASE_URL}/api/v3.1/",
        f"{BASE_URL}/api/v2/models/",
        f"{BASE_URL}/api/v2/advice/",
    )
    unexpected_http = [
        item
        for item in http_failures
        if item[0] != 404 or not item[1].startswith(allowed_optional_prefixes)
    ]
    assert not unexpected_http, unexpected_http
    unexpected_console = [
        item
        for item in console_errors
        if item != "Failed to load resource: the server responded with a status of 404 (Not Found)"
    ]
    assert not unexpected_console, unexpected_console
    return {**measurements, **counters, "screenshot": str(target)}


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        desktop = browser.new_page(
            viewport={"width": 1500, "height": 1000}, device_scale_factor=1
        )
        desktop_result = review_page(desktop, mobile=False)
        desktop.close()
        mobile = browser.new_page(
            viewport={"width": 412, "height": 915}, device_scale_factor=1
        )
        mobile_result = review_page(mobile, mobile=True)
        mobile.close()
        browser.close()
    print(
        json.dumps(
            {"status": "PASS", "desktop": desktop_result, "mobile": mobile_result},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
