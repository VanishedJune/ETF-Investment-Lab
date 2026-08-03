"""Deterministic Playwright visual review for the position calendar."""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import Route, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BASE_URL = "http://127.0.0.1:8765"


EVENTS = [
    {
        "id": 901,
        "instrument_code": "399006",
        "direction": "increase",
        "operation_date": "2026-08-01",
        "change_percent": 15,
        "position_after": 15,
        "sequence": 1,
        "note": "月初分批增仓",
    },
    {
        "id": 902,
        "instrument_code": "399006",
        "direction": "decrease",
        "operation_date": "2026-08-01",
        "change_percent": 5,
        "position_after": 10,
        "sequence": 2,
        "note": "盘中风险控制",
    },
    {
        "id": 903,
        "instrument_code": "399006",
        "direction": "increase",
        "operation_date": "2026-07-17",
        "change_percent": 10,
        "position_after": 10,
        "sequence": 1,
        "note": "历史仓位记录",
    },
]


def mock_positions(route: Route) -> None:
    route.fulfill(
        status=200,
        content_type="application/json",
        body=json.dumps({"399006": 10, "159941": 0, "NDX": 80}),
    )


def mock_events(route: Route) -> None:
    route.fulfill(status=200, content_type="application/json", body=json.dumps(EVENTS))


def review_page(page, viewport_name: str) -> dict[str, object]:
    console_errors: list[str] = []
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
    page.route("**/api/investment-calendar/current-positions", mock_positions)
    page.route("**/api/v2/position-events?*", mock_events)
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    calendar = page.locator(".position-calendar")
    calendar.scroll_into_view_if_needed()
    calendar.screenshot(path=str(REPORTS / f"position-calendar-{viewport_name}.png"))
    measurements = page.evaluate(
        """
        () => ({
          dayCount: document.querySelectorAll('[data-calendar-date]').length,
          selectedCount: document.querySelectorAll('[aria-selected="true"]').length,
          increaseMarkers: document.querySelectorAll('.increase-marker').length,
          decreaseMarkers: document.querySelectorAll('.decrease-marker').length,
          bodyOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
          calendarOverflow: (() => {
            const value = document.querySelector('.position-calendar');
            return value ? value.scrollWidth > value.clientWidth : true;
          })(),
        })
        """
    )
    measurements["consoleErrors"] = console_errors
    return measurements


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        desktop = browser.new_page(viewport={"width": 1440, "height": 1200}, device_scale_factor=1)
        desktop_result = review_page(desktop, "desktop")
        desktop.close()
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        mobile_result = review_page(mobile, "mobile")
        mobile.close()
        browser.close()

    result = {"desktop": desktop_result, "mobile": mobile_result}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for name, measurements in result.items():
        assert measurements["dayCount"] == 42, f"{name}: calendar must contain 42 days"
        assert measurements["selectedCount"] == 1, f"{name}: one date must be selected"
        assert not measurements["bodyOverflow"], f"{name}: page has horizontal overflow"
        assert not measurements["calendarOverflow"], f"{name}: calendar has horizontal overflow"
        assert not measurements["consoleErrors"], f"{name}: console errors were emitted"


if __name__ == "__main__":
    main()
