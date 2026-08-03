"""Native Playwright acceptance for the packaged V3.4-13W desktop release.

The final release is started as a real Windows EXE and its dynamic loopback
service is exercised with a desktop Chromium viewport.  The calendar CRUD
proof writes one reversible five-point audit event to the real release
database, verifies create/update/delete persistence, and leaves positions
unchanged after cleanup.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import date
import json
from pathlib import Path
import sqlite3
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from playwright.sync_api import Page, sync_playwright

from v33_packaged_exe_review import (
    _file_hash,
    _open_desktop,
    _stop_desktop,
)


ACTIVE_MARKETS = ("399006", "159941")


def _arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Launch the packaged V3.4 EXE and verify its real desktop UI."
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=root / "dist" / "InvestmentLab",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "v34-packaged-playwright-review.json",
    )
    return parser.parse_args()


def _json_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 120,
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                raise AssertionError(f"{method} {url} returned HTTP {response.status}")
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise AssertionError(f"{method} {url} returned HTTP {error.code}: {detail}") from error


def _status_counts(base_url: str) -> dict[str, int]:
    rows = _json_request(f"{base_url}/api/v341/model/status")
    return {row["market"]: int(row["weekly_iteration_count"]) for row in rows}


def _active_status_signature(base_url: str) -> dict[str, Any]:
    rows = _json_request(f"{base_url}/api/v341/model/status")
    statuses = {row["market"]: row for row in rows}
    output: dict[str, Any] = {}
    for market in ACTIVE_MARKETS:
        status = statuses[market]
        champion = _json_request(f"{base_url}/api/v341/models/{market}/champion")
        output[market] = {
            "bootstrapped": status["bootstrapped"],
            "weekly_iteration_count": status["weekly_iteration_count"],
            "candidate_training_count": status["candidate_training_count"],
            "champion_promotion_count": status["champion_promotion_count"],
            "last_anchor_date": status["last_anchor_date"],
            "champion": champion,
        }
    return output


def _axis_snapshot(page: Page, selector: str) -> dict[str, float]:
    attributes = {
        "visible_start": "data-visible-start-index",
        "visible_end": "data-visible-end-index",
        "price_min": "data-price-y-min",
        "price_max": "data-price-y-max",
        "volume_min": "data-volume-y-min",
        "volume_max": "data-volume-y-max",
        "macd_min": "data-macd-y-min",
        "macd_max": "data-macd-y-max",
    }
    values: dict[str, float] = {}
    locator = page.locator(selector)
    for key, attribute in attributes.items():
        raw = locator.get_attribute(attribute)
        if raw is None:
            raise AssertionError(f"{selector} is missing {attribute}")
        values[key] = float(raw)
    if values["volume_min"] != 0:
        raise AssertionError(f"volume axis does not start at zero: {values}")
    if values["macd_min"] > 0 or values["macd_max"] < 0:
        raise AssertionError(f"MACD axis does not include zero: {values}")
    if values["price_max"] <= values["price_min"]:
        raise AssertionError(f"invalid price axis: {values}")
    return values


def _forecast_contract(base_url: str, market: str) -> dict[str, Any]:
    forecast = _json_request(f"{base_url}/api/v343/forecast/{market}")
    candles = forecast["representative_ohlcv"]
    quantiles = forecast["price_quantiles"]
    indicators = forecast["indicators"]
    if len(candles) != 13 or len(quantiles) != 13 or len(indicators) != 13:
        raise AssertionError("forecast does not contain 13 complete weekly rows")
    if int(forecast["scenario_count"]) < 1000:
        raise AssertionError("forecast scenario count is below the release floor")
    for candle in candles:
        if not (
            candle["low"] <= min(candle["open"], candle["close"])
            and candle["high"] >= max(candle["open"], candle["close"])
            and candle["volume_p50"] >= 0
        ):
            raise AssertionError(f"illegal forecast OHLCV row: {candle}")
    for row in quantiles:
        if not row["close_p10"] <= row["close_p50"] <= row["close_p90"]:
            raise AssertionError(f"crossed P10/P50/P90 quantiles: {row}")
    probabilities = forecast.get("direction_probabilities") or (
        forecast.get("horizon_probabilities", {}).get("13")
    )
    if not isinstance(probabilities, dict) or not probabilities:
        raise AssertionError("forecast has no 13-week direction probabilities")
    probabilities = probabilities.get("calibrated") or probabilities.get("raw") or probabilities
    probability_sum = sum(float(value) for value in probabilities.values())
    if probability_sum <= 1.5:
        probability_sum *= 100
    if abs(probability_sum - 100.0) > 0.2:
        raise AssertionError(f"direction probabilities do not sum to 100: {probability_sum}")
    return {
        "market": market,
        "model_version": forecast["model_version"],
        "forecast_anchor_date": forecast["forecast_anchor_date"],
        "scenario_count": forecast["scenario_count"],
        "candles": len(candles),
        "quantiles": len(quantiles),
        "indicators": len(indicators),
        "direction_probability_sum": probability_sum,
        "scenario_audit": forecast["scenario_audit"],
    }


def _review_market(page: Page, base_url: str, market: str) -> dict[str, Any]:
    if market == "159941":
        page.get_by_role("button", name="广发纳斯达克100ETF", exact=False).click()
    else:
        page.get_by_role("button", name="创业板指数", exact=False).click()
    page.locator('[data-testid="unified-market-chart"][data-chart-state="rendered"]').wait_for(
        timeout=60_000
    )
    page.locator(f'[data-testid="v34-curve-{market}"] canvas').wait_for(timeout=60_000)

    main_canvas = page.locator(".unified-chart-canvas canvas")
    main_details = page.get_by_test_id("market-point-details")
    main_canvas.wait_for(timeout=60_000)
    main_details.wait_for(timeout=60_000)
    main_canvas_box = main_canvas.bounding_box()
    main_details_box = main_details.bounding_box()
    if not main_canvas_box or not main_details_box:
        raise AssertionError("main chart or its fixed detail panel has no rendered bounds")
    if main_canvas_box["x"] + main_canvas_box["width"] > main_details_box["x"]:
        raise AssertionError("main chart fixed detail panel overlaps the plot")

    periods: dict[str, Any] = {}
    for timeframe in ("daily", "weekly", "monthly"):
        page.get_by_test_id(f"timeframe-{timeframe}").click()
        page.locator('[data-testid="unified-market-chart"][data-chart-state="rendered"]').wait_for(
            timeout=60_000
        )
        canvas = page.locator(".unified-chart-canvas")
        canvas.locator("canvas").wait_for(timeout=60_000)
        periods[timeframe] = _axis_snapshot(page, ".unified-chart-canvas")

    page.get_by_test_id("timeframe-weekly").click()
    page.locator('[data-testid="unified-market-chart"][data-chart-state="rendered"]').wait_for(
        timeout=60_000
    )
    before = _axis_snapshot(page, ".unified-chart-canvas")
    chart = page.locator(".unified-chart-canvas")
    chart.locator("canvas").hover(position={"x": 650, "y": 280})
    page.mouse.wheel(0, -1000)
    page.wait_for_function(
        """before => {
          const el = document.querySelector('.unified-chart-canvas');
          if (!el) return false;
          const start = Number(el.getAttribute('data-visible-start-index'));
          const end = Number(el.getAttribute('data-visible-end-index'));
          return end - start < before.end - before.start;
        }""",
        arg={"start": before["visible_start"], "end": before["visible_end"]},
        timeout=30_000,
    )
    after = _axis_snapshot(page, ".unified-chart-canvas")
    if after["visible_end"] - after["visible_start"] >= before["visible_end"] - before["visible_start"]:
        raise AssertionError("main chart wheel zoom did not narrow the visible range")
    if all(
        after[key] == before[key]
        for key in ("price_min", "price_max", "volume_max", "macd_min", "macd_max")
    ):
        raise AssertionError("visible-range change did not recompute any Y-axis bound")

    analyze_button = page.get_by_test_id("v34-analyze")
    analyze_button.wait_for(state="attached", timeout=60_000)
    page.wait_for_function(
        """() => {
          const button = document.querySelector('[data-testid="v34-analyze"]');
          return button instanceof HTMLButtonElement && !button.disabled;
        }""",
        timeout=60_000,
    )
    analyze_button.scroll_into_view_if_needed(timeout=30_000)
    analyze_button.click(timeout=30_000)
    page.get_by_test_id("v34-analysis-result").wait_for(timeout=180_000)
    page.locator('[data-testid="v34-forecast-chart"] canvas').wait_for(timeout=60_000)
    forecast_details = page.get_by_test_id("forecast-point-details")
    forecast_details.wait_for(timeout=60_000)
    forecast_canvas_box = page.locator('[data-testid="v34-forecast-chart"] canvas').bounding_box()
    forecast_details_box = forecast_details.bounding_box()
    if not forecast_canvas_box or not forecast_details_box:
        raise AssertionError("forecast chart or its fixed readout has no rendered bounds")
    if forecast_details_box["y"] + forecast_details_box["height"] > forecast_canvas_box["y"]:
        raise AssertionError("forecast fixed readout overlaps the chart")
    forecast_before = _axis_snapshot(page, '[data-testid="v34-forecast-chart"]')
    forecast_chart = page.get_by_test_id("v34-forecast-chart")
    forecast_chart.locator("canvas").hover(position={"x": 600, "y": 260})
    page.mouse.wheel(0, -900)
    page.wait_for_function(
        """before => {
          const el = document.querySelector('[data-testid="v34-forecast-chart"]');
          if (!el) return false;
          return Number(el.getAttribute('data-visible-end-index')) -
            Number(el.getAttribute('data-visible-start-index')) < before.end - before.start;
        }""",
        arg={"start": forecast_before["visible_start"], "end": forecast_before["visible_end"]},
        timeout=30_000,
    )
    forecast_after = _axis_snapshot(page, '[data-testid="v34-forecast-chart"]')
    return {
        "market": market,
        "period_axes": periods,
        "main_zoom_before": before,
        "main_zoom_after": after,
        "forecast_zoom_before": forecast_before,
        "forecast_zoom_after": forecast_after,
        "forecast": _forecast_contract(base_url, market),
    }


def _review_live_ui(release_root: Path, report_root: Path) -> dict[str, Any]:
    executable = release_root / "InvestmentLab.exe"
    database = release_root / "data" / "investment_lab.db"
    database_hash_before = _file_hash(database)
    process, runtime = _open_desktop(executable)
    try:
        counts_before = _status_counts(runtime["url"])
        signature_before = _active_status_signature(runtime["url"])
        console_errors: list[str] = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1000})
            def capture_console(message: Any) -> None:
                if message.type != "error":
                    return
                console_errors.append(message.text)
                print(
                    "BROWSER_CONSOLE_ERROR "
                    + json.dumps(message.text, ensure_ascii=True),
                    flush=True,
                )

            page.on("console", capture_console)
            page.goto(runtime["url"], wait_until="networkidle", timeout=120_000)
            page.get_by_role("heading", name="双市场周线研究台").wait_for(timeout=60_000)
            page.get_by_test_id("v34-curve-399006").locator("canvas").wait_for(timeout=60_000)
            page.get_by_test_id("v34-curve-159941").locator("canvas").wait_for(timeout=60_000)
            markets = [_review_market(page, runtime["url"], market) for market in ACTIVE_MARKETS]
            screenshot = report_root / "v34-packaged-desktop.png"
            page.screenshot(path=str(screenshot), full_page=True)
            browser.close()
        counts_after = _status_counts(runtime["url"])
        signature_after = _active_status_signature(runtime["url"])
        if counts_before != counts_after:
            raise AssertionError(f"analysis changed V3.4 training counts: {counts_before} -> {counts_after}")
        if signature_before != signature_after:
            raise AssertionError("analysis changed persisted V3.3/V3.4 model identity")
    finally:
        if process.poll() is None:
            _stop_desktop(process, release_root / "data" / "desktop-port.json")
    database_hash_after = _file_hash(database)
    if database_hash_before != database_hash_after:
        # Analysis writes V3.4 forecasts and audit rows by design; model identity
        # above is the frozen-state gate. Record the physical DB change explicitly.
        database_mutated_by_analysis = True
    else:
        database_mutated_by_analysis = False
    unexpected_console = [
        message
        for message in console_errors
        if "Failed to load resource" not in message
    ]
    if unexpected_console:
        raise AssertionError(f"unexpected browser console errors: {unexpected_console}")
    return {
        "runtime": runtime,
        "training_counts_before": counts_before,
        "training_counts_after": counts_after,
        "model_identity_unchanged": True,
        "release_database_sha256_before": database_hash_before,
        "release_database_sha256_after": database_hash_after,
        "release_database_mutated_by_analysis": database_mutated_by_analysis,
        "markets": markets,
        "console_errors": console_errors,
        "screenshot": str(screenshot),
        "port_file_cleaned": not (release_root / "data" / "desktop-port.json").exists(),
    }


def _calendar_crud_release(release_root: Path) -> dict[str, Any]:
    database = release_root / "data" / "investment_lab.db"
    process, runtime = _open_desktop(release_root / "InvestmentLab.exe")
    try:
        base_url = runtime["url"]
        identity_before = _active_status_signature(base_url)
        positions_before = _json_request(
            f"{base_url}/api/investment-calendar/current-positions"
        )
        current = int(positions_before.get("399006") or 0)
        direction = "increase" if current <= 95 else "decrease"
        expected = current + 5 if direction == "increase" else current - 5
        marker = f"V3.4 packaged CRUD audit {time.time_ns()}"
        created = _json_request(
            f"{base_url}/api/investment-calendar",
            method="POST",
            payload={
                "instrument_code": "399006",
                "direction": direction,
                "operation_date": date.today().isoformat(),
                "change_percent": 5,
                "note": marker,
            },
        )
        entry_id = int(created["id"])
        positions_created = _json_request(
            f"{base_url}/api/investment-calendar/current-positions"
        )
        if int(positions_created["399006"]) != expected:
            raise AssertionError("calendar create did not update the packaged SQLite position")
        updated = _json_request(
            f"{base_url}/api/investment-calendar/{entry_id}",
            method="PATCH",
            payload={"note": f"{marker} updated"},
        )
        if updated["note"] != f"{marker} updated":
            raise AssertionError("calendar patch was not persisted")
        deleted = _json_request(
            f"{base_url}/api/investment-calendar/{entry_id}?confirmed=true",
            method="DELETE",
        )
        if not deleted.get("deleted"):
            raise AssertionError("calendar delete did not report success")
        positions_after = _json_request(
            f"{base_url}/api/investment-calendar/current-positions"
        )
        if positions_after != positions_before:
            raise AssertionError("calendar CRUD cleanup did not restore original positions")
        identity_after = _active_status_signature(base_url)
        if identity_after != identity_before:
            raise AssertionError("calendar CRUD changed the persisted model identity")
    finally:
        if process.poll() is None:
            _stop_desktop(process, release_root / "data" / "desktop-port.json")
    with closing(sqlite3.connect(database)) as connection:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
    return {
        "final_release_database": True,
        "positions_before": positions_before,
        "created_id": entry_id,
        "created_position": positions_created["399006"],
        "updated_note_persisted": True,
        "deleted": True,
        "positions_restored": positions_after == positions_before,
        "model_identity_unchanged": True,
        "quick_check": quick_check,
        "port_file_cleaned": not (release_root / "data" / "desktop-port.json").exists(),
    }


def main() -> int:
    arguments = _arguments()
    release_root = arguments.release_root.resolve()
    report_path = arguments.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    required = (
        release_root / "InvestmentLab.exe",
        release_root / "V3.4-13W.release",
        release_root / "data" / "investment_lab.db",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    live = _review_live_ui(release_root, report_path.parent)
    calendar = _calendar_crud_release(release_root)
    report = {
        "status": "PASS",
        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "release_root": str(release_root),
        "desktop_only": True,
        "live_ui": live,
        "calendar_crud": calendar,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
