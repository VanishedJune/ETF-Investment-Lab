"""Native-Python Playwright visual and interaction review for V3.2."""

from __future__ import annotations

from pathlib import Path
import os

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
BASE_URL = os.environ.get("V32_BASE_URL", "http://127.0.0.1:8765")

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1)
    console_errors: list[str] = []
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
    page.goto(BASE_URL, wait_until="networkidle")

    assert page.get_by_role("heading", name="双市场周线研究台").is_visible()
    assert page.get_by_test_id("v32-train").inner_text() == "训练模型"
    assert page.get_by_test_id("v32-analyze").inner_text() == "数据分析"
    page.wait_for_selector('[data-testid="v32-curve-399006"] canvas', timeout=20_000)
    page.wait_for_selector('[data-testid="v32-curve-NDX"] canvas', timeout=20_000)
    page.wait_for_selector('[data-testid="v32-path-chart"] canvas', timeout=20_000)
    assert page.get_by_test_id("v32-curve-399006").locator("canvas").count() == 1
    assert page.get_by_test_id("v32-curve-NDX").locator("canvas").count() == 1
    assert page.get_by_test_id("v32-path-chart").locator("canvas").count() == 1
    page.screenshot(path=str(REPORTS / "v32-visual-desktop.png"), full_page=True)

    page.get_by_role("button", name="美股 / NASDAQ 纳斯达克100 NDX").click()
    page.wait_for_load_state("networkidle")
    assert page.get_by_test_id("timeframe-weekly").get_attribute("aria-pressed") == "true"
    assert page.get_by_test_id("v32-curve-NDX").locator("canvas").count() == 1

    mobile = browser.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=1)
    mobile.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
    mobile.goto(BASE_URL, wait_until="networkidle")
    assert mobile.get_by_test_id("v32-train").is_visible()
    assert mobile.get_by_test_id("v32-analyze").is_visible()
    mobile.wait_for_function(
        "document.querySelector('[data-testid=data-cutoff]')?.textContent?.trim() !== '—'",
        timeout=30_000,
    )
    mobile.wait_for_selector('[data-testid="v32-curve-399006"] canvas', timeout=30_000)
    mobile.wait_for_selector('[data-testid="v32-curve-NDX"] canvas', timeout=30_000)
    mobile.screenshot(path=str(REPORTS / "v32-visual-mobile.png"), full_page=True)
    mobile.close()
    page.close()
    browser.close()

assert not console_errors, console_errors
print("V3.2 visual review passed")
