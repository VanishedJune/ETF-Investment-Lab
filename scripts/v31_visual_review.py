from __future__ import annotations

from pathlib import Path
import re

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import expect, sync_playwright


root = Path(__file__).resolve().parents[1]
output = root / "reports" / "v31"
output.mkdir(parents=True, exist_ok=True)

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        headless=True,
        executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    )
    page = browser.new_page(viewport={"width": 1520, "height": 980}, device_scale_factor=1)
    console_errors: list[str] = []
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
    page.goto("http://127.0.0.1:8765", wait_until="domcontentloaded", timeout=60_000)
    try:
        page.wait_for_load_state("networkidle", timeout=30_000)
    except PlaywrightTimeoutError:
        # Some audit endpoints intentionally remain busy while the shell is
        # already interactive. Concrete UI readiness is asserted below.
        pass
    page.get_by_role("heading", name="双市场周线研究台").wait_for()
    page.get_by_role("button", name=re.compile("纳斯达克100")).click()
    analyze_button = page.get_by_test_id("v31-analyze")
    previous_run = page.evaluate(
        "async () => await (await fetch('/api/v3.1/analysis/NDX/latest')).json()"
    )
    analyze_button.click()
    expect(analyze_button).to_be_disabled()
    expect(analyze_button).to_be_enabled(timeout=120_000)
    page.get_by_test_id("v31-path-chart").wait_for(state="visible", timeout=120_000)
    page.wait_for_function(
        "document.body.innerText.includes('未来13周累计收益路径') && document.body.innerText.includes('当前默认为清仓状态（0%）')",
        timeout=120_000,
    )
    latest_run = page.evaluate(
        "async () => await (await fetch('/api/v3.1/analysis/NDX/latest')).json()"
    )
    assert latest_run["id"] != previous_run.get("id")
    expect(page.get_by_test_id("data-cutoff")).to_have_text(
        latest_run["result"]["freshness"]["price_data_as_of"], timeout=30_000
    )
    page.screenshot(path=str(output / "v31-ndx-success.png"), full_page=True)

    page.get_by_role("button", name=re.compile("创业板指数")).click()
    previous_run = page.evaluate(
        "async () => await (await fetch('/api/v3.1/analysis/399006/latest')).json()"
    )
    analyze_button.click()
    expect(analyze_button).to_be_disabled()
    expect(analyze_button).to_be_enabled(timeout=120_000)
    page.get_by_test_id("v31-failure").wait_for(state="visible", timeout=120_000)
    assert "STALE_DAILY_DATA" in page.get_by_test_id("v31-failure").inner_text()
    latest_run = page.evaluate(
        "async () => await (await fetch('/api/v3.1/analysis/399006/latest')).json()"
    )
    assert latest_run["id"] != previous_run.get("id")
    expect(page.get_by_test_id("data-cutoff")).to_have_text(
        latest_run["result"]["last_successful_data_date"], timeout=30_000
    )
    page.screenshot(path=str(output / "v31-cyb-stale-gate.png"), full_page=True)

    unexpected = [message for message in console_errors if "favicon" not in message.lower()]
    if unexpected:
        raise AssertionError(f"browser console errors: {unexpected}")
    browser.close()

print("V3.1 Playwright review passed")
