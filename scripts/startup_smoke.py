from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        headless=True,
        executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    )
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    console_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.goto("http://127.0.0.1:8765", wait_until="domcontentloaded", timeout=60_000)
    try:
        page.wait_for_load_state("networkidle", timeout=30_000)
    except PlaywrightTimeoutError:
        pass
    page.get_by_role("heading", name="双市场周线研究台").wait_for(timeout=60_000)
    unexpected = [item for item in console_errors if "favicon" not in item.lower()]
    if unexpected:
        raise AssertionError(f"browser console errors: {unexpected}")
    browser.close()

print("One-click startup browser smoke passed")
