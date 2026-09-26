import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const { chromium } = require("../frontend/node_modules/playwright");

const baseUrl = process.argv[2] ?? "http://127.0.0.1:8765/";
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1500, height: 960 }, deviceScaleFactor: 1 });
const consoleErrors = [];
page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
});

await page.goto(baseUrl, { waitUntil: "networkidle", timeout: 120_000 });
const chart = page.getByTestId("unified-market-chart");
await chart.waitFor({ state: "visible", timeout: 60_000 });
await chart.locator(".unified-chart-canvas canvas").waitFor({ timeout: 60_000 });
await chart.screenshot({
  path: fileURLToPath(new URL("../frontend/reports/v37-daily-weekly-shared.png", import.meta.url)),
});
const calendar = page.locator(".position-calendar");
await calendar.scrollIntoViewIfNeeded();
await calendar.screenshot({
  path: fileURLToPath(new URL("../frontend/reports/v37-position-calendar.png", import.meta.url)),
});

const chartBox = await chart.boundingBox();
const canvasBox = await chart.locator(".unified-chart-canvas canvas").boundingBox();
const dailyBox = await chart.locator(".daily-region-label").boundingBox();
const weeklyBox = await chart.locator(".weekly-region-label").boundingBox();
console.log(JSON.stringify({
  chartState: await chart.getAttribute("data-chart-state"),
  weeklyCandleWidth: await chart.getAttribute("data-weekly-candle-width"),
  weeklyVolumeWidth: await chart.getAttribute("data-weekly-volume-width"),
  chartBox,
  canvasBox,
  dailyLabelVisible: Boolean(dailyBox),
  weeklyLabelVisible: Boolean(weeklyBox),
  marketCardCount: await page.locator(".market-switch > button").count(),
  legacyIndexCardCount: await page.getByRole("button", { name: /创业板指数/ }).count(),
  separateUniversePanelCount: await page.locator("details.universe-panel").count(),
  totalPosition: await page.getByTestId("total-position").textContent(),
  bothRegionsInsideChart: Boolean(
    chartBox && dailyBox && weeklyBox
    && dailyBox.y >= chartBox.y
    && weeklyBox.y + weeklyBox.height <= chartBox.y + chartBox.height
  ),
  consoleErrors,
}, null, 2));

await browser.close();
