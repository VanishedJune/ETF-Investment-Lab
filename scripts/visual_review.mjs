import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const { chromium } = require("../frontend/node_modules/playwright");

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 1200 }, deviceScaleFactor: 1 });

await page.goto("http://127.0.0.1:8765/");
await page.waitForLoadState("networkidle");
await page.getByTestId("market-chart").locator("canvas").waitFor();
await page.getByTestId("iteration-chart").locator("canvas").waitFor();
await page.screenshot({
  path: fileURLToPath(new URL("../frontend/reports/dual-market-workbench.png", import.meta.url)),
  fullPage: true,
});

await browser.close();
