import { test, expect } from "@playwright/test";

test.describe("V3.7 data-only desktop panel", () => {
  test("loads the expanded ETF universe and daily/weekly indicators", async ({ page, request }) => {
    const instruments = await request.get("/api/instruments");
    expect(instruments.ok()).toBeTruthy();
    expect((await instruments.json())).toHaveLength(9);

    await page.goto("/");
    await expect(page.getByRole("heading", { name: "ETF 行情面板" })).toBeVisible();
    await expect(page.getByText("日 K / 周 K / 月 K 同步查看")).toBeVisible();
    await expect(page.getByText("DIF", { exact: true }).first()).toBeVisible();
    await expect(page.getByText("DEA", { exact: true }).first()).toBeVisible();

    await expect(page.getByText("日 K + 周 K 共用横轴", { exact: true })).toBeVisible();
    await expect(page.getByTestId("timeframe-monthly")).toHaveAttribute("aria-pressed", "false");
    const chart = page.getByTestId("unified-market-chart");
    await expect(chart).toBeVisible();
    await expect(chart).toHaveAttribute("data-chart-state", "rendered");
    await expect(chart).toHaveAttribute("data-weekly-candle-width", "18");
    await expect(chart).toHaveAttribute("data-weekly-volume-width", "18");
    await expect(chart.locator(".daily-region-label")).toBeVisible();
    await expect(chart.locator(".weekly-region-label")).toBeVisible();
    const canvas = chart.locator(".unified-chart-canvas canvas");
    await expect(canvas).toHaveCount(1);
    const canvasBox = await canvas.boundingBox();
    expect(canvasBox?.width ?? 0).toBeGreaterThan(0);
    expect(canvasBox?.height ?? 0).toBeGreaterThan(0);
    expect(canvasBox?.height ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(900);

    await expect(page.locator(".market-switch > button")).toHaveCount(8);
    await expect(page.getByRole("button", { name: /创业板指数/ })).toHaveCount(0);
    await expect(page.locator("details.universe-panel")).toHaveCount(0);
    await expect(page.getByRole("button", { name: /创新药ETF东财/ })).toBeVisible();
    await expect(page.getByRole("button", { name: /创业板ETF易方达/ })).toBeVisible();
    await expect(page.getByRole("button", { name: /稀土ETF嘉实/ })).toBeVisible();
  });

  test("does not expose legacy AI, training, or forecast routes", async ({ page }) => {
    await page.goto("/ai-settings");
    await expect(page).toHaveURL(/\/$/);
    await expect(page.getByText("数据分析")).toHaveCount(0);
    await page.goto("/v37-analysis");
    await expect(page).toHaveURL(/\/$/);
    await expect(page.getByText("训练模型")).toHaveCount(0);
  });
});
