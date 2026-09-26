import { expect, test } from "@playwright/test";

test("旧13周训练预测界面已由纯行情面板替代", async ({ page }) => {
  await page.goto("/v34-model");
  await page.waitForLoadState("networkidle");
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByText(/13周预测|P10|P50|P90/)).toHaveCount(0);
  await expect(page.getByRole("button", { name: "训练模型" })).toHaveCount(0);
  await expect(page.getByTestId("unified-market-chart")).toHaveAttribute("data-chart-state", "rendered");
});

test("当前共享图保留日周独立区域与可视区间自动缩放", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  const chart = page.getByTestId("unified-market-chart");
  await expect(chart).toHaveAttribute("data-chart-mode", "daily-weekly-shared");
  await expect(chart.locator(".daily-region-label")).toBeVisible();
  await expect(chart.locator(".weekly-region-label")).toBeVisible();
  await expect(chart.locator("canvas")).toHaveCount(1);
  await expect(page.getByRole("button", { name: "重置共享缩放" })).toBeVisible();
});
