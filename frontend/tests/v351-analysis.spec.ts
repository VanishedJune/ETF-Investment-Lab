import { expect, test } from "@playwright/test";

test("旧8周分析地址重定向且不显示预测K线", async ({ page }) => {
  await page.goto("/v351-analysis");
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByText(/8W 预测|预测K线/)).toHaveCount(0);
});

test("纯行情面板不会留下空白的仓位建议批次", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("batch-plan")).toHaveCount(0);
  await expect(page.getByText(/原因码|等待确认/)).toHaveCount(0);
  await expect(page.locator(".position-calendar")).toBeVisible();
});

test("切换ETF只更新对应行情和技术指标", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  await page.getByRole("button", { name: /医药ETF易方达/ }).click();
  await expect(page.getByRole("heading", { name: /医药ETF易方达 · 日线 \/ 周线指标/ })).toBeVisible();
  await expect(page.getByText("数据分析", { exact: true })).toHaveCount(0);
  await expect(page.getByTestId("unified-market-chart")).toHaveAttribute("data-chart-state", "rendered");
});
