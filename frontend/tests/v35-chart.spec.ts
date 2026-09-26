import { expect, test } from "@playwright/test";

test("共享图例与重置缩放位于绘图区上方", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  const chart = page.getByTestId("unified-market-chart");
  const rail = chart.locator(".chart-rail");
  const canvas = chart.locator(".unified-chart-canvas canvas");
  await expect(rail).toBeVisible();
  await expect(chart.getByTestId("ma5-label")).toBeVisible();
  await expect(chart.getByTestId("ma10-label")).toBeVisible();
  await expect(chart.getByTestId("ma20-label")).toBeVisible();
  await expect(page.getByRole("button", { name: "重置共享缩放" })).toBeVisible();
  const railBox = await rail.boundingBox();
  const canvasBox = await canvas.boundingBox();
  expect(railBox).not.toBeNull();
  expect(canvasBox).not.toBeNull();
  expect(railBox!.y + railBox!.height).toBeLessThanOrEqual(canvasBox!.y);
});

test("DIF图例切换、悬停详情与重置缩放可交互", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  const dif = page.getByTestId("dif-label");
  await expect(dif).toHaveClass(/selected/);
  await dif.click();
  await expect(dif).not.toHaveClass(/selected/);
  await dif.click();
  await expect(dif).toHaveClass(/selected/);
  const canvas = page.getByTestId("unified-market-chart").locator(".unified-chart-canvas canvas");
  await canvas.hover({ position: { x: 140, y: 220 } });
  await expect(page.getByTestId("market-point-details")).toContainText("收盘");
  await page.getByRole("button", { name: "重置共享缩放" }).click();
});
