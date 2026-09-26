import { expect, test } from "@playwright/test";

test("V3.2模型页面不再进入当前纯行情桌面", async ({ page }) => {
  const modelRequests: string[] = [];
  page.on("request", (request) => {
    if (request.url().includes("/api/v3.2/")) modelRequests.push(request.url());
  });
  await page.goto("/v32-model");
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("heading", { name: "ETF 行情面板" })).toBeVisible();
  await expect(page.getByText("训练模型", { exact: true })).toHaveCount(0);
  expect(modelRequests).toEqual([]);
});
