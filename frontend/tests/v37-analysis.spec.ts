import { expect, test } from "@playwright/test";

test("V3.7多模型分析地址重定向且Local/AI/Fusion卡片均不加载", async ({ page }) => {
  const analysisRequests: string[] = [];
  page.on("request", (request) => {
    if (/\/api\/v37\/(analysis|fusion|ai)/.test(request.url())) {
      analysisRequests.push(request.url());
    }
  });
  await page.goto("/v37-analysis");
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByText("Local", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Fusion", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "ETF 行情面板" })).toBeVisible();
  expect(analysisRequests).toEqual([]);
});
