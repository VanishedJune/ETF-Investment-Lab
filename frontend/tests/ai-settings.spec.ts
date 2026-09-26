import { expect, test } from "@playwright/test";

test("旧 AI 设置地址重定向到纯行情主页", async ({ page }) => {
  await page.goto("/ai-settings");
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("heading", { name: "ETF 行情面板" })).toBeVisible();
});

test("纯行情版不展示 API Key 或模型配置输入", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel(/API Key/i)).toHaveCount(0);
  await expect(page.getByText("DeepSeek", { exact: true })).toHaveCount(0);
});

test("纯行情版不展示 AI 分析入口", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: /AI 分析/ })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "数据分析" })).toHaveCount(0);
});

test("纯行情版不展示行情对话输入区", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByPlaceholder(/提问|对话|发送给 AI/)).toHaveCount(0);
  await expect(page.getByRole("button", { name: /发送/ })).toHaveCount(0);
});

test("浏览行情不会发起 AI 或大模型请求", async ({ page }) => {
  const forbidden: string[] = [];
  page.on("request", (request) => {
    if (/deepseek|\/api\/v37\/ai|\/api\/ai\//i.test(request.url())) {
      forbidden.push(request.url());
    }
  });
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  await page.getByRole("button", { name: /广发黄金ETF/ }).click();
  await expect(page.getByTestId("unified-market-chart")).toHaveAttribute("data-chart-state", "rendered");
  expect(forbidden).toEqual([]);
});
