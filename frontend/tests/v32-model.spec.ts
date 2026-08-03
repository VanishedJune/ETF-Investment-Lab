import { expect, test } from "@playwright/test";

const status = {
  "399006": {
    is_training: false,
    bootstrapped: true,
    iteration_count: 511,
    last_training_week_key: "2026-W31",
    last_successful_training_at: "2026-08-01T03:00:00Z",
    next_training_eligible_at: "2026-08-08T03:00:00Z",
    champion_weekly_version: "CYB_WEEKLY_V3.2.31",
    champion_daily_version: "CYB_DAILY_CORRECTOR_V3.2.31",
    latest_run: null,
  },
  NDX: {
    is_training: false,
    bootstrapped: true,
    iteration_count: 522,
    last_training_week_key: "2026-W30",
    last_successful_training_at: "2026-08-01T03:05:00Z",
    next_training_eligible_at: "2026-08-08T03:05:00Z",
    champion_weekly_version: "NDX_WEEKLY_V3.2.24",
    champion_daily_version: "NDX_DAILY_CORRECTOR_V3.2.24",
    latest_run: null,
  },
};

function curve(market: "399006" | "NDX") {
  return {
    market,
    points: Array.from({ length: 60 }, (_, index) => ({
      iteration: index + 1,
      parent_iteration: index === 0 ? null : index,
      week_key: `2025-W${String((index % 52) + 1).padStart(2, "0")}`,
      cutoff_date: "2026-07-31",
      maturity_status: index >= 55 ? "pending" : "full",
      composite_loss: index >= 55 ? null : 1.2 - index * 0.004,
      path_error: index >= 55 ? null : 1.1,
      terminal_return_error: index >= 55 ? null : 0.8,
      direction_score: index >= 55 ? null : 0.2,
      interval_coverage: index >= 55 ? null : 78,
      high_week_error: index >= 55 ? null : 2,
      low_week_error: index >= 55 ? null : 3,
      high_low_deviation_days: index >= 55 ? null : 12.5,
      rolling_20_loss: index >= 55 ? null : 1.1 - index * 0.003,
      rolling_52_loss: index >= 55 ? null : 1.15 - index * 0.002,
      rolling_20_deviation_days: index >= 55 ? null : 10.5,
      rolling_52_deviation_days: index >= 55 ? null : 12.0,
      champion_version: `${market}-V3.2.4`,
      challenger_version: index > 51 ? `${market}-CANDIDATE-${index + 1}` : null,
      promoted: index === 52,
      rejection_reason: index > 51 ? "样本外损失未达到晋级门槛" : null,
    })),
  };
}

const analysis = {
  implementation_revision: "V3.2.0",
  run_id: "V32-ANALYSIS-399006-TEST",
  market: "399006",
  training_mutated: false,
  model: { weekly: { version: "CYB_WEEKLY_V3.2.31", iteration_number: 511 }, daily: { version: "CYB_DAILY_CORRECTOR_V3.2.31", window: 100 } },
  freshness: { price_data_as_of: "2026-07-31" },
  weekly: { market_state: "低估反转", state: "中度看多", features: { valuation_percentile: 12, macd_spread: 0.002 } },
  daily: { state: "短线节奏确认", features: { rsi: 54 } },
  path: {
    up_probability: 78,
    sideways_probability: 12,
    down_probability: 10,
    points: Array.from({ length: 13 }, (_, index) => ({
      horizon_week: index + 1,
      p10_cumulative_return: index - 6,
      p50_cumulative_return: index * 0.8,
      p90_cumulative_return: index * 1.4 + 4,
      expected_cumulative_return: index * 0.9,
    })),
  },
  position: { confirmed_position: 20 },
  advice: {
    current_position: 20,
    final_target_position: 80,
    direction: "increase",
    fund_etf_ratio: "7:3",
    batches: [20, 15, 15, 10].map((amount, index) => ({
      sequence: index + 1,
      direction: "increase",
      position_points: amount,
      execution_window: ["2026-08-03", "2026-08-05"],
      trigger: "DIF不低于DEA",
      invalidation: "跌破20日均线",
      position_after: [40, 55, 70, 80][index],
    })),
  },
  data_snapshot_id: "V32-DS-TEST",
  forecast_id: 1,
};

test("V3.2 remains an immutable 399006 and NDX audit baseline", async ({ page }) => {
  await page.route("**/api/v3.2/training/status", (route) => route.fulfill({ json: status }));
  await page.route("**/api/v3.2/training/399006/curve", (route) => route.fulfill({ json: curve("399006") }));
  await page.route("**/api/v3.2/training/NDX/curve", (route) => route.fulfill({ json: curve("NDX") }));
  await page.route("**/api/v3.2/analysis/399006/latest", (route) =>
    route.fulfill({ json: { id: analysis.run_id, market: "399006", status: "completed", current_stage: "已完成", stages: [], result: analysis } }));

  await page.goto("/");
  await page.waitForLoadState("networkidle");
  await page.getByText("V3.2只读历史审计", { exact: true }).click();
  await expect(page.getByRole("heading", { name: "V3.2只读历史对照" })).toBeVisible();
  await expect(page.getByTestId("v32-readonly-badge")).toContainText("READ ONLY");
  await expect(page.getByText("纳斯达克100底层指数（V3.2审计）")).toBeVisible();
  await expect(page.getByTestId("v32-train")).toHaveCount(0);
  await expect(page.getByTestId("v32-analyze")).toHaveCount(0);
  await expect(page.getByTestId("v32-curve-399006").locator("canvas")).toBeVisible();
  await expect(page.getByTestId("v32-curve-NDX").locator("canvas")).toBeVisible();
  await expect(page.getByTestId("v32-analysis-result")).toContainText("低估反转");
  await expect(page.getByTestId("v32-analysis-result")).toContainText("20%");
  await expect(page.getByTestId("v32-analysis-result")).toContainText("80%");
  await expect(page.getByTestId("v32-analysis-result")).toContainText("7:3");
  await expect(page.getByTestId("v32-batches").locator("li")).toHaveCount(4);
  await expect(page.getByTestId("v32-batches")).toContainText("买入 20%");
  await expect(page.getByTestId("v32-path-chart").locator("canvas")).toBeVisible();
  await page.screenshot({ path: "../reports/v32-readonly-audit.png", fullPage: true });
});
